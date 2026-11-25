"""Gold source : la version de référence du dossier patient, celle que la reconstruction doit retrouver.

Après le préfiltre déterministe, certaines conditions restent ambiguës (« Stress »,
« Received higher education »…). On demande au LLM de trier la liste candidate de
chaque patient. Pour limiter le risque, il ne peut que choisir des index dans la
liste, jamais ajouter ou reformuler un élément, et on garde tout en cas de doute.
"""

from __future__ import annotations

import hashlib
import json
from typing import List

import pandas as pd

from config import settings
from src.llm import get_llm, load_json_lenient
from src.utils import list_json, read_json, safe_read_json, write_json

FILTER_SCHEMA = {
    "keep_indexes": ["int"],
    "drop_indexes": ["int"],
    "reasons": {"<index>": "medical|sdoh|admin|legal|behavior|other"},
}


def filter_prompt(candidates: List[dict]) -> str:
    items_txt = "\n".join(
        f"{i}. [{c.get('system') or ''}|{c.get('code') or ''}] {c.get('display') or ''}"
        for i, c in enumerate(candidates)
    )
    return f"""You are validating patient FHIR Conditions from synthetic data.

Goal:
Keep ONLY true medical conditions (diagnoses, diseases, symptoms, disorders, injuries).
Drop social determinants (employment, housing, education, violence exposure), legal/admin items, and similar non-medical content.

Rules:
- DO NOT invent items.
- Decide only by selecting indexes from the list.
- If unsure whether something is medical, KEEP it.

Return ONLY valid JSON (no markdown, no commentary).
The first character must be '{{' and the last must be '}}'.
Use exactly this schema:
{json.dumps(FILTER_SCHEMA, indent=2)}

Condition candidates:
{items_txt}

JSON:"""


def fix_prompt(raw: str) -> str:
    return f"""Return ONLY valid JSON matching this schema:
{json.dumps(FILTER_SCHEMA, indent=2)}

Convert the following text to valid JSON. Do not add commentary.
Text:
{raw}

JSON:"""


def _parse_keep_indexes(obj, n_candidates: int) -> List[int]:
    if not isinstance(obj, dict) or not isinstance(obj.get("keep_indexes"), list):
        return []
    keep = []
    for x in obj["keep_indexes"]:
        try:
            ix = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= ix < n_candidates:
            keep.append(ix)
    return sorted(set(keep))


def llm_keep_condition_indexes(patient_id: str, candidates: List[dict], llm=None) -> List[int]:
    """Renvoie les index des conditions à garder. Si le LLM ne répond rien d'exploitable, on garde tout."""
    llm = llm or get_llm()
    keep_all = list(range(len(candidates)))

    # Le nom du cache dépend des candidats : si le préfiltre change, le tri est refait.
    key_payload = {
        "patient_id": patient_id,
        "candidates": [{"system": c.get("system"), "code": c.get("code"), "display": c.get("display")} for c in candidates],
    }
    cache_id = hashlib.sha1(json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_path = settings.FILTER_CACHE_DIR / f"llm_keepidx_{patient_id}_{cache_id}.json"

    cached = safe_read_json(cache_path)
    if cached and isinstance(cached.get("keep_indexes"), list):
        return [int(x) for x in cached["keep_indexes"] if str(x).isdigit()]

    if not (settings.USE_LLM_PATIENT_FILTER and llm.available()):
        write_json(cache_path, {"keep_indexes": keep_all, "mode": "no_llm"})
        return keep_all

    raw = llm.generate(
        filter_prompt(candidates),
        max_new_tokens=settings.LLM_FILTER_MAX_NEW_TOKENS,
        temperature=settings.LLM_FILTER_TEMPERATURE,
        do_sample=settings.LLM_FILTER_DO_SAMPLE,
    )
    obj = load_json_lenient(raw)

    if obj is None and settings.LLM_FILTER_RETRIES > 0:
        fixed = llm.generate(fix_prompt(raw), max_new_tokens=settings.LLM_FILTER_MAX_NEW_TOKENS,
                             temperature=0.0, do_sample=False)
        obj = load_json_lenient(fixed)

    # Une liste vide est presque toujours un échec du modèle, pas une décision : on garde tout.
    keep = _parse_keep_indexes(obj, len(candidates)) or keep_all
    write_json(cache_path, {"keep_indexes": keep, "raw": raw[:2000]})
    return keep


def apply_llm_filter_to_sample(patients_in: List[dict], llm=None) -> List[dict]:
    out = []
    for p in patients_in:
        cands = p.get("conditions_candidates") or []
        keep_idx = llm_keep_condition_indexes(p["patient_id"], cands, llm=llm)
        kept = [cands[i].get("display") for i in keep_idx if i < len(cands) and cands[i].get("display")]
        out.append({
            **p,
            "conditions_llm": kept,
            "conditions": list(dict.fromkeys(kept))[: settings.MAX_CONDITIONS_PER_PATIENT],
        })
    return out


def build_gold(patients_prefilter: List[dict], llm=None) -> List[dict]:
    patients = apply_llm_filter_to_sample(patients_prefilter, llm=llm)
    save_gold(patients)
    return patients


# =========================
# Lecture / écriture
# =========================


def save_gold(patients: List[dict]) -> None:
    for p in patients:
        write_json(settings.GOLD_DIR / f'{p["patient_id"]}.json', p)


def load_gold_patients() -> List[dict]:
    """Relit le gold depuis le disque, pour reprendre le pipeline dans un autre notebook."""
    patients = [read_json(p) for p in list_json(settings.GOLD_DIR)]
    return [p for p in patients if isinstance(p, dict) and p.get("patient_id")]


def gold_dataframe(patients: List[dict]) -> pd.DataFrame:
    n = settings.PREVIEW_N
    rows = []
    for p in patients:
        row = {
            "patient_id": p["patient_id"],
            "full_name": p.get("full_name"),
            "birthDate": p.get("birthDate"),
            "gender": p.get("gender"),
        }
        if "note_style" in p:
            row["note_style"] = p.get("note_style")
            row["note_style_label"] = p.get("note_style_label")
        row.update({
            "n_conditions_candidates": len(p.get("conditions_candidates") or []),
            "n_conditions_kept": len(p.get("conditions_llm") or []),
            "conditions_preview": "; ".join((p.get("conditions") or [])[:n]),
            "n_medication_requests": len(p.get("medication_requests") or []),
            "medreq_preview": "; ".join((p.get("medication_requests") or [])[:n]),
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_cols = ["note_style", "patient_id"] if "note_style" in df.columns else ["patient_id"]
    return df.sort_values(sort_cols).reset_index(drop=True)
