"""Étape 4 : finaliser le `gold_source` avec un filtre LLM sur les conditions.

Repris de la cellule 4D) du notebook original. Le gold source est la
référence structurée utilisée pour générer les notes et pour le scoring
« source vs recon ».
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
    lines = []
    for i, c in enumerate(candidates):
        sys_ = c.get("system") or ""
        code = c.get("code") or ""
        disp = c.get("display") or ""
        lines.append(f"{i}. [{sys_}|{code}] {disp}")
    items_txt = "\n".join(lines) if lines else ""

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


def llm_keep_condition_indexes(patient_id: str, candidates: List[dict], llm=None) -> List[int]:
    """Renvoie les index à garder (conditions médicales uniquement). Cache disque. Ne lève jamais."""
    llm = llm or get_llm()
    settings.FILTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key_payload = {
        "patient_id": patient_id,
        "candidates": [{"system": c.get("system"), "code": c.get("code"), "display": c.get("display")} for c in candidates],
    }
    key_str = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
    cache_id = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    cache_path = settings.FILTER_CACHE_DIR / f"llm_keepidx_{patient_id}_{cache_id}.json"
    if cache_path.exists():
        obj = safe_read_json(cache_path)
        if obj and isinstance(obj.get("keep_indexes"), list):
            return [int(x) for x in obj["keep_indexes"] if str(x).isdigit()]

    # Sans LLM : on garde tout
    if not (settings.USE_LLM_PATIENT_FILTER and llm.available()):
        keep = list(range(len(candidates)))
        write_json(cache_path, {"keep_indexes": keep, "mode": "no_llm"})
        return keep

    raw = ""
    try:
        raw = llm.generate(
            filter_prompt(candidates),
            max_new_tokens=settings.LLM_FILTER_MAX_NEW_TOKENS,
            temperature=settings.LLM_FILTER_TEMPERATURE,
            do_sample=settings.LLM_FILTER_DO_SAMPLE,
        )
    except Exception:  # noqa: BLE001
        keep = list(range(len(candidates)))
        write_json(cache_path, {"keep_indexes": keep, "mode": "llm_error"})
        return keep

    obj = load_json_lenient(raw)

    # Une relance avec un prompt de correction si besoin
    if obj is None and settings.LLM_FILTER_RETRIES > 0:
        fix_prompt = f"""Return ONLY valid JSON matching this schema:
{json.dumps(FILTER_SCHEMA, indent=2)}

Convert the following text to valid JSON. Do not add commentary.
Text:
{raw}

JSON:"""
        try:
            fixed = llm.generate(
                fix_prompt,
                max_new_tokens=settings.LLM_FILTER_MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
            )
            obj = load_json_lenient(fixed)
        except Exception:  # noqa: BLE001
            obj = None

    keep_indexes: List[int] = []
    if isinstance(obj, dict) and isinstance(obj.get("keep_indexes"), list):
        for x in obj["keep_indexes"]:
            try:
                ix = int(x)
                if 0 <= ix < len(candidates):
                    keep_indexes.append(ix)
            except Exception:  # noqa: BLE001
                continue

    # Sécurité : si vide ou invalide, on garde tout
    if not keep_indexes:
        keep_indexes = list(range(len(candidates)))

    keep_indexes = sorted(set(keep_indexes))
    write_json(cache_path, {"keep_indexes": keep_indexes, "raw": raw[:2000]})
    return keep_indexes


def apply_llm_filter_to_sample(patients_in: List[dict], llm=None) -> List[dict]:
    out = []
    for p in patients_in:
        pid = p["patient_id"]
        cands = p.get("conditions_candidates") or []
        keep_idx = llm_keep_condition_indexes(pid, cands, llm=llm)
        kept = [cands[i] for i in keep_idx if i < len(cands)]
        kept_disp = [x.get("display") for x in kept if x.get("display")]

        # Enregistrement patient final pour les étapes suivantes
        p2 = dict(p)
        p2["conditions_llm"] = kept_disp
        p2["conditions"] = list(dict.fromkeys(kept_disp))[: settings.MAX_CONDITIONS_PER_PATIENT]
        out.append(p2)
    return out


# =========================
# Lecture / écriture du gold
# =========================


def save_gold(patients: List[dict]) -> None:
    settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for p in patients:
        write_json(settings.GOLD_DIR / f'{p["patient_id"]}.json', p)


def load_gold_patients() -> List[dict]:
    """Relit le gold depuis data/gold_standard/ (permet de reprendre le pipeline dans un autre notebook)."""
    patients = []
    for p in list_json(settings.GOLD_DIR):
        obj = read_json(p)
        if isinstance(obj, dict) and obj.get("patient_id"):
            patients.append(obj)
    return patients


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


def build_gold(patients_prefilter: List[dict], llm=None) -> List[dict]:
    """Applique le filtre LLM puis écrit le gold sur disque."""
    patients = apply_llm_filter_to_sample(patients_prefilter, llm=llm)
    save_gold(patients)
    return patients
