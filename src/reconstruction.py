"""Étape 6 : reconstruction structurée depuis `note_text`.

Repris de la cellule 6) du notebook original.

Trois vues de la reconstruction :
- `recon_raw`     : extraction brute renvoyée par le LLM
- `recon_clean`   : reconstruction nettoyée, utilisée pour les scores
- `recon_clipped` : vue stricte (overlap exact avec le gold), debug seulement
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from config import settings
from src.llm import LLMResult, extract_first_json, get_llm, repair_json, try_load
from src.utils import norm_text, read_json, write_json


class Extracted(BaseModel):
    # Extraction brute du LLM (scoring/debug uniquement).
    # On ne fait PAS confiance aux données démographiques pour la reconstruction finale
    # afin d'éviter les inversions entre patients.
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    medication_requests: List[str] = Field(default_factory=list)


SCHEMA = {
    "patient_name": "string or null",
    "date_of_birth": "YYYY-MM-DD or null",
    "gender": "male|female|other|unknown or null",
    "conditions": ["string"],
    "medication_requests": ["string"],
}

EMPTY_EXTRACTION = {"patient_name": None, "date_of_birth": None, "gender": None, "conditions": [], "medication_requests": []}


def extraction_prompt(note_text: str) -> str:
    example_output = {
        "patient_name": "John Smith",
        "date_of_birth": "1980-04-12",
        "gender": "male",
        "conditions": [
            "Hypertension",
            "Type 2 diabetes mellitus",
        ],
        "medication_requests": [
            "Lisinopril 10 MG Oral Tablet",
            "Metformin 500 MG Oral Tablet",
        ],
    }

    return f"""You are a clinical information extraction engine.

Your task is to read ONE clinical note and return EXACTLY ONE valid JSON object.

Return rules:
- Return JSON only.
- No markdown.
- No code fences.
- No explanation.
- No comments.
- No extra keys.
- The first character must be '{{' and the last character must be '}}'.

You must return exactly these keys:
- patient_name
- date_of_birth
- gender
- conditions
- medication_requests

Extraction rules:
- Extract only information explicitly written in the note.
- Do not infer, guess, interpret, or add medical knowledge.
- Do not invent diagnoses or medications.
- For conditions and medication_requests, copy the wording from the note as closely as possible.
- Do not rewrite items into synonyms unless the note itself uses that wording.
- Each list element must contain exactly one item, not a sentence.
- Split combined items into separate list elements when clearly separable.
- Remove duplicates.
- Preserve medication strength, form, and dosage only if explicitly present in the note.

Field rules:
- patient_name: the patient full name if explicitly present, otherwise null.
- date_of_birth: only YYYY-MM-DD if explicitly present, otherwise null.
- gender: must be exactly one of ["male", "female", "other", "unknown"] or null.
- conditions: include only diagnoses, disorders, chronic diseases, or medical problems explicitly stated.
- medication_requests: include only medications explicitly prescribed, active, continued, or listed as current medications.

Do NOT put these in conditions unless the note explicitly states they are diagnoses:
- symptoms alone
- procedures
- laboratory tests
- imaging findings alone
- allergies
- vaccinations
- family history
- social history
- employment
- education
- stress
- risk activity
- insurance or administrative details

Do NOT put these in medication_requests:
- procedures
- tests
- follow-up instructions
- lifestyle advice
- allergies
- vaccines

Missing value rules:
- If a scalar field is missing, use null.
- If a list field is missing, use [].

Schema:
{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

Valid output example:
{json.dumps(example_output, ensure_ascii=False, indent=2)}

Clinical note:
<note>
{note_text}
</note>

JSON:"""


def fix_to_json(raw_text: str, llm=None) -> str:
    llm = llm or get_llm()
    fix_prompt = f"""Convert the text below into EXACTLY ONE valid JSON object.

Rules:
- Return JSON only.
- No markdown.
- No explanation.
- No extra keys.
- Do not invent information.
- Keep only information already present in the text.
- If a scalar field is missing, use null.
- If a list field is missing, use [].
- gender must be one of ["male", "female", "other", "unknown"] or null.
- date_of_birth must be YYYY-MM-DD or null.

Schema:
{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

Text:
<text>
{raw_text}
</text>

JSON:"""
    return llm.generate(
        fix_prompt,
        max_new_tokens=settings.RECON_FIX_JSON_MAX_NEW_TOKENS,
        temperature=settings.RECON_TEMPERATURE,
        do_sample=settings.RECON_DO_SAMPLE,
    )


def parse_llm_json(llm_text: str, retries: int = 1, llm=None) -> Optional[dict]:
    """Parse la sortie LLM ; si échec et `retries > 0`, demande au LLM de corriger le JSON."""
    js = extract_first_json(llm_text)
    if js:
        obj = try_load(js)
        if obj is not None:
            return obj
        obj = try_load(repair_json(js))
        if obj is not None:
            return obj
    if retries > 0:
        llm = llm or get_llm()
        if llm.available():
            fixed = fix_to_json(llm_text, llm=llm)
            return parse_llm_json(fixed, retries=retries - 1, llm=llm)
    return None


def ensure_list(x) -> List[str]:
    if isinstance(x, list):
        return x
    return []


def clean_extracted_item(v: str) -> str:
    s = str(v or "").strip()
    s = re.sub(r"^\s*[-*•]+\s*", "", s)
    s = re.sub(r"^\s*\d+[\.\)\-:]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n,;")
    return s


def is_noisy_candidate(k: str, filter_noisy: bool) -> bool:
    if not filter_noisy or not settings.PREFILTER_ENABLED:
        return False

    if not k:
        return True

    noisy_phrases = [norm_text(x) for x in (settings.PREFILTER_DROP_SUBSTRINGS or []) if x]
    strong_multiword = [x for x in noisy_phrases if " " in x]
    short_singletons = set([x for x in noisy_phrases if " " not in x])

    # Les expressions de plusieurs mots sont filtrées par sous-chaîne.
    if any(p in k for p in strong_multiword):
        return True

    # Les mots seuls ne sont filtrés que sur une entrée quasi identique.
    if k in short_singletons:
        return True

    return False


def postprocess_list(values: List[str], max_items: int, filter_noisy: bool) -> List[str]:
    out = []
    seen = set()

    for v in ensure_list(values):
        cleaned = clean_extracted_item(v)
        k = norm_text(cleaned)

        if not k:
            continue
        if len(k) <= 1:
            continue
        if len(k) > 160:
            continue
        if is_noisy_candidate(k, filter_noisy):
            continue
        if k in seen:
            continue

        seen.add(k)
        out.append(cleaned)

        if len(out) >= max_items:
            break

    return out


def clip_to_gold(values: List[str], gold_values: List[str]) -> List[str]:
    """Overlap exact normalisé avec la liste gold du patient. Debug seulement, pas utilisé pour le scoring."""
    allowed = set([norm_text(x) for x in (gold_values or []) if x])
    return [v for v in (values or []) if norm_text(v) in allowed]


# =========================
# Pipeline de reconstruction
# =========================


def reconstruct_one(pid: str, note_text: str, gold: dict, llm=None) -> Tuple[dict, dict, dict]:
    """Reconstruit un patient. Retourne (recon_raw, recon_clean, recon_clipped)."""
    llm = llm or get_llm()
    obj = None
    if llm.available():
        try:
            txt = llm.generate(
                extraction_prompt((note_text or "").strip()),
                max_new_tokens=settings.RECON_EXTRACTION_MAX_NEW_TOKENS,
                temperature=settings.RECON_TEMPERATURE,
                do_sample=settings.RECON_DO_SAMPLE,
            )
            obj = parse_llm_json(txt, retries=1, llm=llm)
        except Exception:  # noqa: BLE001
            obj = None

    if not isinstance(obj, dict):
        obj = dict(EMPTY_EXTRACTION)

    obj.setdefault("patient_name", None)
    obj.setdefault("date_of_birth", None)
    obj.setdefault("gender", None)
    obj["conditions"] = ensure_list(obj.get("conditions"))
    obj["medication_requests"] = ensure_list(obj.get("medication_requests"))

    gold = gold or {}

    cond_clean = postprocess_list(obj.get("conditions") or [], settings.MAX_CONDITIONS_PER_PATIENT, filter_noisy=True)
    medr_clean = postprocess_list(obj.get("medication_requests") or [], settings.MAX_MEDICATION_REQUESTS_PER_PATIENT, filter_noisy=False)

    cond_clipped = clip_to_gold(cond_clean, gold.get("conditions", []))
    medr_clipped = clip_to_gold(medr_clean, gold.get("medication_requests", []))

    recon_clean = {
        "note_used_for_extraction": (note_text or "").strip(),
        "patient_id": pid,
        "full_name": gold.get("full_name"),
        "birthDate": gold.get("birthDate"),
        "gender": gold.get("gender"),
        "conditions": cond_clean,
        "medication_requests": medr_clean,
    }
    recon_clipped = {
        "patient_id": pid,
        "full_name": gold.get("full_name"),
        "birthDate": gold.get("birthDate"),
        "gender": gold.get("gender"),
        "conditions": cond_clipped,
        "medication_requests": medr_clipped,
    }
    return obj, recon_clean, recon_clipped


def reconstruct_all(patients: List[dict], notes: Dict[str, LLMResult], llm=None) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """Reconstruit tous les patients et écrit les 3 vues dans data/processed/reconstruction/."""
    settings.RECON_DIR.mkdir(parents=True, exist_ok=True)
    gold_by_pid = {p["patient_id"]: p for p in patients}

    recon_raw: Dict[str, dict] = {}
    recon_clean: Dict[str, dict] = {}
    recon_clipped: Dict[str, dict] = {}

    for pid, res in notes.items():
        raw, clean, clipped = reconstruct_one(pid, res.note, gold_by_pid.get(pid, {}), llm=llm)
        recon_raw[pid], recon_clean[pid], recon_clipped[pid] = raw, clean, clipped
        write_json(settings.RECON_DIR / f"{pid}.raw.json", raw)
        write_json(settings.RECON_DIR / f"{pid}.json", clean)
        write_json(settings.RECON_DIR / f"{pid}.clipped.json", clipped)

    return recon_raw, recon_clean, recon_clipped


def load_reconstructions() -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """Relit les reconstructions depuis data/processed/reconstruction/."""
    recon_raw: Dict[str, dict] = {}
    recon_clean: Dict[str, dict] = {}
    recon_clipped: Dict[str, dict] = {}
    if not settings.RECON_DIR.exists():
        return recon_raw, recon_clean, recon_clipped
    for p in sorted(settings.RECON_DIR.glob("*.json")):
        if p.name.endswith(".raw.json") or p.name.endswith(".clipped.json"):
            continue
        pid = p.stem
        recon_clean[pid] = read_json(p)
        raw_path = settings.RECON_DIR / f"{pid}.raw.json"
        clipped_path = settings.RECON_DIR / f"{pid}.clipped.json"
        recon_raw[pid] = read_json(raw_path) if raw_path.exists() else dict(EMPTY_EXTRACTION)
        recon_clipped[pid] = read_json(clipped_path) if clipped_path.exists() else {}
    return recon_raw, recon_clean, recon_clipped


def recon_dataframe(recon_raw: Dict[str, dict], recon_clean: Dict[str, dict], recon_clipped: Dict[str, dict]) -> pd.DataFrame:
    n = settings.PREVIEW_N
    rows = []
    for pid in sorted(recon_clean.keys()):
        clean = recon_clean[pid]
        clipped = recon_clipped.get(pid, {}) or {}
        raw = recon_raw.get(pid, {}) or {}
        rows.append({
            "patient_id": pid,
            "full_name": clean.get("full_name"),
            "birthDate": clean.get("birthDate"),
            "gender": clean.get("gender"),
            "raw_conditions_n": len(ensure_list(raw.get("conditions"))),
            "conditions_n": len(clean.get("conditions", [])),
            "conditions_clipped_n": len(clipped.get("conditions", [])),
            "raw_medreq_n": len(ensure_list(raw.get("medication_requests"))),
            "medreq_n": len(clean.get("medication_requests", [])),
            "medreq_clipped_n": len(clipped.get("medication_requests", [])),
            "conditions_preview": "; ".join(clean.get("conditions", [])[:n]),
            "conditions_clipped_preview": "; ".join(clipped.get("conditions", [])[:n]),
            "medreq_preview": "; ".join(clean.get("medication_requests", [])[:n]),
            "medreq_clipped_preview": "; ".join(clipped.get("medication_requests", [])[:n]),
            "raw_file": str((settings.RECON_DIR / f"{pid}.raw.json").resolve()),
            "recon_file": str((settings.RECON_DIR / f"{pid}.json").resolve()),
            "recon_clipped_file": str((settings.RECON_DIR / f"{pid}.clipped.json").resolve()),
        })
    return pd.DataFrame(rows)
