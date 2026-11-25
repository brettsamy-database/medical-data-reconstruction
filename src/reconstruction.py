"""Reconstruction : relire la note clinique et en extraire un dossier structuré.

Le modèle ne voit que la note nettoyée, jamais le gold. On garde trois versions du résultat :
- `recon_raw`     : le JSON tel que le modèle l'a renvoyé ;
- `recon_clean`   : les listes nettoyées (puces, doublons, bruit), seule version utilisée pour les scores ;
- `recon_clipped` : l'intersection exacte avec le gold, utile pour déboguer mais jamais scorée.

Les données démographiques du gold sont recopiées dans `recon_clean` plutôt que
celles extraites : on mesure la reconstruction des conditions et des prescriptions,
et une erreur sur le nom pourrait mélanger deux patients dans les tables. Les
valeurs extraites restent dans `recon_raw` et sont scorées à part.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import settings
from src.llm import LLMResult, get_llm, load_json_lenient
from src.utils import ensure_list, norm_text, read_json, write_json

SCHEMA = {
    "patient_name": "string or null",
    "date_of_birth": "YYYY-MM-DD or null",
    "gender": "male|female|other|unknown or null",
    "conditions": ["string"],
    "medication_requests": ["string"],
}

EMPTY_EXTRACTION = {"patient_name": None, "date_of_birth": None, "gender": None, "conditions": [], "medication_requests": []}

# Un élément extrait de plus de 160 caractères est une phrase recopiée, pas un diagnostic.
MAX_ITEM_CHARS = 160

EXAMPLE_OUTPUT = {
    "patient_name": "John Smith",
    "date_of_birth": "1980-04-12",
    "gender": "male",
    "conditions": ["Hypertension", "Type 2 diabetes mellitus"],
    "medication_requests": ["Lisinopril 10 MG Oral Tablet", "Metformin 500 MG Oral Tablet"],
}


def extraction_prompt(note_text: str) -> str:
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
{json.dumps(EXAMPLE_OUTPUT, ensure_ascii=False, indent=2)}

Clinical note:
<note>
{note_text}
</note>

JSON:"""


def fix_to_json(raw_text: str, llm=None) -> str:
    """Deuxième chance : on renvoie au modèle sa réponse illisible en lui demandant de la convertir en JSON."""
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
    """Lit le JSON de la réponse ; s'il est illisible, demande jusqu'à `retries` fois au modèle de le réparer."""
    obj = load_json_lenient(llm_text)
    if obj is not None:
        return obj
    if retries > 0:
        llm = llm or get_llm()
        if llm.available():
            return parse_llm_json(fix_to_json(llm_text, llm=llm), retries=retries - 1, llm=llm)
    return None


# =========================
# Nettoyage des listes extraites
# =========================


def clean_extracted_item(v: str) -> str:
    """Retire les puces et numéros que le modèle ajoute parfois (« - Asthma », « 2) Asthma »)."""
    s = str(v or "").strip()
    s = re.sub(r"^\s*[-*•]+\s*", "", s)
    s = re.sub(r"^\s*\d+[\.\)\-:]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\r\n,;")


def _noise_rules() -> Tuple[List[str], set]:
    """Règles de bruit dérivées de PREFILTER_DROP_SUBSTRINGS.

    Une expression de plusieurs mots (« victim of ») est rejetée partout où elle
    apparaît. Un mot seul (« stress ») n'est rejeté que s'il constitue tout
    l'élément, pour ne pas jeter « Post-traumatic stress disorder ».
    """
    phrases = [norm_text(x) for x in settings.PREFILTER_DROP_SUBSTRINGS or [] if x]
    return [x for x in phrases if " " in x], {x for x in phrases if " " not in x}


def is_noisy_candidate(k: str, filter_noisy: bool) -> bool:
    if not filter_noisy or not settings.PREFILTER_ENABLED:
        return False
    if not k:
        return True
    multiword, singletons = _noise_rules()
    return k in singletons or any(p in k for p in multiword)


def postprocess_list(values: List[str], max_items: int, filter_noisy: bool) -> List[str]:
    """Nettoie une liste extraite : puces, doublons, éléments vides ou aberrants, bruit SDOH, puis plafond."""
    out, seen = [], set()
    for v in ensure_list(values):
        cleaned = clean_extracted_item(v)
        k = norm_text(cleaned)
        if len(k) <= 1 or len(k) > MAX_ITEM_CHARS or k in seen or is_noisy_candidate(k, filter_noisy):
            continue
        seen.add(k)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def clip_to_gold(values: List[str], gold_values: List[str]) -> List[str]:
    allowed = {norm_text(x) for x in gold_values or [] if x}
    return [v for v in values or [] if norm_text(v) in allowed]


# =========================
# Reconstruction
# =========================


def reconstruct_one(pid: str, note_text: str, gold: dict, llm=None) -> Tuple[dict, dict, dict]:
    """Renvoie (recon_raw, recon_clean, recon_clipped) pour un patient."""
    llm = llm or get_llm()
    note_text = (note_text or "").strip()
    gold = gold or {}

    obj = None
    if llm.available():
        txt = llm.generate(
            extraction_prompt(note_text),
            max_new_tokens=settings.RECON_EXTRACTION_MAX_NEW_TOKENS,
            temperature=settings.RECON_TEMPERATURE,
            do_sample=settings.RECON_DO_SAMPLE,
        )
        obj = parse_llm_json(txt, retries=1, llm=llm)
    if not isinstance(obj, dict):
        obj = dict(EMPTY_EXTRACTION)

    for key in ("patient_name", "date_of_birth", "gender"):
        obj.setdefault(key, None)
    obj["conditions"] = ensure_list(obj.get("conditions"))
    obj["medication_requests"] = ensure_list(obj.get("medication_requests"))

    cond = postprocess_list(obj["conditions"], settings.MAX_CONDITIONS_PER_PATIENT, filter_noisy=True)
    meds = postprocess_list(obj["medication_requests"], settings.MAX_MEDICATION_REQUESTS_PER_PATIENT, filter_noisy=False)

    demographics = {
        "patient_id": pid,
        "full_name": gold.get("full_name"),
        "birthDate": gold.get("birthDate"),
        "gender": gold.get("gender"),
    }
    recon_clean = {"note_used_for_extraction": note_text, **demographics,
                   "conditions": cond, "medication_requests": meds}
    recon_clipped = {**demographics,
                     "conditions": clip_to_gold(cond, gold.get("conditions", [])),
                     "medication_requests": clip_to_gold(meds, gold.get("medication_requests", []))}
    return obj, recon_clean, recon_clipped


def reconstruct_all(patients: List[dict], notes: Dict[str, LLMResult], llm=None) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
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


def _is_main_recon_file(p) -> bool:
    return not (p.name.endswith(".raw.json") or p.name.endswith(".clipped.json"))


def load_reconstructions() -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    recon_raw: Dict[str, dict] = {}
    recon_clean: Dict[str, dict] = {}
    recon_clipped: Dict[str, dict] = {}
    if not settings.RECON_DIR.exists():
        return recon_raw, recon_clean, recon_clipped
    for p in sorted(filter(_is_main_recon_file, settings.RECON_DIR.glob("*.json"))):
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
    for pid in sorted(recon_clean):
        clean = recon_clean[pid]
        clipped = recon_clipped.get(pid) or {}
        raw = recon_raw.get(pid) or {}
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
