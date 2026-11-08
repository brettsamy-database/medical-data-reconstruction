"""Étape 8 : chaîne de vérification LLM (note nettoyée + reconstruction -> verdict).

Repris de la cellule 8) du notebook original. Ne remplace pas les scores :
ajoute une relecture critique automatique.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Tuple

import pandas as pd

from config import settings
from src.llm import LLMResult, get_llm
from src.reconstruction import parse_llm_json, postprocess_list
from src.utils import write_json

VERIFICATION_SCHEMA = {
    "verdict": "consistent|minor_issues|major_issues|unknown",
    "confidence": "number between 0 and 1",
    "unsupported_conditions": ["string"],
    "missing_conditions_from_reconstruction": ["string"],
    "unsupported_medication_requests": ["string"],
    "missing_medication_requests_from_reconstruction": ["string"],
    "summary": "string",
}

ISSUE_KEYS = [
    "unsupported_conditions",
    "missing_conditions_from_reconstruction",
    "unsupported_medication_requests",
    "missing_medication_requests_from_reconstruction",
]


def verification_prompt(note_text: str, recon_clean_obj: dict) -> str:
    payload = {
        "conditions": recon_clean_obj.get("conditions", []),
        "medication_requests": recon_clean_obj.get("medication_requests", []),
    }
    return f"""You are verifying whether a structured reconstruction is faithfully supported by a clinical note.

Task:
Review the NOTE and the reconstructed structured data.
Return EXACTLY ONE valid JSON object following the schema below.

Rules:
- Use only information explicitly stated in the note.
- Do not infer new diagnoses, medications, procedures, or context.
- unsupported_* = items present in the reconstruction but not clearly supported by the note.
- missing_*_from_reconstruction = items clearly present in the note but missing from the reconstruction.
- verdict:
  - consistent: no material issue
  - minor_issues: small mismatch but reconstruction mostly usable
  - major_issues: important mismatch or several unsupported/missing items
  - unknown: impossible to judge reliably
- Return JSON only.

Schema:
{json.dumps(VERIFICATION_SCHEMA, ensure_ascii=False, indent=2)}

Clinical note:
<note>
{note_text}
</note>

Structured reconstruction:
<reconstruction>
{json.dumps(payload, ensure_ascii=False, indent=2)}
</reconstruction>

JSON:"""


def verify_reconstruction(pid: str, note_text: str, recon_clean_obj: dict, llm=None) -> dict:
    llm = llm or get_llm()
    signature_payload = {
        "note_text": note_text or "",
        "recon": {
            "conditions": recon_clean_obj.get("conditions", []),
            "medication_requests": recon_clean_obj.get("medication_requests", []),
        },
    }
    signature = hashlib.sha256(json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    cache_path = settings.VERIFICATION_DIR / f"{pid}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return cached
        except Exception:  # noqa: BLE001
            pass

    out = {
        "patient_id": pid,
        "signature": signature,
        "verdict": "unknown",
        "confidence": 0.0,
        **{k: [] for k in ISSUE_KEYS},
        "summary": "",
    }

    if llm.available():
        raw = llm.generate(
            verification_prompt(note_text, recon_clean_obj),
            max_new_tokens=settings.VERIFICATION_MAX_NEW_TOKENS,
            temperature=settings.VERIFICATION_TEMPERATURE,
            do_sample=settings.VERIFICATION_DO_SAMPLE,
        )
        obj = parse_llm_json(raw, retries=1, llm=llm) or {}
        out["verdict"] = str(obj.get("verdict", "unknown")).strip() or "unknown"
        try:
            out["confidence"] = float(obj.get("confidence", 0.0))
        except Exception:  # noqa: BLE001
            out["confidence"] = 0.0
        for key in ISSUE_KEYS:
            val = obj.get(key, [])
            out[key] = postprocess_list(val if isinstance(val, list) else [], 100, filter_noisy=("condition" in key))
        out["summary"] = str(obj.get("summary", "")).strip()

    write_json(cache_path, out)
    return out


def verification_row(review: dict, gold_obj: dict) -> dict:
    unsupported_total = len(review["unsupported_conditions"]) + len(review["unsupported_medication_requests"])
    missing_total = len(review["missing_conditions_from_reconstruction"]) + len(review["missing_medication_requests_from_reconstruction"])
    return {
        "patient_id": review["patient_id"],
        "note_style": gold_obj.get("note_style"),
        "note_style_label": gold_obj.get("note_style_label"),
        "verification_verdict": review["verdict"],
        "verification_confidence": float(review["confidence"]),
        "verification_unsupported_conditions_n": len(review["unsupported_conditions"]),
        "verification_missing_conditions_n": len(review["missing_conditions_from_reconstruction"]),
        "verification_unsupported_medication_requests_n": len(review["unsupported_medication_requests"]),
        "verification_missing_medication_requests_n": len(review["missing_medication_requests_from_reconstruction"]),
        "verification_unsupported_total_n": unsupported_total,
        "verification_missing_total_n": missing_total,
        "verification_issue_total_n": unsupported_total + missing_total,
        "verification_summary": review["summary"],
    }


def verification_by_note_style(df_verification: pd.DataFrame) -> pd.DataFrame:
    if df_verification.empty or not {"note_style", "verification_issue_total_n", "verification_verdict"}.issubset(df_verification.columns):
        return pd.DataFrame()
    return (
        df_verification.groupby(["note_style", "note_style_label", "verification_verdict"], dropna=False)
        .agg(
            n_patients=("patient_id", "count"),
            mean_issue_total_n=("verification_issue_total_n", "mean"),
            mean_confidence=("verification_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["note_style", "note_style_label", "n_patients"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def run_verification(patients: List[dict], recon_clean: Dict[str, dict], notes: Dict[str, LLMResult],
                     llm=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Vérifie chaque patient. Retourne (df_verification, df_verification_by_note_style) et écrit les CSV."""
    settings.VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    gold_by_pid = {p["patient_id"]: p for p in patients}

    rows = []
    for pid in sorted(recon_clean.keys()):
        note_obj = notes.get(pid)
        note_text = (note_obj.note if note_obj else "") or ""
        review = verify_reconstruction(pid, note_text, recon_clean.get(pid, {}) or {}, llm=llm)
        rows.append(verification_row(review, gold_by_pid.get(pid, {}) or {}))

    df_verification = pd.DataFrame(rows)
    if df_verification.empty:
        return df_verification, pd.DataFrame()
    df_verification = df_verification.sort_values(["note_style", "patient_id"]).reset_index(drop=True)
    df_verification.to_csv(settings.VERIFICATION_DIR / "df_verification.csv", index=False)

    df_by_style = verification_by_note_style(df_verification)
    if not df_by_style.empty:
        df_by_style.to_csv(settings.VERIFICATION_DIR / "df_verification_by_note_style.csv", index=False)
    return df_verification, df_by_style


def merge_verification_into_scores(df_scores: pd.DataFrame, df_verification: pd.DataFrame) -> pd.DataFrame:
    if df_verification.empty:
        return df_scores
    return df_scores.merge(df_verification, on=["patient_id", "note_style", "note_style_label"], how="left")
