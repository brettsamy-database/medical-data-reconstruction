"""Vérification : une relecture critique de chaque reconstruction par le LLM.

Les scores comparent des listes élément par élément. La vérification pose une
question plus globale : « cette reconstruction est-elle fidèle à la note ? ».
Elle complète les scores sans les remplacer. Comme c'est le même modèle qui a
produit la reconstruction, son verdict est à prendre avec recul : il a tendance
à être indulgent.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd

from config import settings
from src.llm import LLMResult, get_llm
from src.reconstruction import parse_llm_json, postprocess_list
from src.utils import ensure_list, read_cached, signature, write_json

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

MAX_ISSUES_PER_LIST = 100


def _recon_payload(recon_clean_obj: dict) -> dict:
    return {
        "conditions": recon_clean_obj.get("conditions", []),
        "medication_requests": recon_clean_obj.get("medication_requests", []),
    }


def verification_prompt(note_text: str, recon_clean_obj: dict) -> str:
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
{json.dumps(_recon_payload(recon_clean_obj), ensure_ascii=False, indent=2)}
</reconstruction>

JSON:"""


def verify_reconstruction(pid: str, note_text: str, recon_clean_obj: dict, llm=None) -> dict:
    llm = llm or get_llm()
    sig = signature({"note_text": note_text or "", "recon": _recon_payload(recon_clean_obj)})
    cache_path = settings.VERIFICATION_DIR / f"{pid}.json"
    cached = read_cached(cache_path, sig)
    if cached:
        return cached

    # Sans réponse exploitable du modèle, le verdict reste « unknown » avec une confiance nulle.
    out = {
        "patient_id": pid,
        "signature": sig,
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
        except (TypeError, ValueError):
            out["confidence"] = 0.0
        for key in ISSUE_KEYS:
            out[key] = postprocess_list(ensure_list(obj.get(key)), MAX_ISSUES_PER_LIST, filter_noisy=("condition" in key))
        out["summary"] = str(obj.get("summary", "")).strip()

    write_json(cache_path, out)
    return out


def verification_row(review: dict, gold_obj: dict) -> dict:
    n = {k: len(review[k]) for k in ISSUE_KEYS}
    unsupported_total = n["unsupported_conditions"] + n["unsupported_medication_requests"]
    missing_total = n["missing_conditions_from_reconstruction"] + n["missing_medication_requests_from_reconstruction"]
    return {
        "patient_id": review["patient_id"],
        "note_style": gold_obj.get("note_style"),
        "note_style_label": gold_obj.get("note_style_label"),
        "verification_verdict": review["verdict"],
        "verification_confidence": float(review["confidence"]),
        "verification_unsupported_conditions_n": n["unsupported_conditions"],
        "verification_missing_conditions_n": n["missing_conditions_from_reconstruction"],
        "verification_unsupported_medication_requests_n": n["unsupported_medication_requests"],
        "verification_missing_medication_requests_n": n["missing_medication_requests_from_reconstruction"],
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
    """Vérifie chaque patient. Renvoie (df_verification, df_verification_by_note_style) et écrit les CSV."""
    gold_by_pid = {p["patient_id"]: p for p in patients}
    rows = []
    for pid in sorted(recon_clean):
        note_text = notes[pid].note if pid in notes else ""
        review = verify_reconstruction(pid, note_text or "", recon_clean.get(pid) or {}, llm=llm)
        rows.append(verification_row(review, gold_by_pid.get(pid) or {}))

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
