"""Étapes 7, 9 et 9.1 : scoring, analyse des erreurs et tables de synthèse.

Repris des cellules 7), 9) et 9.1) du notebook original.

Deux lectures complémentaires :
- note vs recon   (officielle) : la note générée vs la reconstruction
- source vs recon (secondaire) : le gold source vs la reconstruction

Chacune en version exacte (texte normalisé) et sémantique (jugée par le LLM).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import settings
from src.llm import LLMResult, get_llm
from src.reconstruction import ensure_list, parse_llm_json, postprocess_list
from src.utils import norm_text, write_json

KINDS = ("conditions", "medication_requests")

# =========================
# Métriques de base
# =========================


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if tp == 0 and fp == 0 and fn == 0:
        f1 = 1.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def metrics_from_sets(gold_set: set, pred_set: set) -> dict:
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return metrics_from_counts(tp, fp, fn)


def name_exact(gold_name: str, pred_name: str) -> int:
    return int(bool(norm_text(gold_name)) and norm_text(gold_name) == norm_text(pred_name or ""))


def dob_exact(gold_dob: str, pred_dob: str) -> int:
    g = (gold_dob or "").strip()
    p = (pred_dob or "").strip()
    return int(bool(g) and g == p)


def gender_exact(gold_gender: str, pred_gender: str) -> int:
    return int(bool(norm_text(gold_gender or "")) and norm_text(gold_gender or "") == norm_text(pred_gender or ""))


def _parse_json_obj(text: str) -> dict:
    obj = parse_llm_json(text, retries=0)
    return obj if isinstance(obj, dict) else {}


def exact_items_supported_by_note(note_text: str, source_items: List[str]) -> List[str]:
    """Items du gold présents mot pour mot (après normalisation) dans la note."""
    note_norm = norm_text(note_text or "")
    supported = []
    for item in source_items or []:
        item_norm = norm_text(item)
        if item_norm and item_norm in note_norm:
            supported.append(item)
    return supported


# =========================
# Matching sémantique source vs recon
# =========================


def _llm_judge_match_index(target: str, candidates: List[str], llm=None) -> Tuple[int, float]:
    llm = llm or get_llm()
    if not candidates:
        return -1, 0.0

    candidates = candidates[:25]

    prompt = (
        "You are a strict matcher for clinical concepts.\n"
        "Task: decide whether the TARGET is semantically the same as one of the CANDIDATES.\n"
        "If yes, return the index of the best candidate. If no, return -1.\n\n"
        "Rules:\n"
        "- Only match if it is clearly the same diagnosis/condition or the same medication.\n"
        "- Do not match if it is merely related or in the same family.\n"
        "- Output ONLY valid JSON. No markdown. No commentary.\n"
        "- JSON schema: {\"match_index\": integer, \"confidence\": number}\n\n"
        f"TARGET: {target}\n"
        "CANDIDATES:\n"
    )
    for i, c in enumerate(candidates):
        prompt += f"{i}. {c}\n"
    prompt += "\nJSON:"

    txt = llm.generate(
        prompt,
        max_new_tokens=settings.SEMANTIC_MATCH_MAX_NEW_TOKENS,
        temperature=settings.SCORING_TEMPERATURE,
        do_sample=settings.SCORING_DO_SAMPLE,
    )
    obj = _parse_json_obj(txt) or {}
    try:
        idx = int(obj.get("match_index", -1))
    except Exception:  # noqa: BLE001
        idx = -1
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:  # noqa: BLE001
        conf = 0.0

    if idx < -1 or idx >= len(candidates):
        idx = -1
    conf = min(max(conf, 0.0), 1.0)
    return idx, conf


def _hash_lists(pred_list: List[str], gold_list: List[str]) -> str:
    payload = json.dumps({"pred": pred_list, "gold": gold_list}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(cache_path, signature: str) -> Optional[dict]:
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return cached
        except Exception:  # noqa: BLE001
            pass
    return None


def semantic_align_and_score(pid: str, kind: str, pred_items: List[str], gold_items: List[str], llm=None) -> dict:
    """Source vs recon : alignement gold / reconstruction (exact d'abord, puis juge LLM)."""
    llm = llm or get_llm()
    pred_items = [x for x in pred_items if x]
    gold_items = [x for x in gold_items if x]

    gold_norm = [norm_text(x) for x in gold_items]
    pred_norm = [norm_text(x) for x in pred_items]

    cache_path = settings.LLM_ALIGN_DIR / f"{pid}.{kind}.json"
    sig = _hash_lists(pred_norm, gold_norm)
    cached = _read_cache(cache_path, sig)
    if cached:
        return cached

    used_gold = set()
    matches = []
    tp = 0

    for p_raw, p in zip(pred_items, pred_norm):
        match_idx = -1
        conf = 1.0

        if p and p in gold_norm:
            for i, g in enumerate(gold_norm):
                if g == p and i not in used_gold:
                    match_idx = i
                    break

        if match_idx == -1 and llm.available():
            match_idx, conf = _llm_judge_match_index(p_raw, gold_items, llm=llm)
            if match_idx in used_gold:
                match_idx = -1
                conf = 0.0

        if match_idx >= 0:
            used_gold.add(match_idx)
            tp += 1

        matches.append({
            "pred": p_raw,
            "match_index": int(match_idx),
            "matched_gold": gold_items[match_idx] if match_idx >= 0 else None,
            "confidence": float(conf),
        })

    fp = len(pred_items) - tp
    fn = len(gold_items) - len(used_gold)
    metrics = metrics_from_counts(tp, fp, fn)

    out = {
        "patient_id": pid,
        "kind": kind,
        "signature": sig,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision_semantic": metrics["precision"],
        "recall_semantic": metrics["recall"],
        "f1_semantic": metrics["f1"],
        "pred_n": len(pred_items),
        "gold_n": len(gold_items),
        "matches": matches,
    }
    write_json(cache_path, out)
    return out


# =========================
# Cohérence sémantique note vs recon
# =========================


def _hash_note_and_items(note_text: str, kind: str, pred_items: List[str]) -> str:
    payload = json.dumps({
        "note_text": note_text or "",
        "kind": kind,
        "pred_items": pred_items or [],
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def note_consistency_prompt(note_text: str, kind: str, pred_items: List[str]) -> str:
    label = "conditions" if kind == "conditions" else "medication requests"
    item_lines = "\n".join([f"{i}. {x}" for i, x in enumerate(pred_items)]) or "(none)"
    return f"""You are reviewing a clinical note and a structured extracted list of {label}.

Task:
1. Decide which listed items are explicitly supported by the note or are clear semantic equivalents of what is explicitly stated.
2. Identify additional {label} explicitly present in the note but missing from the extracted list.

Rules:
- Use only what is explicitly stated in the note.
- Do not infer or invent anything.
- For conditions: count diagnoses, disorders, chronic diseases, medical problems, or explicitly stated past history items only if clearly present as such in the note.
- For medication requests: count only medications explicitly requested, prescribed, active, continued, or listed as current in the note.
- A synonym or very close canonical equivalent counts as the same item.
- Return JSON only. No markdown. No commentary.

Clinical note:
<note>
{note_text}
</note>

Extracted {label}:
{item_lines}

JSON schema:
{{"supported_indexes": [integer], "missing_items": ["string"]}}

JSON:"""


def note_vs_recon_semantic_score(pid: str, kind: str, note_text: str, pred_items: List[str], llm=None) -> dict:
    """Scoring officiel : note nettoyée vs reconstruction nettoyée, jugé par le LLM."""
    llm = llm or get_llm()
    pred_items = [x for x in (pred_items or []) if x]
    note_text = (note_text or "").strip()

    cache_path = settings.NOTE_ALIGN_DIR / f"{pid}.{kind}.json"
    sig = _hash_note_and_items(note_text, kind, pred_items)
    cached = _read_cache(cache_path, sig)
    if cached:
        return cached

    if not note_text and not pred_items:
        out = {
            "patient_id": pid, "kind": kind, "signature": sig,
            "supported_n": 0, "unsupported_n": 0, "missing_n": 0, "pred_n": 0,
            "precision_note_consistency": 1.0, "recall_note_consistency": 1.0, "f1_note_consistency": 1.0,
            "supported_items": [], "unsupported_items": [], "missing_items": [],
        }
        write_json(cache_path, out)
        return out

    supported_indexes: List[int] = []
    missing_items: List[str] = []

    if llm.available():
        obj = _parse_json_obj(
            llm.generate(
                note_consistency_prompt(note_text, kind, pred_items),
                max_new_tokens=settings.NOTE_CONSISTENCY_MAX_NEW_TOKENS,
                temperature=settings.SCORING_TEMPERATURE,
                do_sample=settings.SCORING_DO_SAMPLE,
            )
        ) or {}
        raw_idxs = obj.get("supported_indexes", [])
        if isinstance(raw_idxs, list):
            seen = set()
            for x in raw_idxs:
                try:
                    idx = int(x)
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= idx < len(pred_items) and idx not in seen:
                    supported_indexes.append(idx)
                    seen.add(idx)
        raw_missing = obj.get("missing_items", [])
        if isinstance(raw_missing, list):
            missing_items = [str(x).strip() for x in raw_missing if str(x).strip()]

    supported_items = [pred_items[i] for i in supported_indexes]
    supported_idx_set = set(supported_indexes)
    unsupported_items = [x for i, x in enumerate(pred_items) if i not in supported_idx_set]
    missing_items = postprocess_list(
        missing_items,
        settings.MAX_CONDITIONS_PER_PATIENT if kind == "conditions" else settings.MAX_MEDICATION_REQUESTS_PER_PATIENT,
        filter_noisy=(kind == "conditions"),
    )

    metrics = metrics_from_counts(len(supported_items), len(unsupported_items), len(missing_items))

    out = {
        "patient_id": pid,
        "kind": kind,
        "signature": sig,
        "supported_n": metrics["tp"],
        "unsupported_n": metrics["fp"],
        "missing_n": metrics["fn"],
        "pred_n": len(pred_items),
        "precision_note_consistency": metrics["precision"],
        "recall_note_consistency": metrics["recall"],
        "f1_note_consistency": metrics["f1"],
        "supported_items": supported_items,
        "unsupported_items": unsupported_items,
        "missing_items": missing_items,
    }
    write_json(cache_path, out)
    return out


# =========================
# Table des scores
# =========================


def _metric_cols(prefix: str, m: dict, suffix: str = "") -> dict:
    return {
        f"{prefix}_tp{suffix}": int(m["tp"]),
        f"{prefix}_fp{suffix}": int(m["fp"]),
        f"{prefix}_fn{suffix}": int(m["fn"]),
        f"{prefix}_precision{suffix}": float(m["precision"]),
        f"{prefix}_recall{suffix}": float(m["recall"]),
        f"{prefix}_f1{suffix}": float(m["f1"]),
    }


def _note_sem_as_metrics(d: dict) -> dict:
    return {"tp": d["supported_n"], "fp": d["unsupported_n"], "fn": d["missing_n"],
            "precision": d["precision_note_consistency"], "recall": d["recall_note_consistency"],
            "f1": d["f1_note_consistency"]}


def _source_sem_as_metrics(d: dict) -> dict:
    return {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
            "precision": d["precision_semantic"], "recall": d["recall_semantic"], "f1": d["f1_semantic"]}


def score_patient(gold: dict, pred_clean: dict, raw: dict, note_text: str, llm=None) -> dict:
    """Calcule toutes les métriques d'un patient (une ligne de `df_scores`)."""
    pid = gold["patient_id"]
    max_items = {"conditions": settings.MAX_CONDITIONS_PER_PATIENT,
                 "medication_requests": settings.MAX_MEDICATION_REQUESTS_PER_PATIENT}

    row = {
        "patient_id": pid,
        "note_style": gold.get("note_style"),
        "note_style_label": gold.get("note_style_label"),
        # Démographie (diagnostic seulement)
        "gold_full_name": gold.get("full_name"),
        "gold_birthDate": gold.get("birthDate"),
        "gold_gender": gold.get("gender"),
        "raw_patient_name": raw.get("patient_name"),
        "raw_date_of_birth": raw.get("date_of_birth"),
        "raw_gender": raw.get("gender"),
        "name_exact_raw": name_exact(gold.get("full_name") or "", raw.get("patient_name")),
        "dob_exact_raw": dob_exact(gold.get("birthDate"), raw.get("date_of_birth")),
        "gender_exact_raw": gender_exact(gold.get("gender"), raw.get("gender")),
    }

    res: Dict[str, dict] = {}
    for kind in KINDS:
        filt = kind == "conditions"
        gold_list = postprocess_list(gold.get(kind) or [], 10_000, filter_noisy=filt)
        pred_list = postprocess_list(pred_clean.get(kind) or [], max_items[kind], filter_noisy=filt)
        gold_set = set(norm_text(x) for x in gold_list if x)
        pred_set = set(norm_text(x) for x in pred_list if x)

        note_exact_items = exact_items_supported_by_note(note_text, gold_list)
        note_exact_set = set(norm_text(x) for x in note_exact_items if x)

        res[kind] = {
            "gold_list": gold_list,
            "pred_list": pred_list,
            "note_exact_set": note_exact_set,
            # Officiel : items gold explicitement présents dans la note vs recon
            "note_exact": metrics_from_sets(note_exact_set, pred_set),
            "note_sem": note_vs_recon_semantic_score(pid, kind, note_text, pred_list, llm=llm),
            # Secondaire : source vs recon
            "source_exact": metrics_from_sets(gold_set, pred_set),
            "source_sem": semantic_align_and_score(pid, kind, pred_list, gold_list, llm=llm),
        }

    for kind in KINDS:
        row.update(_metric_cols(f"score_note_vs_recon_{kind}_exact", res[kind]["note_exact"]))
    for kind in KINDS:
        row.update(_metric_cols(f"score_note_vs_recon_{kind}_semantic", _note_sem_as_metrics(res[kind]["note_sem"]), "_llm"))
    for kind in KINDS:
        row.update(_metric_cols(f"score_source_vs_recon_{kind}_exact", res[kind]["source_exact"]))
    for kind in KINDS:
        row.update(_metric_cols(f"score_source_vs_recon_{kind}_semantic", _source_sem_as_metrics(res[kind]["source_sem"]), "_llm"))

    c, m = res["conditions"], res["medication_requests"]
    row.update({
        # Comptages et métadonnées
        "note_vs_recon_note_conditions_exact_source_items_detected_n": len(c["note_exact_set"]),
        "note_vs_recon_note_medication_requests_exact_source_items_detected_n": len(m["note_exact_set"]),
        "gold_cond_n": len(c["gold_list"]),
        "pred_cond_n": len(c["pred_list"]),
        "raw_cond_n": len(ensure_list(raw.get("conditions"))),
        "gold_med_n": len(m["gold_list"]),
        "pred_med_n": len(m["pred_list"]),
        "raw_med_n": len(ensure_list(raw.get("medication_requests"))),
        # Alias historiques (compatibilité avec les anciens exports)
        "conditions_f1_string": float(c["source_exact"]["f1"]),
        "medication_requests_f1_string": float(m["source_exact"]["f1"]),
        "conditions_f1_semantic_llm": float(c["source_sem"]["f1_semantic"]),
        "medication_requests_f1_semantic_llm": float(m["source_sem"]["f1_semantic"]),
        "note_vs_recon_conditions_f1_llm": float(c["note_sem"]["f1_note_consistency"]),
        "note_vs_recon_medication_requests_f1_llm": float(m["note_sem"]["f1_note_consistency"]),
        "note_vs_recon_conditions_precision_llm": float(c["note_sem"]["precision_note_consistency"]),
        "note_vs_recon_medication_requests_precision_llm": float(m["note_sem"]["precision_note_consistency"]),
        "note_vs_recon_conditions_recall_llm": float(c["note_sem"]["recall_note_consistency"]),
        "note_vs_recon_medication_requests_recall_llm": float(m["note_sem"]["recall_note_consistency"]),
        "note_vs_recon_conditions_supported_n_llm": int(c["note_sem"]["supported_n"]),
        "note_vs_recon_medication_requests_supported_n_llm": int(m["note_sem"]["supported_n"]),
        "note_vs_recon_conditions_unsupported_n_llm": int(c["note_sem"]["unsupported_n"]),
        "note_vs_recon_medication_requests_unsupported_n_llm": int(m["note_sem"]["unsupported_n"]),
        "note_vs_recon_conditions_missing_n_llm": int(c["note_sem"]["missing_n"]),
        "note_vs_recon_medication_requests_missing_n_llm": int(m["note_sem"]["missing_n"]),
        "cond_fp_semantic": int(c["source_sem"]["fp"]),
        "cond_fn_semantic": int(c["source_sem"]["fn"]),
        "med_fp_semantic": int(m["source_sem"]["fp"]),
        "med_fn_semantic": int(m["source_sem"]["fn"]),
    })
    return row


def score_all(patients: List[dict], recon_clean: Dict[str, dict], recon_raw: Dict[str, dict],
              notes: Dict[str, LLMResult], llm=None) -> pd.DataFrame:
    """Construit `df_scores` et écrit un JSON par patient + df_scores.csv dans data/processed/scores/."""
    settings.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    gold_by_pid = {p["patient_id"]: p for p in patients}

    rows = []
    for pid in sorted(gold_by_pid.keys()):
        note_obj = notes.get(pid)
        note_text = (note_obj.note if note_obj else "") or ""
        rows.append(score_patient(gold_by_pid[pid], recon_clean.get(pid, {}) or {},
                                  recon_raw.get(pid, {}) or {}, note_text, llm=llm))

    df_scores = pd.DataFrame(rows)
    if df_scores.empty:
        return df_scores
    df_scores = df_scores.sort_values(["note_style", "patient_id"]).reset_index(drop=True)

    for r in rows:
        write_json(settings.SCORES_DIR / f'{r["patient_id"]}.json', r)
    df_scores.sort_values("patient_id").to_csv(settings.SCORES_DIR / "df_scores.csv", index=False)
    return df_scores


def print_mean_scores(df_scores: pd.DataFrame) -> None:
    for col in [
        "name_exact_raw", "dob_exact_raw", "gender_exact_raw",
        "score_note_vs_recon_conditions_exact_f1", "score_note_vs_recon_medication_requests_exact_f1",
        "score_note_vs_recon_conditions_semantic_f1_llm", "score_note_vs_recon_medication_requests_semantic_f1_llm",
        "score_source_vs_recon_conditions_exact_f1", "score_source_vs_recon_medication_requests_exact_f1",
        "score_source_vs_recon_conditions_semantic_f1_llm", "score_source_vs_recon_medication_requests_semantic_f1_llm",
    ]:
        if col in df_scores.columns:
            print(f"Mean {col}: {float(df_scores[col].mean()):.3f}")


# =========================
# 9) Analyse des erreurs
# =========================


def classify_error_type(
    gold_n: int,
    pred_n: int,
    f1_exact: float,
    f1_semantic: float,
    note_f1: float,
    note_precision: float,
    note_recall: float,
    fp_semantic: int,
    fn_semantic: int,
    note_unsupported_n: int,
    note_missing_n: int,
) -> dict:
    """Donne overall / stage / severity / explanation pour un patient et un type d'entité."""
    if gold_n == 0 and pred_n == 0:
        overall = "no_target_and_no_prediction"
        stage = "none"
    elif gold_n > 0 and pred_n == 0:
        overall = "complete_omission"
        stage = "note_to_reconstruction" if note_missing_n > 0 else "source_to_note_or_reconstruction"
    elif f1_exact == 1.0 and f1_semantic == 1.0 and note_f1 == 1.0:
        overall = "perfect_reconstruction"
        stage = "none"
    elif f1_exact < 1.0 and f1_semantic == 1.0 and note_f1 == 1.0:
        overall = "exact_wording_gap_only"
        stage = "canonicalization_exact_matching"
    elif note_f1 >= 0.9 and f1_semantic < 0.9:
        overall = "source_to_note_gap_or_generation_loss"
        stage = "source_to_note"
    elif f1_semantic >= 0.9 and note_f1 < 0.9:
        if note_missing_n > 0 and note_unsupported_n == 0:
            overall = "under_extraction_from_note"
        elif note_unsupported_n > 0 and note_missing_n == 0:
            overall = "over_extraction_from_note"
        else:
            overall = "mixed_note_to_reconstruction_error"
        stage = "note_to_reconstruction"
    elif f1_semantic > f1_exact and note_f1 >= 0.9:
        overall = "canonicalization_or_exact_match_gap"
        stage = "canonicalization_exact_matching"
    else:
        if fp_semantic > 0 and fn_semantic == 0:
            overall = "precision_dominant_pipeline_error"
        elif fn_semantic > 0 and fp_semantic == 0:
            overall = "recall_dominant_pipeline_error"
        elif fp_semantic > 0 and fn_semantic > 0:
            overall = "mixed_pipeline_error"
        else:
            overall = "partial_pipeline_error"
        stage = "mixed"

    quality = min(f1_semantic, note_f1)
    if overall in {"perfect_reconstruction", "no_target_and_no_prediction"}:
        severity = "none"
    elif quality >= 0.85:
        severity = "mild"
    elif quality >= 0.55:
        severity = "moderate"
    else:
        severity = "severe"

    explanation = (
        f"overall={overall}; stage={stage}; exact={f1_exact:.2f}; "
        f"semantic={f1_semantic:.2f}; note_consistency={note_f1:.2f}; "
        f"unsupported_from_note={note_unsupported_n}; missing_from_note={note_missing_n}; "
        f"fp_semantic={fp_semantic}; fn_semantic={fn_semantic}"
    )
    return {"overall": overall, "stage": stage, "severity": severity, "explanation": explanation}


def _classify_row(row: pd.Series, kind: str) -> dict:
    short = "cond" if kind == "conditions" else "med"
    return classify_error_type(
        gold_n=int(row[f"gold_{short}_n"]),
        pred_n=int(row[f"pred_{short}_n"]),
        f1_exact=float(row[f"score_source_vs_recon_{kind}_exact_f1"]),
        f1_semantic=float(row[f"score_source_vs_recon_{kind}_semantic_f1_llm"]),
        note_f1=float(row[f"score_note_vs_recon_{kind}_semantic_f1_llm"]),
        note_precision=float(row[f"score_note_vs_recon_{kind}_semantic_precision_llm"]),
        note_recall=float(row[f"score_note_vs_recon_{kind}_semantic_recall_llm"]),
        fp_semantic=int(row[f"{short}_fp_semantic"]),
        fn_semantic=int(row[f"{short}_fn_semantic"]),
        note_unsupported_n=int(row[f"note_vs_recon_{kind}_unsupported_n_llm"]),
        note_missing_n=int(row[f"note_vs_recon_{kind}_missing_n_llm"]),
    )


def build_error_analysis(df_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_scores.iterrows():
        cond_error = _classify_row(row, "conditions")
        med_error = _classify_row(row, "medication_requests")
        pipeline_summary = (
            cond_error["overall"]
            if cond_error["overall"] == med_error["overall"] and cond_error["stage"] == med_error["stage"]
            else f"conditions={cond_error['overall']} | medication_requests={med_error['overall']}"
        )
        rows.append({
            "patient_id": row["patient_id"],
            "error_type_conditions_overall": cond_error["overall"],
            "error_type_conditions_stage": cond_error["stage"],
            "error_type_conditions_severity": cond_error["severity"],
            "error_type_conditions_explanation": cond_error["explanation"],
            "error_type_medication_requests_overall": med_error["overall"],
            "error_type_medication_requests_stage": med_error["stage"],
            "error_type_medication_requests_severity": med_error["severity"],
            "error_type_medication_requests_explanation": med_error["explanation"],
            "error_type_pipeline_summary": pipeline_summary,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("patient_id").reset_index(drop=True) if not df.empty else df


def error_types_by_note_style(df_scores: pd.DataFrame, kind: str) -> pd.DataFrame:
    col = f"error_type_{kind}_overall"
    if not {"note_style", "note_style_label", col}.issubset(df_scores.columns):
        return pd.DataFrame()
    return (
        df_scores.groupby(["note_style", "note_style_label", col], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["note_style", "note_style_label", "n"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


DIFFICULTY_COLS = [
    "score_source_vs_recon_conditions_semantic_f1_llm",
    "score_source_vs_recon_medication_requests_semantic_f1_llm",
    "score_note_vs_recon_conditions_semantic_f1_llm",
    "score_note_vs_recon_medication_requests_semantic_f1_llm",
]


def add_difficulty_score(df_scores: pd.DataFrame) -> pd.DataFrame:
    df = df_scores.copy()
    cols = [c for c in DIFFICULTY_COLS if c in df.columns]
    df["difficulty_score"] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if cols else None
    return df


def hard_cases_by_note_style(df_scores: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Les `top_n` patients les plus difficiles (difficulty_score le plus bas) par style de note."""
    needed = {"note_style", "note_style_label", "patient_id", "difficulty_score"}
    if not needed.issubset(df_scores.columns):
        return pd.DataFrame()
    cols = ["note_style", "note_style_label", "patient_id", "difficulty_score", *DIFFICULTY_COLS,
            "error_type_conditions_overall", "error_type_medication_requests_overall", "error_type_pipeline_summary"]
    cols = [c for c in cols if c in df_scores.columns]
    return (
        df_scores.sort_values(["note_style", "difficulty_score", "patient_id"], ascending=[True, True, True])
        .groupby(["note_style", "note_style_label"], dropna=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )


def run_error_analysis(df_scores: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Étape 9 complète. Écrit les CSV dans data/processed/scores/ et renvoie toutes les tables.

    Clés : df_scores (enrichi), df_error_analysis, df_error_types_by_note_style_conditions,
    df_error_types_by_note_style_medication_requests, df_hard_cases_by_note_style.
    """
    df_error_analysis = build_error_analysis(df_scores)
    df_scores = df_scores.merge(df_error_analysis, on="patient_id", how="left")

    out_dir = settings.SCORES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df_error_analysis.to_csv(out_dir / "df_error_analysis.csv", index=False)
    df_scores.sort_values("patient_id").to_csv(out_dir / "df_scores.csv", index=False)

    # Met à jour les JSON patient avec les champs d'analyse des erreurs
    for _, r in df_scores.iterrows():
        payload = {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in r.to_dict().items()}
        write_json(out_dir / f'{r["patient_id"]}.json', payload)

    df_cond = error_types_by_note_style(df_scores, "conditions")
    df_med = error_types_by_note_style(df_scores, "medication_requests")
    df_scores = add_difficulty_score(df_scores)
    df_hard = hard_cases_by_note_style(df_scores)

    for name, df in [
        ("df_error_types_by_note_style_conditions", df_cond),
        ("df_error_types_by_note_style_medication_requests", df_med),
        ("df_hard_cases_by_note_style", df_hard),
    ]:
        if not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)

    return {
        "df_scores": df_scores,
        "df_error_analysis": df_error_analysis,
        "df_error_types_by_note_style_conditions": df_cond,
        "df_error_types_by_note_style_medication_requests": df_med,
        "df_hard_cases_by_note_style": df_hard,
    }


# =========================
# 9.1) Tables de synthèse
# =========================

SCORE_SUMMARY_METRIC_SPECS = [("demographics", "demographics", m) for m in ("name_exact_raw", "dob_exact_raw", "gender_exact_raw")]
for _family in ("note_vs_recon", "source_vs_recon"):
    for _kind in KINDS:
        for _mode, _suffix in (("exact", ""), ("semantic", "_llm")):
            for _metric in ("precision", "recall", "f1"):
                SCORE_SUMMARY_METRIC_SPECS.append(
                    (_family, _kind, f"score_{_family}_{_kind}_{_mode}_{_metric}{_suffix}")
                )

VERDICT_RATE_SPECS = [
    ("verification_consistent_rate", "consistent"),
    ("verification_minor_issues_rate", "minor_issues"),
    ("verification_major_issues_rate", "major_issues"),
    ("verification_unknown_rate", "unknown"),
]
VERIFICATION_MEAN_SPECS = [
    ("verification_mean_confidence", "verification_confidence", "confidence"),
    ("verification_mean_issue_total_n", "verification_issue_total_n", "verification_issue_total_n"),
]


def _stats_row(series: pd.Series) -> dict:
    return {
        "mean_value": float(series.mean()) if len(series) else None,
        "median_value": float(series.median()) if len(series) else None,
        "std_value": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
        "min_value": float(series.min()) if len(series) else None,
        "max_value": float(series.max()) if len(series) else None,
        "n_non_null": int(series.notna().sum()),
        "aggregation_level": "patient_mean",
    }


def _summary_rows(df_scores_grp: Optional[pd.DataFrame], df_verif_grp: Optional[pd.DataFrame], base: dict) -> List[dict]:
    rows = []
    if df_scores_grp is not None:
        n_total = int(len(df_scores_grp))
        for metric_family, entity_type, metric_name in SCORE_SUMMARY_METRIC_SPECS:
            if metric_name not in df_scores_grp.columns:
                continue
            series = pd.to_numeric(df_scores_grp[metric_name], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append({**base, "metric_family": metric_family, "entity_type": entity_type,
                         "metric_name": metric_name, **_stats_row(series), "n_patients_total": n_total})

    if df_verif_grp is not None and not df_verif_grp.empty:
        n_total = int(len(df_verif_grp))
        verdict_col = "verification_verdict" if "verification_verdict" in df_verif_grp.columns else "verdict"
        for metric_name, verdict_value in VERDICT_RATE_SPECS:
            series = (df_verif_grp[verdict_col].fillna("") == verdict_value).astype(float)
            rows.append({**base, "metric_family": "verification", "entity_type": "all",
                         "metric_name": metric_name, **_stats_row(series), "n_patients_total": n_total})
        for metric_name, preferred_col, fallback_col in VERIFICATION_MEAN_SPECS:
            source_col = preferred_col if preferred_col in df_verif_grp.columns else fallback_col
            if source_col not in df_verif_grp.columns:
                continue
            series = pd.to_numeric(df_verif_grp[source_col], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append({**base, "metric_family": "verification", "entity_type": "all",
                         "metric_name": metric_name, **_stats_row(series), "n_patients_total": n_total})
    return rows


SUMMARY_COLUMNS = ["summary_scope", "note_style", "note_style_label", "metric_family", "entity_type", "metric_name",
                   "mean_value", "median_value", "std_value", "min_value", "max_value",
                   "n_patients_total", "n_non_null", "aggregation_level"]


def build_scores_overall_summary(df_scores_in: pd.DataFrame, df_verification_in: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = _summary_rows(df_scores_in, df_verification_in, {"summary_scope": "overall"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    return df[cols].sort_values(["metric_family", "entity_type", "metric_name"]).reset_index(drop=True)


def build_scores_summary_by_note_style(df_scores_in: pd.DataFrame, df_verification_in: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    if "note_style" in df_scores_in.columns:
        for (style, label), grp in df_scores_in.groupby(["note_style", "note_style_label"], dropna=False):
            rows += _summary_rows(grp, None, {"summary_scope": "by_note_style", "note_style": style, "note_style_label": label})
    if df_verification_in is not None and not df_verification_in.empty and "note_style" in df_verification_in.columns:
        for (style, label), grp in df_verification_in.groupby(["note_style", "note_style_label"], dropna=False):
            rows += _summary_rows(None, grp, {"summary_scope": "by_note_style", "note_style": style, "note_style_label": label})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    return df[cols].sort_values(["note_style", "metric_family", "entity_type", "metric_name"]).reset_index(drop=True)


def run_summaries(df_scores: pd.DataFrame, df_verification: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall = build_scores_overall_summary(df_scores, df_verification)
    by_style = build_scores_summary_by_note_style(df_scores, df_verification)
    if not overall.empty:
        overall.to_csv(settings.SCORES_DIR / "df_scores_overall_summary.csv", index=False)
    if not by_style.empty:
        by_style.to_csv(settings.SCORES_DIR / "df_scores_summary_by_note_style.csv", index=False)
    return overall, by_style
