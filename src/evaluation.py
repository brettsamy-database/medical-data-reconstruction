"""Évaluation : scores par patient, analyse des erreurs et tables de synthèse.

Chaque reconstruction est comparée à deux références :
- la note (note vs recon) : le modèle a-t-il bien extrait ce qui est écrit ?
- le gold source (source vs recon) : que reste-t-il du dossier d'origine après le
  passage par la note ? Cette comparaison cumule les pertes de la génération et
  celles de l'extraction.

Chaque comparaison est faite de deux façons :
- exacte : même libellé après normalisation (casse, espaces) ;
- sémantique : le LLM juge si deux libellés désignent la même chose
  (« anemia » = « Anemia (disorder) »).

L'écart entre les deux lectures, et entre les deux références, permet de dire
à quelle étape une information a été perdue (voir `classify_error_type`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import settings
from src.llm import LLMResult, get_llm, load_json_lenient
from src.reconstruction import postprocess_list
from src.utils import ensure_list, norm_text, read_cached, signature, write_json

KINDS = ("conditions", "medication_requests")
SHORT = {"conditions": "cond", "medication_requests": "med"}

# Le juge sémantique reçoit au plus 25 candidats : au-delà, MedGemma 4B se perd dans la liste.
MAX_MATCH_CANDIDATES = 25


def max_items(kind: str) -> int:
    return settings.MAX_CONDITIONS_PER_PATIENT if kind == "conditions" else settings.MAX_MEDICATION_REQUESTS_PER_PATIENT


# =========================
# Métriques
# =========================


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    """Précision, rappel, F1. Un patient sans rien à trouver et rien de trouvé compte comme parfait."""
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if tp == fp == fn == 0:
        f1 = 1.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def metrics_from_sets(gold_set: set, pred_set: set) -> dict:
    return metrics_from_counts(len(gold_set & pred_set), len(pred_set - gold_set), len(gold_set - pred_set))


def name_exact(gold_name: str, pred_name: str) -> int:
    return int(bool(norm_text(gold_name)) and norm_text(gold_name) == norm_text(pred_name))


def dob_exact(gold_dob: str, pred_dob: str) -> int:
    g, p = (gold_dob or "").strip(), (pred_dob or "").strip()
    return int(bool(g) and g == p)


def gender_exact(gold_gender: str, pred_gender: str) -> int:
    return int(bool(norm_text(gold_gender)) and norm_text(gold_gender) == norm_text(pred_gender))


def _json_dict(text: str) -> dict:
    obj = load_json_lenient(text)
    return obj if isinstance(obj, dict) else {}


def exact_items_supported_by_note(note_text: str, source_items: List[str]) -> List[str]:
    """Éléments du gold recopiés mot pour mot (après normalisation) dans la note."""
    note_norm = norm_text(note_text)
    return [item for item in source_items or [] if norm_text(item) and norm_text(item) in note_norm]


# =========================
# Source vs recon, sémantique
# =========================
# Chaque élément reconstruit est d'abord cherché tel quel dans le gold. S'il n'y
# est pas, le LLM choisit l'élément du gold qui lui correspond, ou aucun.
# Un élément du gold ne peut être apparié qu'une fois.


def _llm_judge_match_index(target: str, candidates: List[str], llm=None) -> Tuple[int, float]:
    llm = llm or get_llm()
    if not candidates:
        return -1, 0.0
    candidates = candidates[:MAX_MATCH_CANDIDATES]

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
    prompt += "".join(f"{i}. {c}\n" for i, c in enumerate(candidates))
    prompt += "\nJSON:"

    obj = _json_dict(llm.generate(
        prompt,
        max_new_tokens=settings.SEMANTIC_MATCH_MAX_NEW_TOKENS,
        temperature=settings.SCORING_TEMPERATURE,
        do_sample=settings.SCORING_DO_SAMPLE,
    ))
    try:
        idx = int(obj.get("match_index", -1))
    except (TypeError, ValueError):
        idx = -1
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0

    if not -1 <= idx < len(candidates):
        idx = -1
    return idx, min(max(conf, 0.0), 1.0)


def semantic_align_and_score(pid: str, kind: str, pred_items: List[str], gold_items: List[str], llm=None) -> dict:
    llm = llm or get_llm()
    pred_items = [x for x in pred_items if x]
    gold_items = [x for x in gold_items if x]
    gold_norm = [norm_text(x) for x in gold_items]
    pred_norm = [norm_text(x) for x in pred_items]

    cache_path = settings.LLM_ALIGN_DIR / f"{pid}.{kind}.json"
    sig = signature({"pred": pred_norm, "gold": gold_norm})
    cached = read_cached(cache_path, sig)
    if cached:
        return cached

    used_gold = set()
    matches = []
    for p_raw, p in zip(pred_items, pred_norm):
        match_idx, conf = -1, 1.0
        exact = [i for i, g in enumerate(gold_norm) if g == p and i not in used_gold]
        if p and exact:
            match_idx = exact[0]
        elif llm.available():
            match_idx, conf = _llm_judge_match_index(p_raw, gold_items, llm=llm)
            if match_idx in used_gold:
                match_idx, conf = -1, 0.0
        if match_idx >= 0:
            used_gold.add(match_idx)
        matches.append({
            "pred": p_raw,
            "match_index": int(match_idx),
            "matched_gold": gold_items[match_idx] if match_idx >= 0 else None,
            "confidence": float(conf),
        })

    tp = len(used_gold)
    m = metrics_from_counts(tp, len(pred_items) - tp, len(gold_items) - tp)
    out = {
        "patient_id": pid,
        "kind": kind,
        "signature": sig,
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "precision_semantic": m["precision"],
        "recall_semantic": m["recall"],
        "f1_semantic": m["f1"],
        "pred_n": len(pred_items),
        "gold_n": len(gold_items),
        "matches": matches,
    }
    write_json(cache_path, out)
    return out


# =========================
# Note vs recon, sémantique
# =========================
# Ici il n'y a pas de liste de référence : seulement la note. On demande au LLM
# quels éléments extraits la note justifie, et ce qu'elle contient en plus.


def note_consistency_prompt(note_text: str, kind: str, pred_items: List[str]) -> str:
    label = "conditions" if kind == "conditions" else "medication requests"
    item_lines = "\n".join(f"{i}. {x}" for i, x in enumerate(pred_items)) or "(none)"
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


def _note_consistency_result(pid: str, kind: str, sig: str, pred_items: List[str],
                             supported: List[int], missing: List[str]) -> dict:
    supported_set = set(supported)
    supported_items = [pred_items[i] for i in supported]
    unsupported_items = [x for i, x in enumerate(pred_items) if i not in supported_set]
    m = metrics_from_counts(len(supported_items), len(unsupported_items), len(missing))
    return {
        "patient_id": pid,
        "kind": kind,
        "signature": sig,
        "supported_n": m["tp"],
        "unsupported_n": m["fp"],
        "missing_n": m["fn"],
        "pred_n": len(pred_items),
        "precision_note_consistency": m["precision"],
        "recall_note_consistency": m["recall"],
        "f1_note_consistency": m["f1"],
        "supported_items": supported_items,
        "unsupported_items": unsupported_items,
        "missing_items": missing,
    }


def note_vs_recon_semantic_score(pid: str, kind: str, note_text: str, pred_items: List[str], llm=None) -> dict:
    llm = llm or get_llm()
    pred_items = [x for x in pred_items or [] if x]
    note_text = (note_text or "").strip()

    cache_path = settings.NOTE_ALIGN_DIR / f"{pid}.{kind}.json"
    sig = signature({"note_text": note_text, "kind": kind, "pred_items": pred_items})
    cached = read_cached(cache_path, sig)
    if cached:
        return cached

    supported: List[int] = []
    missing: List[str] = []
    if (note_text or pred_items) and llm.available():
        obj = _json_dict(llm.generate(
            note_consistency_prompt(note_text, kind, pred_items),
            max_new_tokens=settings.NOTE_CONSISTENCY_MAX_NEW_TOKENS,
            temperature=settings.SCORING_TEMPERATURE,
            do_sample=settings.SCORING_DO_SAMPLE,
        ))
        for x in ensure_list(obj.get("supported_indexes")):
            try:
                idx = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(pred_items) and idx not in supported:
                supported.append(idx)
        missing = [str(x).strip() for x in ensure_list(obj.get("missing_items")) if str(x).strip()]
        missing = postprocess_list(missing, max_items(kind), filter_noisy=(kind == "conditions"))

    out = _note_consistency_result(pid, kind, sig, pred_items, supported, missing)
    write_json(cache_path, out)
    return out


# =========================
# Table des scores
# =========================


def _metric_cols(prefix: str, m: dict, suffix: str = "") -> dict:
    return {f"{prefix}_{k}{suffix}": (int(m[k]) if k in ("tp", "fp", "fn") else float(m[k]))
            for k in ("tp", "fp", "fn", "precision", "recall", "f1")}


def _note_sem_as_metrics(d: dict) -> dict:
    return {"tp": d["supported_n"], "fp": d["unsupported_n"], "fn": d["missing_n"],
            "precision": d["precision_note_consistency"], "recall": d["recall_note_consistency"],
            "f1": d["f1_note_consistency"]}


def _source_sem_as_metrics(d: dict) -> dict:
    return {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
            "precision": d["precision_semantic"], "recall": d["recall_semantic"], "f1": d["f1_semantic"]}


def score_patient(gold: dict, pred_clean: dict, raw: dict, note_text: str, llm=None) -> dict:
    """Toutes les métriques d'un patient : une ligne de `df_scores`."""
    pid = gold["patient_id"]
    row = {
        "patient_id": pid,
        "note_style": gold.get("note_style"),
        "note_style_label": gold.get("note_style_label"),
        # Démographie : extraite de la note, comparée au gold, mais jamais reprise dans recon_clean.
        "gold_full_name": gold.get("full_name"),
        "gold_birthDate": gold.get("birthDate"),
        "gold_gender": gold.get("gender"),
        "raw_patient_name": raw.get("patient_name"),
        "raw_date_of_birth": raw.get("date_of_birth"),
        "raw_gender": raw.get("gender"),
        "name_exact_raw": name_exact(gold.get("full_name"), raw.get("patient_name")),
        "dob_exact_raw": dob_exact(gold.get("birthDate"), raw.get("date_of_birth")),
        "gender_exact_raw": gender_exact(gold.get("gender"), raw.get("gender")),
    }

    res: Dict[str, dict] = {}
    for kind in KINDS:
        filt = kind == "conditions"
        gold_items = gold.get(kind) or []
        # Le gold passe par le même nettoyage que la reconstruction, sans plafond : il est déjà plafonné.
        gold_list = postprocess_list(gold_items, len(gold_items), filter_noisy=filt)
        pred_list = postprocess_list(pred_clean.get(kind) or [], max_items(kind), filter_noisy=filt)
        gold_set = {norm_text(x) for x in gold_list}
        pred_set = {norm_text(x) for x in pred_list}
        # En note vs recon exact, la référence est la part du gold recopiée telle quelle dans la note.
        note_exact_set = {norm_text(x) for x in exact_items_supported_by_note(note_text, gold_list)}
        res[kind] = {
            "gold_list": gold_list,
            "pred_list": pred_list,
            "note_exact_set": note_exact_set,
            "note_exact": metrics_from_sets(note_exact_set, pred_set),
            "note_sem": _note_sem_as_metrics(note_vs_recon_semantic_score(pid, kind, note_text, pred_list, llm=llm)),
            "source_exact": metrics_from_sets(gold_set, pred_set),
            "source_sem": _source_sem_as_metrics(semantic_align_and_score(pid, kind, pred_list, gold_list, llm=llm)),
        }

    for key, family, mode, suffix in (("note_exact", "note_vs_recon", "exact", ""),
                                      ("note_sem", "note_vs_recon", "semantic", "_llm"),
                                      ("source_exact", "source_vs_recon", "exact", ""),
                                      ("source_sem", "source_vs_recon", "semantic", "_llm")):
        for kind in KINDS:
            row.update(_metric_cols(f"score_{family}_{kind}_{mode}", res[kind][key], suffix))

    for kind in KINDS:
        row[f"note_vs_recon_note_{kind}_exact_source_items_detected_n"] = len(res[kind]["note_exact_set"])
    for kind in KINDS:
        short = SHORT[kind]
        row[f"gold_{short}_n"] = len(res[kind]["gold_list"])
        row[f"pred_{short}_n"] = len(res[kind]["pred_list"])
        row[f"raw_{short}_n"] = len(ensure_list(raw.get(kind)))
    return row


def score_all(patients: List[dict], recon_clean: Dict[str, dict], recon_raw: Dict[str, dict],
              notes: Dict[str, LLMResult], llm=None) -> pd.DataFrame:
    """Construit `df_scores` et écrit un JSON par patient dans data/processed/scores/."""
    gold_by_pid = {p["patient_id"]: p for p in patients}
    rows = []
    for pid in sorted(gold_by_pid):
        note_text = notes[pid].note if pid in notes else ""
        rows.append(score_patient(gold_by_pid[pid], recon_clean.get(pid) or {},
                                  recon_raw.get(pid) or {}, note_text or "", llm=llm))

    df_scores = pd.DataFrame(rows)
    if df_scores.empty:
        return df_scores
    for r in rows:
        write_json(settings.SCORES_DIR / f'{r["patient_id"]}.json', r)
    df_scores.sort_values("patient_id").to_csv(settings.SCORES_DIR / "df_scores.csv", index=False)
    return df_scores.sort_values(["note_style", "patient_id"]).reset_index(drop=True)


def print_mean_scores(df_scores: pd.DataFrame) -> None:
    cols = ["name_exact_raw", "dob_exact_raw", "gender_exact_raw"]
    cols += [f"score_{family}_{kind}_{mode}_f1{suffix}"
             for family in ("note_vs_recon", "source_vs_recon")
             for mode, suffix in (("exact", ""), ("semantic", "_llm"))
             for kind in KINDS]
    for col in cols:
        if col in df_scores.columns:
            print(f"Mean {col}: {float(df_scores[col].mean()):.3f}")


# =========================
# Analyse des erreurs
# =========================


def classify_error_type(
    gold_n: int,
    pred_n: int,
    f1_exact: float,
    f1_semantic: float,
    note_f1: float,
    fp_semantic: int,
    fn_semantic: int,
    note_unsupported_n: int,
    note_missing_n: int,
) -> dict:
    """Situe l'erreur dominante d'un patient dans le pipeline.

    Les règles se lisent dans l'ordre, la première qui s'applique l'emporte :
    - si note vs recon est bon mais source vs recon mauvais, l'information
      manquait déjà dans la note : la perte vient de la génération ;
    - si c'est l'inverse, la note était complète et c'est l'extraction qui a
      raté ou inventé des éléments ;
    - si seul le score exact est mauvais, il s'agit d'une reformulation, pas
      d'une erreur de fond ;
    - sinon, l'erreur est répartie sur plusieurs étapes.
    Le seuil de 0.9 laisse passer un élément manqué sur une dizaine.
    """
    if gold_n == 0 and pred_n == 0:
        overall, stage = "no_target_and_no_prediction", "none"
    elif gold_n > 0 and pred_n == 0:
        overall = "complete_omission"
        stage = "note_to_reconstruction" if note_missing_n > 0 else "source_to_note_or_reconstruction"
    elif f1_exact == f1_semantic == note_f1 == 1.0:
        overall, stage = "perfect_reconstruction", "none"
    elif f1_exact < 1.0 and f1_semantic == 1.0 and note_f1 == 1.0:
        overall, stage = "exact_wording_gap_only", "canonicalization_exact_matching"
    elif note_f1 >= 0.9 and f1_semantic < 0.9:
        overall, stage = "source_to_note_gap_or_generation_loss", "source_to_note"
    elif f1_semantic >= 0.9 and note_f1 < 0.9:
        if note_missing_n > 0 and note_unsupported_n == 0:
            overall = "under_extraction_from_note"
        elif note_unsupported_n > 0 and note_missing_n == 0:
            overall = "over_extraction_from_note"
        else:
            overall = "mixed_note_to_reconstruction_error"
        stage = "note_to_reconstruction"
    elif f1_semantic > f1_exact and note_f1 >= 0.9:
        overall, stage = "canonicalization_or_exact_match_gap", "canonicalization_exact_matching"
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

    # La sévérité suit la plus mauvaise des deux lectures sémantiques.
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
    note_sem = f"score_note_vs_recon_{kind}_semantic"
    source_sem = f"score_source_vs_recon_{kind}_semantic"
    return classify_error_type(
        gold_n=int(row[f"gold_{SHORT[kind]}_n"]),
        pred_n=int(row[f"pred_{SHORT[kind]}_n"]),
        f1_exact=float(row[f"score_source_vs_recon_{kind}_exact_f1"]),
        f1_semantic=float(row[f"{source_sem}_f1_llm"]),
        note_f1=float(row[f"{note_sem}_f1_llm"]),
        fp_semantic=int(row[f"{source_sem}_fp_llm"]),
        fn_semantic=int(row[f"{source_sem}_fn_llm"]),
        note_unsupported_n=int(row[f"{note_sem}_fp_llm"]),
        note_missing_n=int(row[f"{note_sem}_fn_llm"]),
    )


def build_error_analysis(df_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_scores.iterrows():
        err = {kind: _classify_row(row, kind) for kind in KINDS}
        c, m = err["conditions"], err["medication_requests"]
        same = c["overall"] == m["overall"] and c["stage"] == m["stage"]
        out = {"patient_id": row["patient_id"]}
        for kind in KINDS:
            for field in ("overall", "stage", "severity", "explanation"):
                out[f"error_type_{kind}_{field}"] = err[kind][field]
        out["error_type_pipeline_summary"] = (
            c["overall"] if same else f"conditions={c['overall']} | medication_requests={m['overall']}"
        )
        rows.append(out)
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


# Le score de difficulté d'un patient est la moyenne de ses quatre F1 sémantiques.
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
    """Les `top_n` patients au plus faible score de difficulté, pour chaque style de note."""
    if not {"note_style", "note_style_label", "patient_id", "difficulty_score"}.issubset(df_scores.columns):
        return pd.DataFrame()
    cols = ["note_style", "note_style_label", "patient_id", "difficulty_score", *DIFFICULTY_COLS,
            "error_type_conditions_overall", "error_type_medication_requests_overall", "error_type_pipeline_summary"]
    cols = [c for c in cols if c in df_scores.columns]
    return (
        df_scores.sort_values(["note_style", "difficulty_score", "patient_id"])
        .groupby(["note_style", "note_style_label"], dropna=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )


def run_error_analysis(df_scores: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Classe les erreurs, enrichit df_scores et les JSON patient, et écrit les tables par style."""
    df_error_analysis = build_error_analysis(df_scores)
    df_scores = df_scores.merge(df_error_analysis, on="patient_id", how="left")

    out_dir = settings.SCORES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df_error_analysis.to_csv(out_dir / "df_error_analysis.csv", index=False)
    df_scores.sort_values("patient_id").to_csv(out_dir / "df_scores.csv", index=False)

    for _, r in df_scores.iterrows():
        payload = {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in r.to_dict().items()}
        write_json(out_dir / f'{r["patient_id"]}.json', payload)

    df_scores = add_difficulty_score(df_scores)
    tables = {
        "df_error_types_by_note_style_conditions": error_types_by_note_style(df_scores, "conditions"),
        "df_error_types_by_note_style_medication_requests": error_types_by_note_style(df_scores, "medication_requests"),
        "df_hard_cases_by_note_style": hard_cases_by_note_style(df_scores),
    }
    for name, df in tables.items():
        if not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)

    return {"df_scores": df_scores, "df_error_analysis": df_error_analysis, **tables}


# =========================
# Tables de synthèse
# =========================

SCORE_SUMMARY_METRIC_SPECS = (
    [("demographics", "demographics", m) for m in ("name_exact_raw", "dob_exact_raw", "gender_exact_raw")]
    + [(family, kind, f"score_{family}_{kind}_{mode}_{metric}{suffix}")
       for family in ("note_vs_recon", "source_vs_recon")
       for kind in KINDS
       for mode, suffix in (("exact", ""), ("semantic", "_llm"))
       for metric in ("precision", "recall", "f1")]
)

VERDICT_RATE_SPECS = [
    ("verification_consistent_rate", "consistent"),
    ("verification_minor_issues_rate", "minor_issues"),
    ("verification_major_issues_rate", "major_issues"),
    ("verification_unknown_rate", "unknown"),
]
VERIFICATION_MEAN_SPECS = [
    ("verification_mean_confidence", "verification_confidence"),
    ("verification_mean_issue_total_n", "verification_issue_total_n"),
]

SUMMARY_COLUMNS = ["summary_scope", "note_style", "note_style_label", "metric_family", "entity_type", "metric_name",
                   "mean_value", "median_value", "std_value", "min_value", "max_value",
                   "n_patients_total", "n_non_null", "aggregation_level"]


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

    def add(family: str, entity: str, metric: str, series: pd.Series, n_total: int) -> None:
        rows.append({**base, "metric_family": family, "entity_type": entity, "metric_name": metric,
                     **_stats_row(series), "n_patients_total": n_total})

    if df_scores_grp is not None:
        for family, entity, metric in SCORE_SUMMARY_METRIC_SPECS:
            if metric in df_scores_grp.columns:
                series = pd.to_numeric(df_scores_grp[metric], errors="coerce").dropna()
                if not series.empty:
                    add(family, entity, metric, series, len(df_scores_grp))

    if df_verif_grp is not None and not df_verif_grp.empty:
        n_total = len(df_verif_grp)
        for metric, verdict in VERDICT_RATE_SPECS:
            add("verification", "all", metric, (df_verif_grp["verification_verdict"].fillna("") == verdict).astype(float), n_total)
        for metric, col in VERIFICATION_MEAN_SPECS:
            if col in df_verif_grp.columns:
                series = pd.to_numeric(df_verif_grp[col], errors="coerce").dropna()
                if not series.empty:
                    add("verification", "all", metric, series, n_total)
    return rows


def _summary_frame(rows: List[dict], sort_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[[c for c in SUMMARY_COLUMNS if c in df.columns]].sort_values(sort_cols).reset_index(drop=True)


def build_scores_overall_summary(df_scores_in: pd.DataFrame, df_verification_in: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = _summary_rows(df_scores_in, df_verification_in, {"summary_scope": "overall"})
    return _summary_frame(rows, ["metric_family", "entity_type", "metric_name"])


def build_scores_summary_by_note_style(df_scores_in: pd.DataFrame, df_verification_in: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    style_keys = ["note_style", "note_style_label"]
    if "note_style" in df_scores_in.columns:
        for (style, label), grp in df_scores_in.groupby(style_keys, dropna=False):
            rows += _summary_rows(grp, None, {"summary_scope": "by_note_style", "note_style": style, "note_style_label": label})
    if df_verification_in is not None and not df_verification_in.empty and "note_style" in df_verification_in.columns:
        for (style, label), grp in df_verification_in.groupby(style_keys, dropna=False):
            rows += _summary_rows(None, grp, {"summary_scope": "by_note_style", "note_style": style, "note_style_label": label})
    return _summary_frame(rows, ["note_style", "metric_family", "entity_type", "metric_name"])


def run_summaries(df_scores: pd.DataFrame, df_verification: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall = build_scores_overall_summary(df_scores, df_verification)
    by_style = build_scores_summary_by_note_style(df_scores, df_verification)
    if not overall.empty:
        overall.to_csv(settings.SCORES_DIR / "df_scores_overall_summary.csv", index=False)
    if not by_style.empty:
        by_style.to_csv(settings.SCORES_DIR / "df_scores_summary_by_note_style.csv", index=False)
    return overall, by_style
