"""Export : rassembler les sorties du pipeline en tables CSV et publier celles du benchmark.

Tout est relu depuis le disque (`data/`), sans appel au modèle. On peut donc
régénérer les exports à tout moment, par exemple après avoir changé une table
de synthèse, sans relancer les étapes coûteuses.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import settings
from src.evaluation import (
    add_difficulty_score,
    build_scores_overall_summary,
    build_scores_summary_by_note_style,
    dob_exact,
    error_types_by_note_style,
    gender_exact,
    hard_cases_by_note_style,
    metrics_from_sets,
    name_exact,
)
from src.fhir import index_filtered_dir
from src.utils import ensure_list, list_json, norm_text, read_json, safe_read_json, shorten, to_json_str
from src.verification import verification_by_note_style, verification_row

ERROR_TYPE_COLS = [
    f"error_type_{kind}_{field}"
    for kind in ("conditions", "medication_requests")
    for field in ("overall", "stage", "severity", "explanation")
] + ["error_type_pipeline_summary"]

# Organisation des CSV publiés dans results/.
RESULTS_LAYOUT = {
    "intermediates": ["df_gold", "df_notes", "df_recon"],
    "other_results": ["df_export_review", "df_gold_conditions_long", "df_gold_medreq_long", "df_index",
                      "df_recon_conditions_long", "df_recon_medreq_long"],
    "summaries/metrics": ["df_scores", "df_scores_overall_summary", "df_scores_summary_by_note_style"],
    "summaries/errors": ["df_error_analysis", "df_error_types_by_note_style_conditions",
                         "df_error_types_by_note_style_medication_requests", "df_hard_cases_by_note_style"],
    "summaries/verification": ["df_verification"],
}


# =========================
# Remise à zéro
# =========================


def reset_benchmark_outputs() -> List[str]:
    """Supprime scores, vérification et exports, pour relancer l'évaluation de zéro.

    Le dataset filtré, le gold, les notes et les reconstructions sont conservés :
    ce sont les étapes longues, et l'évaluation peut être refaite à partir d'elles.
    """
    removed: List[str] = []
    for d in (settings.SCORES_DIR, settings.LLM_ALIGN_DIR, settings.NOTE_ALIGN_DIR,
              settings.VERIFICATION_DIR, settings.EXPORT_DIR):
        if not d.exists():
            continue
        for pattern in ("*.json", "*.csv"):
            for p in sorted(d.glob(pattern)):
                p.unlink()
                removed.append(str(p))
    if settings.EXPORT_ZIP_PATH.exists():
        settings.EXPORT_ZIP_PATH.unlink()
        removed.append(str(settings.EXPORT_ZIP_PATH))
    return removed


# =========================
# Reconstruction des tables depuis le disque
# =========================


def _sort(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(cols).reset_index(drop=True) if (not df.empty and cols) else df


def _long(rows: List[dict], key: str, value_col: str) -> pd.DataFrame:
    """Une ligne par élément de liste, pour filtrer ou compter facilement dans un tableur."""
    out = [{"patient_id": row["patient_id"], "rank": i, value_col: v}
           for row in rows for i, v in enumerate(row[key])]
    return _sort(pd.DataFrame(out, columns=["patient_id", "rank", value_col]), ["patient_id", "rank"])


def _list_cols(label: str, values: list, n: int) -> dict:
    return {f"{label}_n": len(values), label: values,
            f"{label}_json": to_json_str(values), f"{label}_preview": "; ".join(values[:n])}


def _exact_only_score_row(pid: str, g: dict, r: dict, raw: dict) -> dict:
    """Scores calculables sans LLM, utilisés quand l'évaluation n'a pas été lancée."""
    row = {
        "patient_id": pid,
        "note_style": g.get("note_style"),
        "note_style_label": g.get("note_style_label"),
        "gold_full_name": g.get("full_name"),
        "gold_birthDate": g.get("birthDate"),
        "gold_gender": g.get("gender"),
        "raw_patient_name": raw.get("patient_name"),
        "raw_date_of_birth": raw.get("date_of_birth"),
        "raw_gender": raw.get("gender"),
        "name_exact_raw": name_exact(g.get("full_name"), raw.get("patient_name")),
        "dob_exact_raw": dob_exact(g.get("birthDate"), raw.get("date_of_birth")),
        "gender_exact_raw": gender_exact(g.get("gender"), raw.get("gender")),
    }
    for kind, short in (("conditions", "cond"), ("medication_requests", "med")):
        gold_set = {norm_text(x) for x in ensure_list(g.get(kind)) if x}
        pred_set = {norm_text(x) for x in ensure_list(r.get(kind)) if x}
        for k, v in metrics_from_sets(gold_set, pred_set).items():
            row[f"score_source_vs_recon_{kind}_exact_{k}"] = v
        row[f"gold_{short}_n"] = len(gold_set)
        row[f"pred_{short}_n"] = len(pred_set)
        row[f"raw_{short}_n"] = len(ensure_list(raw.get(kind)))
    return row


def _gold_rows(gold_by_pid: Dict[str, dict], n: int) -> List[dict]:
    rows = []
    for pid, g in gold_by_pid.items():
        conditions = ensure_list(g.get("conditions"))
        medreq = ensure_list(g.get("medication_requests"))
        rows.append({
            "patient_id": pid,
            "full_name": g.get("full_name"),
            "birthDate": g.get("birthDate"),
            "gender": g.get("gender"),
            "note_style": g.get("note_style"),
            "note_style_label": g.get("note_style_label"),
            "n_conditions": len(conditions),
            "conditions": conditions,
            "conditions_json": to_json_str(conditions),
            "conditions_preview": "; ".join(conditions[:n]),
            "n_medication_requests": len(medreq),
            "medication_requests": medreq,
            "medication_requests_json": to_json_str(medreq),
            "medreq_preview": "; ".join(medreq[:n]),
            "gold_file": str(settings.GOLD_DIR / f"{pid}.json"),
        })
    return rows


def _note_rows(gold_by_pid: Dict[str, dict]) -> List[dict]:
    rows = []
    if not settings.NOTES_DIR.exists():
        return rows
    for p in sorted(settings.NOTES_DIR.glob("*.txt")):
        pid = p.stem
        note = p.read_text(encoding="utf-8")
        raw_path = settings.RAW_NOTES_DIR / f"{pid}.txt"
        raw_note = raw_path.read_text(encoding="utf-8") if raw_path.exists() else note
        g = gold_by_pid.get(pid, {})
        rows.append({
            "patient_id": pid,
            "full_name": g.get("full_name"),
            "birthDate": g.get("birthDate"),
            "gender": g.get("gender"),
            "note_style": g.get("note_style"),
            "note_style_label": g.get("note_style_label"),
            "raw_note_text": raw_note,
            "raw_note_preview": shorten(raw_note, 240),
            "note_text": note,
            "note_preview": shorten(note, 240),
            "raw_chars": len(raw_note),
            "clean_chars": len(note),
            "note_file": str(p),
            "raw_note_file": str(raw_path) if raw_path.exists() else "",
        })
    return rows


def _recon_rows(gold_by_pid: Dict[str, dict], n: int):
    rows, recon_by_pid, raw_by_pid = [], {}, {}
    for p in list_json(settings.RECON_DIR):
        if p.name.endswith((".raw.json", ".clipped.json")):
            continue
        pid = p.stem
        r = read_json(p)
        raw_path = settings.RECON_DIR / f"{pid}.raw.json"
        clipped_path = settings.RECON_DIR / f"{pid}.clipped.json"
        raw = safe_read_json(raw_path) if raw_path.exists() else {}
        clipped = safe_read_json(clipped_path) if clipped_path.exists() else {}
        raw = raw if isinstance(raw, dict) else {}
        clipped = clipped if isinstance(clipped, dict) else {}
        recon_by_pid[pid], raw_by_pid[pid] = r, raw

        g = gold_by_pid.get(pid, {})
        row = {
            "patient_id": pid,
            "full_name": r.get("full_name"),
            "birthDate": r.get("birthDate"),
            "gender": r.get("gender"),
            "note_style": g.get("note_style"),
            "note_style_label": g.get("note_style_label"),
        }
        for kind in ("conditions", "medication_requests"):
            row.update(_list_cols(f"raw_{kind}", ensure_list(raw.get(kind)), n))
            row.update(_list_cols(kind, ensure_list(r.get(kind)), n))
            row.update(_list_cols(f"{kind}_clipped", ensure_list(clipped.get(kind)), n))
        row.update({
            "raw_patient_name": raw.get("patient_name"),
            "raw_date_of_birth": raw.get("date_of_birth"),
            "raw_gender": raw.get("gender"),
            "raw_file": str(raw_path) if raw_path.exists() else None,
            "recon_file": str(p),
            "recon_clipped_file": str(clipped_path) if clipped_path.exists() else None,
        })
        rows.append(row)
    return rows, recon_by_pid, raw_by_pid


def rebuild_datasets_from_cache() -> Dict[str, pd.DataFrame]:
    """Relit data/ en entier et reconstruit toutes les tables finales."""
    n = settings.PREVIEW_N
    exports: Dict[str, pd.DataFrame] = {"df_index": _sort(index_filtered_dir(settings.FILTERED_DIR), ["patient_id"])}

    gold_by_pid = {g.get("patient_id") or p.stem: g for p, g in ((p, read_json(p)) for p in list_json(settings.GOLD_DIR))}
    gold_rows = _gold_rows(gold_by_pid, n)
    exports["df_gold"] = _sort(pd.DataFrame(gold_rows), ["note_style", "patient_id"])
    exports["df_gold_conditions_long"] = _long(gold_rows, "conditions", "condition")
    exports["df_gold_medreq_long"] = _long(gold_rows, "medication_requests", "medication_request")

    exports["df_notes"] = _sort(pd.DataFrame(_note_rows(gold_by_pid)), ["note_style", "patient_id"])

    recon_rows, recon_by_pid, raw_by_pid = _recon_rows(gold_by_pid, n)
    exports["df_recon"] = _sort(pd.DataFrame(recon_rows), ["note_style", "patient_id"])
    exports["df_recon_conditions_long"] = _long(recon_rows, "conditions", "condition")
    exports["df_recon_medreq_long"] = _long(recon_rows, "medication_requests", "medication_request")

    score_rows = [obj for obj in map(read_json, list_json(settings.SCORES_DIR))
                  if isinstance(obj, dict) and "patient_id" in obj]
    if not score_rows:
        print("No per-patient score JSON found: falling back to exact source-vs-recon metrics only.")
        score_rows = [_exact_only_score_row(pid, gold_by_pid.get(pid, {}), recon_by_pid.get(pid, {}), raw_by_pid.get(pid, {}))
                      for pid in sorted(set(gold_by_pid) | set(recon_by_pid))]
    df_scores = _sort(pd.DataFrame(score_rows), ["note_style", "patient_id"])
    if not df_scores.empty:
        df_scores = add_difficulty_score(df_scores)
    exports["df_scores"] = df_scores

    verif_rows = [verification_row(obj, gold_by_pid.get(obj["patient_id"], {}))
                  for obj in map(read_json, list_json(settings.VERIFICATION_DIR))
                  if isinstance(obj, dict) and "patient_id" in obj and "verdict" in obj]
    df_verification = _sort(pd.DataFrame(verif_rows), ["note_style", "patient_id"])
    exports["df_verification"] = df_verification
    exports["df_verification_by_note_style"] = verification_by_note_style(df_verification)

    err_cols = [c for c in ["patient_id", *ERROR_TYPE_COLS] if c in df_scores.columns]
    exports["df_error_analysis"] = df_scores[err_cols].copy() if len(err_cols) > 1 else pd.DataFrame()
    exports["df_error_types_by_note_style_conditions"] = error_types_by_note_style(df_scores, "conditions")
    exports["df_error_types_by_note_style_medication_requests"] = error_types_by_note_style(df_scores, "medication_requests")
    exports["df_hard_cases_by_note_style"] = hard_cases_by_note_style(df_scores)
    if not df_scores.empty:
        exports["df_scores_overall_summary"] = build_scores_overall_summary(df_scores, df_verification)
        exports["df_scores_summary_by_note_style"] = build_scores_summary_by_note_style(df_scores, df_verification)

    return {k: v for k, v in exports.items() if v is not None and not v.empty}


# =========================
# Écriture
# =========================


def _zip_add_dir(z: zipfile.ZipFile, dir_path: Path, arc_prefix: str) -> None:
    if not dir_path.exists():
        return
    for p in sorted(dir_path.glob("**/*")):
        if p.is_file():
            z.write(p, arcname=f"{arc_prefix}/{p.relative_to(dir_path).as_posix()}")


def export_datasets(exports: Optional[Dict[str, pd.DataFrame]] = None, make_zip: bool = True) -> pd.DataFrame:
    """Écrit tous les CSV dans data/processed/exports/, avec un récapitulatif et, en option, une archive complète."""
    exports = exports if exports is not None else rebuild_datasets_from_cache()
    settings.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in exports.items():
        df.to_csv(settings.EXPORT_DIR / f"{name}.csv", index=False)

    df_review = pd.DataFrame([{
        "dataset_name": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "csv_path": str(settings.EXPORT_DIR / f"{name}.csv"),
    } for name, df in exports.items()]).sort_values("dataset_name").reset_index(drop=True)
    df_review.to_csv(settings.EXPORT_DIR / "df_export_review.csv", index=False)

    if make_zip:
        with zipfile.ZipFile(settings.EXPORT_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(settings.EXPORT_DIR.glob("*.csv")):
                z.write(p, arcname=f"exports/{p.name}")
            for d, prefix in ((settings.FILTERED_DIR, "filtered"), (settings.GOLD_DIR, "gold"),
                              (settings.NOTES_DIR, "notes"), (settings.RECON_DIR, "reconstruction"),
                              (settings.SCORES_DIR, "scores"), (settings.VERIFICATION_DIR, "verification")):
                _zip_add_dir(z, d, prefix)
        print("ZIP saved:", settings.EXPORT_ZIP_PATH)

    return df_review


def publish_results(results_dir: Optional[Path] = None) -> List[Path]:
    """Copie les CSV du benchmark dans results/, le dossier versionné du dépôt."""
    results_dir = Path(results_dir or settings.RESULTS_DIR)
    written = []
    for sub, names in RESULTS_LAYOUT.items():
        for name in names:
            src = settings.EXPORT_DIR / f"{name}.csv"
            if src.exists():
                dest = results_dir / sub / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                written.append(dest)
    return written


def build_patient_example(pid: str, dest_dir: Path) -> Path:
    """Copie toutes les sorties d'un patient dans un dossier, dans l'ordre du pipeline (voir results/examples/)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (settings.FILTERED_DIR / f"{pid}.json", "01_filtered.json"),
        (settings.GOLD_DIR / f"{pid}.json", "02_gold.json"),
        (settings.NOTES_DIR / f"{pid}.txt", "03_note.txt"),
        (settings.RECON_DIR / f"{pid}.json", "04_reconstruction.json"),
        (settings.SCORES_DIR / f"{pid}.json", "05_score_and_07_errors.json"),
        (settings.VERIFICATION_DIR / f"{pid}.json", "06_verification.json"),
    ]
    for src, name in pairs:
        if src.exists():
            shutil.copy2(src, dest_dir / name)
    return dest_dir
