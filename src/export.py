"""Étapes 10 et 11 : reset des sorties, export des datasets depuis le cache, publication dans results/.

Repris des cellules 10.1) et 11) du notebook original. Les fonctions qui y
étaient redéfinies avec un suffixe `_local` (norm_text_local, name_exact_local,
build_scores_* ...) réutilisent maintenant celles de `src/`.

Corrections par rapport au notebook :
- la branche « pas de JSON de scores » utilisait des variables non définies
  (gold_med / pred_med) : elle recalcule maintenant les scores exacts proprement ;
- la vérification relue depuis le cache est remise au format `verification_*`,
  ce qui permet de recalculer correctement la synthèse par style.
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
    error_types_by_note_style,
    hard_cases_by_note_style,
    metrics_from_sets,
)
from src.fhir import index_filtered_dir
from src.reconstruction import ensure_list
from src.utils import list_json, norm_text, read_json, shorten, to_json_str
from src.verification import verification_by_note_style, verification_row

ERROR_TYPE_COLS = [
    "error_type_conditions_overall",
    "error_type_conditions_stage",
    "error_type_conditions_severity",
    "error_type_conditions_explanation",
    "error_type_medication_requests_overall",
    "error_type_medication_requests_stage",
    "error_type_medication_requests_severity",
    "error_type_medication_requests_explanation",
    "error_type_pipeline_summary",
]

# =========================
# 10.1) Reset des sorties finales
# =========================


def clear_dir_files(dir_path: Path, patterns: List[str]) -> List[str]:
    if not dir_path.exists():
        return []
    removed = []
    for pattern in patterns:
        for p in sorted(dir_path.glob(pattern)):
            if p.is_file():
                p.unlink()
                removed.append(str(p))
    return removed


def reset_benchmark_outputs() -> List[str]:
    """Supprime scores / vérification / exports (pas le dataset filtré, le gold, les notes ni la reconstruction)."""
    removed: List[str] = []
    for d in [settings.SCORES_DIR, settings.LLM_ALIGN_DIR, settings.NOTE_ALIGN_DIR,
              settings.VERIFICATION_DIR, settings.EXPORT_DIR]:
        removed.extend(clear_dir_files(d, ["*.json", "*.csv"]))
    if settings.EXPORT_ZIP_PATH.exists():
        settings.EXPORT_ZIP_PATH.unlink()
        removed.append(str(settings.EXPORT_ZIP_PATH))
    return removed


# =========================
# 11) Reconstruction des datasets depuis le cache disque
# =========================


def _sort(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(cols).reset_index(drop=True) if (not df.empty and cols) else df


def _long(rows: List[dict], key: str, value_col: str) -> pd.DataFrame:
    out = []
    for row in rows:
        for i, v in enumerate(row[key]):
            out.append({"patient_id": row["patient_id"], "rank": i, value_col: v})
    return _sort(pd.DataFrame(out, columns=["patient_id", "rank", value_col]), ["patient_id", "rank"])


def load_gold_cache() -> Dict[str, dict]:
    out = {}
    for p in list_json(settings.GOLD_DIR):
        g = read_json(p)
        out[g.get("patient_id") or p.stem] = g
    return out


def _exact_only_score_row(pid: str, g: dict, r: dict, raw: dict) -> dict:
    """Repli quand aucun score JSON n'existe : seulement les métriques exactes source vs recon."""
    from src.evaluation import dob_exact, gender_exact, name_exact

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
        "name_exact_raw": name_exact(g.get("full_name") or "", raw.get("patient_name") or ""),
        "dob_exact_raw": dob_exact(g.get("birthDate") or "", raw.get("date_of_birth") or ""),
        "gender_exact_raw": gender_exact(g.get("gender") or "", raw.get("gender") or ""),
    }
    for kind, short in (("conditions", "cond"), ("medication_requests", "med")):
        gold_set = set(norm_text(x) for x in ensure_list(g.get(kind)) if x)
        pred_set = set(norm_text(x) for x in ensure_list(r.get(kind)) if x)
        m = metrics_from_sets(gold_set, pred_set)
        for k, v in m.items():
            row[f"score_source_vs_recon_{kind}_exact_{k}"] = v
        row[f"gold_{short}_n"] = len(gold_set)
        row[f"pred_{short}_n"] = len(pred_set)
        row[f"raw_{short}_n"] = len(ensure_list(raw.get(kind)))
    return row


def rebuild_datasets_from_cache() -> Dict[str, pd.DataFrame]:
    """Relit tout le cache (data/processed + data/gold_standard) et reconstruit les datasets finaux."""
    n = settings.PREVIEW_N
    exports: Dict[str, pd.DataFrame] = {}

    # 1) Index
    exports["df_index"] = _sort(index_filtered_dir(settings.FILTERED_DIR), ["patient_id"])

    # 2) Gold
    gold_by_pid = load_gold_cache()
    gold_rows = []
    for pid, g in gold_by_pid.items():
        conditions = ensure_list(g.get("conditions"))
        medreq = ensure_list(g.get("medication_requests"))
        gold_rows.append({
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
    exports["df_gold"] = _sort(pd.DataFrame(gold_rows), ["note_style", "patient_id"])
    exports["df_gold_conditions_long"] = _long(gold_rows, "conditions", "condition")
    exports["df_gold_medreq_long"] = _long(gold_rows, "medication_requests", "medication_request")

    # 3) Notes
    note_rows = []
    if settings.NOTES_DIR.exists():
        for p in sorted(settings.NOTES_DIR.glob("*.txt")):
            pid = p.stem
            note = p.read_text(encoding="utf-8")
            raw_path = settings.RAW_NOTES_DIR / f"{pid}.txt"
            raw_note = raw_path.read_text(encoding="utf-8") if raw_path.exists() else note
            g = gold_by_pid.get(pid, {})
            note_rows.append({
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
                "raw_chars": len(raw_note or ""),
                "clean_chars": len(note or ""),
                "note_file": str(p),
                "raw_note_file": str(raw_path) if raw_path.exists() else "",
            })
    exports["df_notes"] = _sort(pd.DataFrame(note_rows), ["note_style", "patient_id"])

    # 4) Reconstruction (vues clean + clipped + raw)
    recon_rows, recon_by_pid, raw_by_pid = [], {}, {}
    for p in list_json(settings.RECON_DIR):
        if p.name.endswith(".raw.json") or p.name.endswith(".clipped.json"):
            continue
        pid = p.stem
        r = read_json(p)
        raw_path = settings.RECON_DIR / f"{pid}.raw.json"
        clipped_path = settings.RECON_DIR / f"{pid}.clipped.json"
        raw = read_json(raw_path) if raw_path.exists() else {}
        clipped = read_json(clipped_path) if clipped_path.exists() else {}
        raw = raw if isinstance(raw, dict) else {}
        clipped = clipped if isinstance(clipped, dict) else {}
        recon_by_pid[pid], raw_by_pid[pid] = r, raw

        row = {
            "patient_id": pid,
            "full_name": r.get("full_name"),
            "birthDate": r.get("birthDate"),
            "gender": r.get("gender"),
            "note_style": gold_by_pid.get(pid, {}).get("note_style"),
            "note_style_label": gold_by_pid.get(pid, {}).get("note_style_label"),
        }
        for label, src, key in [
            ("raw_conditions", raw, "conditions"),
            ("conditions", r, "conditions"),
            ("conditions_clipped", clipped, "conditions"),
            ("raw_medication_requests", raw, "medication_requests"),
            ("medication_requests", r, "medication_requests"),
            ("medication_requests_clipped", clipped, "medication_requests"),
        ]:
            vals = ensure_list(src.get(key))
            row[f"{label}_n"] = len(vals)
            row[label] = vals
            row[f"{label}_json"] = to_json_str(vals)
            row[f"{label}_preview"] = "; ".join(vals[:n])
        row.update({
            "raw_patient_name": raw.get("patient_name"),
            "raw_date_of_birth": raw.get("date_of_birth"),
            "raw_gender": raw.get("gender"),
            "raw_file": str(raw_path) if raw_path.exists() else None,
            "recon_file": str(p),
            "recon_clipped_file": str(clipped_path) if clipped_path.exists() else None,
        })
        recon_rows.append(row)
    exports["df_recon"] = _sort(pd.DataFrame(recon_rows), ["note_style", "patient_id"])
    exports["df_recon_conditions_long"] = _long(recon_rows, "conditions", "condition")
    exports["df_recon_medreq_long"] = _long(recon_rows, "medication_requests", "medication_request")

    # 5) Scores
    score_rows = []
    for p in list_json(settings.SCORES_DIR):
        obj = read_json(p)
        if isinstance(obj, dict) and "patient_id" in obj:
            score_rows.append(obj)
    if not score_rows:
        print("No per-patient score JSON found: falling back to exact source-vs-recon metrics only.")
        pids = sorted(set(gold_by_pid) | set(recon_by_pid))
        score_rows = [_exact_only_score_row(pid, gold_by_pid.get(pid, {}), recon_by_pid.get(pid, {}), raw_by_pid.get(pid, {}))
                      for pid in pids]
    df_scores = _sort(pd.DataFrame(score_rows), ["note_style", "patient_id"])
    df_scores = add_difficulty_score(df_scores) if not df_scores.empty else df_scores
    exports["df_scores"] = df_scores

    # 6) Vérification
    verif_rows = []
    for p in list_json(settings.VERIFICATION_DIR):
        obj = read_json(p)
        if isinstance(obj, dict) and "patient_id" in obj and "verdict" in obj:
            verif_rows.append(verification_row(obj, gold_by_pid.get(obj["patient_id"], {})))
    df_verification = _sort(pd.DataFrame(verif_rows), ["note_style", "patient_id"])
    exports["df_verification"] = df_verification
    exports["df_verification_by_note_style"] = verification_by_note_style(df_verification)

    # 7) Analyse des erreurs et synthèses
    err_cols = [c for c in ["patient_id", *ERROR_TYPE_COLS] if c in df_scores.columns]
    exports["df_error_analysis"] = df_scores[err_cols].copy() if len(err_cols) > 1 else pd.DataFrame()
    exports["df_error_types_by_note_style_conditions"] = error_types_by_note_style(df_scores, "conditions")
    exports["df_error_types_by_note_style_medication_requests"] = error_types_by_note_style(df_scores, "medication_requests")
    exports["df_hard_cases_by_note_style"] = hard_cases_by_note_style(df_scores)
    exports["df_scores_overall_summary"] = build_scores_overall_summary(df_scores, df_verification) if not df_scores.empty else pd.DataFrame()
    exports["df_scores_summary_by_note_style"] = build_scores_summary_by_note_style(df_scores, df_verification) if not df_scores.empty else pd.DataFrame()

    return {k: v for k, v in exports.items() if v is not None and not v.empty}


def zip_add_dir(z: zipfile.ZipFile, dir_path: Path, arc_prefix: str, pattern: str = "**/*") -> None:
    if not dir_path.exists():
        return
    for p in sorted(dir_path.glob(pattern)):
        if p.is_file():
            z.write(p, arcname=f"{arc_prefix}/{p.relative_to(dir_path).as_posix()}")


def export_datasets(exports: Optional[Dict[str, pd.DataFrame]] = None, make_zip: bool = True) -> pd.DataFrame:
    """Écrit tous les CSV dans data/processed/exports/ (+ df_export_review et un zip complet)."""
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
            zip_add_dir(z, settings.FILTERED_DIR, "filtered")
            zip_add_dir(z, settings.GOLD_DIR, "gold")
            zip_add_dir(z, settings.NOTES_DIR, "notes")
            zip_add_dir(z, settings.RECON_DIR, "reconstruction")
            zip_add_dir(z, settings.SCORES_DIR, "scores")
            zip_add_dir(z, settings.VERIFICATION_DIR, "verification")
        print("ZIP saved:", settings.EXPORT_ZIP_PATH)

    return df_review


# =========================
# Publication dans results/ (dossier versionné)
# =========================

RESULTS_LAYOUT = {
    "intermediates": ["df_gold", "df_notes", "df_recon"],
    "other_results": ["df_export_review", "df_gold_conditions_long", "df_gold_medreq_long", "df_index",
                      "df_recon_conditions_long", "df_recon_medreq_long"],
    "summaries/metrics": ["df_scores", "df_scores_overall_summary", "df_scores_summary_by_note_style"],
    "summaries/errors": ["df_error_analysis", "df_error_types_by_note_style_conditions",
                         "df_error_types_by_note_style_medication_requests", "df_hard_cases_by_note_style"],
    "summaries/verification": ["df_verification"],
}


def publish_results(results_dir: Optional[Path] = None) -> List[Path]:
    """Copie les CSV de data/processed/exports/ vers results/ en respectant l'organisation du dépôt."""
    results_dir = Path(results_dir or settings.RESULTS_DIR)
    written = []
    for sub, names in RESULTS_LAYOUT.items():
        for name in names:
            src = settings.EXPORT_DIR / f"{name}.csv"
            if not src.exists():
                continue
            dest = results_dir / sub / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            written.append(dest)
    return written


def build_patient_example(pid: str, dest_dir: Path) -> Path:
    """Crée un dossier d'exemple patient (01_filtered ... 06_verification), comme results/examples/."""
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
