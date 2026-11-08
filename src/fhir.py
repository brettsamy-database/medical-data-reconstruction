"""Étapes 1 à 3 : téléchargement Synthea, préfiltrage FHIR et extraction patient.

Repris des cellules 2), 3A), 3B), 3C), 4A) et 4B) du notebook original.
Aucun LLM n'intervient ici.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm

from config import settings
from src.utils import first, norm_text, safe_read_json

# =========================
# 2) Télécharger et extraire Synthea
# =========================


def download_file(url: str, dest: Path, timeout_s: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()


def ensure_zip(zip_path: Optional[Path] = None, urls: Optional[List[str]] = None) -> Path:
    zip_path = zip_path or settings.ZIP_PATH
    urls = urls or settings.SYNTHEA_ZIP_URLS
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print("Zip already present:", zip_path.resolve())
        return zip_path
    last_err = None
    for url in urls:
        try:
            print("Trying:", url)
            download_file(url, zip_path)
            return zip_path
        except Exception as e:  # noqa: BLE001
            last_err = e
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:  # noqa: BLE001
                    pass
    raise RuntimeError(f"Could not download zip. Last error: {last_err}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        print("Already extracted:", extract_dir.resolve())
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    for p in extract_dir.iterdir():
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    print("Extracted to:", extract_dir.resolve())


def download_synthea() -> Path:
    """Télécharge (si besoin) puis extrait le zip Synthea dans data/raw/."""
    zip_path = ensure_zip(settings.ZIP_PATH)
    extract_zip(zip_path, settings.RAW_DIR)
    return settings.RAW_DIR


def reset_filtered_dataset() -> None:
    """Équivalent de la cellule « Run once to reset everything » du notebook."""
    shutil.rmtree(settings.FILTERED_DIR, ignore_errors=True)


# =========================
# 3A) Fonctions utilitaires et règles de préfiltrage
# =========================


def find_patient(bundle: dict) -> Optional[dict]:
    for e in bundle.get("entry", []) or []:
        r = e.get("resource", {})
        if r.get("resourceType") == "Patient":
            return r
    return None


def iter_resources(bundle: dict, resource_type: str) -> List[dict]:
    out = []
    for e in bundle.get("entry", []) or []:
        r = e.get("resource", {})
        if r.get("resourceType") == resource_type:
            out.append(r)
    return out


def cond_display(cond: dict) -> Optional[str]:
    code = cond.get("code", {}) or {}
    coding = first(code.get("coding", []), {}) or {}
    return coding.get("display") or code.get("text")


def cond_system_code(cond: dict) -> Tuple[str, str]:
    code = cond.get("code", {}) or {}
    coding = first(code.get("coding", []), {}) or {}
    return (coding.get("system") or "", coding.get("code") or "")


def is_social_history(cond: dict) -> bool:
    # Synthea stocke souvent les SDOH dans Condition. La catégorie est le signal déterministe le plus fiable.
    for cat in cond.get("category", []) or []:
        for coding in (cat.get("coding", []) or []):
            code = norm_text(coding.get("code"))
            disp = norm_text(coding.get("display"))
            if code == "social-history":
                return True
            # permissif : toute catégorie explicitement social / sdoh est considérée comme du bruit
            if "social" in code or "social" in disp or "sdoh" in code or "sdoh" in disp:
                return True
    return False


def is_noisy_display(disp: Optional[str]) -> bool:
    """Préfiltre conservateur : ne retire que les motifs non médicaux / SDoH évidents."""
    if not settings.PREFILTER_ENABLED:
        return False
    d = norm_text(disp)
    if not d:
        return False
    for sub in (settings.PREFILTER_DROP_SUBSTRINGS or []):
        subn = norm_text(sub)
        if subn and subn in d:
            return True
    return False


def is_medical_condition(cond: dict) -> bool:
    """Règle du préfiltre (étape A) : garder les conditions médicales codées, retirer les SDoH évidents."""
    if settings.PREFILTER_DROP_SOCIAL_HISTORY and is_social_history(cond):
        return False

    disp = cond_display(cond)
    if is_noisy_display(disp):
        return False

    system, code = cond_system_code(cond)
    if not system or not code:
        return False
    if system not in settings.ALLOWED_CONDITION_SYSTEMS:
        return False
    return True


def dedupe_limit_conditions(conds: List[dict], max_items: int) -> List[dict]:
    out = []
    seen = set()
    for c in conds:
        system, code = cond_system_code(c)
        key = f"{norm_text(system)}|{norm_text(code)}"
        if not system or not code:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_items:
            break
    return out


def med_display(medreq: dict) -> Optional[str]:
    med = medreq.get("medicationCodeableConcept", {}) or {}
    coding = first(med.get("coding", []), {}) or {}
    return coding.get("display") or med.get("text")


def med_system_code(medreq: dict) -> Tuple[str, str]:
    med = medreq.get("medicationCodeableConcept", {}) or {}
    coding = first(med.get("coding", []), {}) or {}
    return (coding.get("system") or "", coding.get("code") or "")


def dedupe_limit_medreqs(meds: List[dict], max_items: int) -> List[dict]:
    out = []
    seen = set()
    for m in meds:
        system, code = med_system_code(m)
        key = f"{norm_text(system)}|{norm_text(code)}"
        if not system or not code:
            # repli sur le libellé
            disp = norm_text(med_display(m))
            if not disp:
                continue
            key = disp
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
        if len(out) >= max_items:
            break
    return out


# =========================
# 3B) Construction du dataset filtré (cache disque)
# =========================


def index_filtered_dir(out_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        obj = safe_read_json(p)
        rows.append({
            "patient_id": p.stem,
            "file_path": str(p),
            "n_conditions": len(iter_resources(obj, "Condition")) if obj else 0,
            "n_medication_requests": len(iter_resources(obj, "MedicationRequest")) if obj else 0,
        })
    return pd.DataFrame(rows, columns=["patient_id", "file_path", "n_conditions", "n_medication_requests"])


def build_filtered_dataset(raw_dir: Optional[Path] = None, out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Un seul passage sur tous les bundles : garde Patient + Conditions médicales + MedicationRequests."""
    raw_dir = Path(raw_dir or settings.RAW_DIR)
    out_dir = Path(out_dir or settings.FILTERED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".filtered_ok"
    if marker.exists():
        print("Filtered dataset already present:", out_dir.resolve())
        print("Note: if you changed PREFILTER_* settings, call reset_filtered_dataset() to rebuild.")
        return index_filtered_dir(out_dir)

    rows = []
    json_files = sorted(raw_dir.rglob("*.json"))
    for p in tqdm(json_files, desc="Filtering Bundles (one pass)"):
        obj = safe_read_json(p)
        if not obj or obj.get("resourceType") != "Bundle":
            continue
        patient = find_patient(obj)
        if not patient:
            continue
        pid = patient.get("id")
        if not pid:
            continue

        conds = [c for c in iter_resources(obj, "Condition") if is_medical_condition(c)]
        conds = dedupe_limit_conditions(conds, max_items=settings.PREFILTER_MAX_CONDITION_CANDIDATES)

        meds = iter_resources(obj, "MedicationRequest")
        meds = dedupe_limit_medreqs(meds, max_items=settings.PREFILTER_MAX_MEDREQ_CANDIDATES)

        filtered = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": patient}] + [{"resource": c} for c in conds] + [{"resource": m} for m in meds],
        }
        out_path = out_dir / f"{pid}.json"
        out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")

        rows.append({"patient_id": pid, "file_path": str(out_path), "n_conditions": len(conds), "n_medication_requests": len(meds)})

    marker.write_text("ok", encoding="utf-8")
    return pd.DataFrame(rows, columns=["patient_id", "file_path", "n_conditions", "n_medication_requests"])


# =========================
# 4A) Représentation patient simple depuis le Bundle FHIR
# =========================


def load_bundle(path: str) -> dict:
    obj = safe_read_json(Path(path))
    if not obj:
        raise RuntimeError(f"Could not read: {path}")
    return obj


def patient_demographics(patient: dict) -> dict:
    name = first(patient.get("name", []), {}) or {}
    given = " ".join(name.get("given", []) or []).strip() or None
    family = (name.get("family") or "").strip() or None
    full_name = " ".join([x for x in [given, family] if x]).strip() or None
    return {
        "given": given,
        "family": family,
        "full_name": full_name,
        "birthDate": patient.get("birthDate"),
        "gender": patient.get("gender"),
    }


def extract_patient_record_prefilter(fhir_bundle: dict) -> dict:
    """Extrait les candidats depuis le bundle préfiltré. Pas encore de LLM."""
    patient = find_patient(fhir_bundle) or {}
    pid = patient.get("id")
    demo = patient_demographics(patient)

    # Candidats Condition (on garde l'info structurée pour aider le LLM à décider)
    cond_candidates: List[dict] = []
    for c in iter_resources(fhir_bundle, "Condition"):
        disp = cond_display(c)
        if not disp:
            continue
        system, code = cond_system_code(c)
        cond_candidates.append({"system": system, "code": code, "display": disp})

    # Déduplication par system|code en préservant l'ordre
    seen = set()
    cond_candidates_dedup = []
    for it in cond_candidates:
        key = f'{norm_text(it.get("system"))}|{norm_text(it.get("code"))}'
        if not it.get("system") or not it.get("code"):
            continue
        if key in seen:
            continue
        seen.add(key)
        cond_candidates_dedup.append(it)
    cond_candidates = cond_candidates_dedup[: settings.PREFILTER_MAX_CONDITION_CANDIDATES]

    # Candidats MedicationRequest (gardés tels quels, déjà assez propres)
    med_candidates: List[str] = []
    for m in iter_resources(fhir_bundle, "MedicationRequest"):
        d = med_display(m)
        if d:
            med_candidates.append(d)
    med_candidates = list(dict.fromkeys(med_candidates))[: settings.PREFILTER_MAX_MEDREQ_CANDIDATES]

    return {
        "patient_id": pid,
        **demo,
        "conditions_candidates": cond_candidates,
        "conditions": [x["display"] for x in cond_candidates],  # liste d'affichage (filtrée par LLM ensuite)
        "medication_requests": med_candidates,
    }


# =========================
# 4B) Choisir l'échantillon et construire la source préfiltrée
# =========================


def select_sample(df_index: pd.DataFrame, n_patients: Optional[int] = None, seed: Optional[int] = None):
    """Tire l'échantillon (déterministe via RANDOM_SEED) et extrait les candidats.

    Retourne (df_sample, patients_prefilter, df_gold_prefilter).
    """
    n_patients = n_patients if n_patients is not None else settings.N_PATIENTS
    seed = seed if seed is not None else settings.RANDOM_SEED

    df_sample = df_index.sample(n=min(n_patients, len(df_index)), random_state=seed).reset_index(drop=True)
    bundles = [load_bundle(p) for p in df_sample["file_path"].tolist()]
    patients_prefilter = [extract_patient_record_prefilter(b) for b in bundles]

    preview_n = settings.PREVIEW_N
    df_gold_prefilter = pd.DataFrame([{
        "patient_id": p["patient_id"],
        "full_name": p.get("full_name"),
        "birthDate": p.get("birthDate"),
        "gender": p.get("gender"),
        "n_conditions_candidates": len(p["conditions"]),
        "conditions_candidates_preview": "; ".join(p["conditions"][:preview_n]),
        "n_medication_requests": len(p["medication_requests"]),
        "medreq_preview": "; ".join(p["medication_requests"][:preview_n]),
    } for p in patients_prefilter])
    if not df_gold_prefilter.empty:
        df_gold_prefilter = df_gold_prefilter.sort_values("patient_id")

    return df_sample, patients_prefilter, df_gold_prefilter
