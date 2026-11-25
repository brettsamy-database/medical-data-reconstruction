"""Données Synthea : téléchargement, préfiltrage des bundles FHIR et tirage de l'échantillon.

Cette étape est entièrement déterministe, sans LLM. Le préfiltre est volontairement
prudent : il ne retire que le bruit évident (déterminants sociaux, éléments non
codés). Le tri fin est fait plus tard par le LLM, sur l'échantillon seulement
(voir `gold.py`).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm

from config import settings
from src.utils import first, norm_text, safe_read_json

INDEX_COLUMNS = ["patient_id", "file_path", "n_conditions", "n_medication_requests"]


# =========================
# Téléchargement
# =========================


def download_file(url: str, dest: Path, timeout_s: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def ensure_zip(zip_path: Optional[Path] = None, urls: Optional[List[str]] = None) -> Path:
    """Télécharge l'archive Synthea si elle n'est pas déjà là, en essayant chaque miroir."""
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
        except requests.RequestException as e:
            last_err = e
            zip_path.unlink(missing_ok=True)  # pas de zip à moitié téléchargé
    raise RuntimeError(f"Could not download the Synthea zip. Last error: {last_err}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    # Le marqueur n'est écrit qu'en fin d'extraction : une extraction interrompue est refaite.
    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        print("Already extracted:", extract_dir.resolve())
        return
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    print("Extracted to:", extract_dir.resolve())


def download_synthea() -> Path:
    zip_path = ensure_zip(settings.ZIP_PATH)
    extract_zip(zip_path, settings.RAW_DIR)
    return settings.RAW_DIR


def reset_filtered_dataset() -> None:
    """À appeler après avoir modifié les règles PREFILTER_* pour forcer un nouveau filtrage."""
    shutil.rmtree(settings.FILTERED_DIR, ignore_errors=True)


# =========================
# Lecture des ressources FHIR
# =========================


def iter_resources(bundle: dict, resource_type: str) -> List[dict]:
    resources = (e.get("resource", {}) for e in bundle.get("entry") or [])
    return [r for r in resources if r.get("resourceType") == resource_type]


def find_patient(bundle: dict) -> Optional[dict]:
    return first(iter_resources(bundle, "Patient"))


def _first_coding(concept: Optional[dict]) -> Tuple[dict, dict]:
    """Synthea ne met qu'un seul coding par concept : on lit le premier."""
    concept = concept or {}
    return concept, first(concept.get("coding") or [], {}) or {}


def cond_display(cond: dict) -> Optional[str]:
    concept, coding = _first_coding(cond.get("code"))
    return coding.get("display") or concept.get("text")


def cond_system_code(cond: dict) -> Tuple[str, str]:
    _, coding = _first_coding(cond.get("code"))
    return coding.get("system") or "", coding.get("code") or ""


def med_display(medreq: dict) -> Optional[str]:
    concept, coding = _first_coding(medreq.get("medicationCodeableConcept"))
    return coding.get("display") or concept.get("text")


def med_system_code(medreq: dict) -> Tuple[str, str]:
    _, coding = _first_coding(medreq.get("medicationCodeableConcept"))
    return coding.get("system") or "", coding.get("code") or ""


def _code_key(system: str, code: str) -> Optional[str]:
    return f"{norm_text(system)}|{norm_text(code)}" if system and code else None


def _dedupe(items: list, key: Callable[[object], Optional[str]], max_items: Optional[int] = None) -> list:
    """Garde le premier élément de chaque clé, dans l'ordre. Les éléments sans clé sont écartés."""
    out, seen = [], set()
    for it in items:
        k = key(it)
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append(it)
        if max_items is not None and len(out) >= max_items:
            break
    return out


# =========================
# Préfiltre déterministe
# =========================


def is_social_history(cond: dict) -> bool:
    """Synthea range les déterminants sociaux (SDOH) dans Condition ; leur catégorie est le signal le plus fiable."""
    for cat in cond.get("category") or []:
        for coding in cat.get("coding") or []:
            code = norm_text(coding.get("code"))
            disp = norm_text(coding.get("display"))
            if code == "social-history":
                return True
            if any(tag in field for tag in ("social", "sdoh") for field in (code, disp)):
                return True
    return False


def is_noisy_display(disp: Optional[str]) -> bool:
    if not settings.PREFILTER_ENABLED:
        return False
    d = norm_text(disp)
    if not d:
        return False
    return any(norm_text(sub) and norm_text(sub) in d for sub in settings.PREFILTER_DROP_SUBSTRINGS or [])


def is_medical_condition(cond: dict) -> bool:
    """Une condition est gardée si elle n'est pas un SDOH évident et qu'elle est codée dans un système reconnu."""
    if settings.PREFILTER_DROP_SOCIAL_HISTORY and is_social_history(cond):
        return False
    if is_noisy_display(cond_display(cond)):
        return False
    system, code = cond_system_code(cond)
    return bool(system and code) and system in settings.ALLOWED_CONDITION_SYSTEMS


def dedupe_limit_conditions(conds: List[dict], max_items: int) -> List[dict]:
    """Une condition est souvent répétée à chaque consultation : on garde la première par code."""
    return _dedupe(conds, lambda c: _code_key(*cond_system_code(c)), max_items)


def dedupe_limit_medreqs(meds: List[dict], max_items: int) -> List[dict]:
    """Même principe ; sans code, on se rabat sur le libellé du médicament."""
    def key(m):
        return _code_key(*med_system_code(m)) or norm_text(med_display(m)) or None

    return _dedupe(meds, key, max_items)


# =========================
# Dataset filtré (un bundle allégé par patient, sur disque)
# =========================


def index_filtered_dir(out_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        obj = safe_read_json(p) or {}
        rows.append({
            "patient_id": p.stem,
            "file_path": str(p),
            "n_conditions": len(iter_resources(obj, "Condition")),
            "n_medication_requests": len(iter_resources(obj, "MedicationRequest")),
        })
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def build_filtered_dataset(raw_dir: Optional[Path] = None, out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Parcourt tous les bundles une seule fois et écrit, par patient, un bundle avec Patient, Conditions et MedicationRequests."""
    raw_dir = Path(raw_dir or settings.RAW_DIR)
    out_dir = Path(out_dir or settings.FILTERED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".filtered_ok"
    if marker.exists():
        print("Filtered dataset already present:", out_dir.resolve())
        print("Note: if you changed PREFILTER_* settings, call reset_filtered_dataset() to rebuild.")
        return index_filtered_dir(out_dir)

    rows = []
    for p in tqdm(sorted(raw_dir.rglob("*.json")), desc="Filtering Bundles (one pass)"):
        obj = safe_read_json(p)
        if not obj or obj.get("resourceType") != "Bundle":
            continue
        patient = find_patient(obj)
        pid = (patient or {}).get("id")
        if not pid:
            continue

        conds = [c for c in iter_resources(obj, "Condition") if is_medical_condition(c)]
        conds = dedupe_limit_conditions(conds, settings.PREFILTER_MAX_CONDITION_CANDIDATES)
        meds = dedupe_limit_medreqs(iter_resources(obj, "MedicationRequest"), settings.PREFILTER_MAX_MEDREQ_CANDIDATES)

        filtered = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": r} for r in [patient, *conds, *meds]],
        }
        out_path = out_dir / f"{pid}.json"
        out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"patient_id": pid, "file_path": str(out_path),
                     "n_conditions": len(conds), "n_medication_requests": len(meds)})

    marker.write_text("ok", encoding="utf-8")
    # Même ordre que index_filtered_dir : sinon le tirage avec RANDOM_SEED donnerait
    # d'autres patients au premier lancement qu'aux suivants.
    return pd.DataFrame(rows, columns=INDEX_COLUMNS).sort_values("patient_id").reset_index(drop=True)


# =========================
# Échantillon et dossier patient
# =========================


def load_bundle(path: str) -> dict:
    obj = safe_read_json(Path(path))
    if not obj:
        raise RuntimeError(f"Could not read: {path}")
    return obj


def patient_demographics(patient: dict) -> dict:
    name = first(patient.get("name") or [], {}) or {}
    given = " ".join(name.get("given") or []).strip() or None
    family = (name.get("family") or "").strip() or None
    return {
        "given": given,
        "family": family,
        "full_name": " ".join(x for x in (given, family) if x) or None,
        "birthDate": patient.get("birthDate"),
        "gender": patient.get("gender"),
    }


def extract_patient_record_prefilter(fhir_bundle: dict) -> dict:
    """Construit le dossier patient avant le filtre LLM.

    Les conditions gardent leur code (system, code) : le LLM s'en sert pour
    distinguer une vraie maladie d'un élément administratif au libellé ambigu.
    """
    patient = find_patient(fhir_bundle) or {}

    cond_candidates = []
    for c in iter_resources(fhir_bundle, "Condition"):
        disp = cond_display(c)
        if disp:
            system, code = cond_system_code(c)
            cond_candidates.append({"system": system, "code": code, "display": disp})
    cond_candidates = _dedupe(cond_candidates, lambda it: _code_key(it["system"], it["code"]),
                              settings.PREFILTER_MAX_CONDITION_CANDIDATES)

    med_names = [d for d in (med_display(m) for m in iter_resources(fhir_bundle, "MedicationRequest")) if d]
    med_candidates = list(dict.fromkeys(med_names))[: settings.PREFILTER_MAX_MEDREQ_CANDIDATES]

    return {
        "patient_id": patient.get("id"),
        **patient_demographics(patient),
        "conditions_candidates": cond_candidates,
        "conditions": [x["display"] for x in cond_candidates],  # remplacée par la liste filtrée dans gold.py
        "medication_requests": med_candidates,
    }


def select_sample(df_index: pd.DataFrame, n_patients: Optional[int] = None, seed: Optional[int] = None):
    """Tire l'échantillon (reproductible grâce à RANDOM_SEED) et construit le dossier de chaque patient.

    Renvoie (df_sample, patients_prefilter, df_gold_prefilter).
    """
    n_patients = settings.N_PATIENTS if n_patients is None else n_patients
    seed = settings.RANDOM_SEED if seed is None else seed

    df_sample = df_index.sample(n=min(n_patients, len(df_index)), random_state=seed).reset_index(drop=True)
    patients_prefilter = [extract_patient_record_prefilter(load_bundle(p)) for p in df_sample["file_path"]]

    n = settings.PREVIEW_N
    df_gold_prefilter = pd.DataFrame([{
        "patient_id": p["patient_id"],
        "full_name": p.get("full_name"),
        "birthDate": p.get("birthDate"),
        "gender": p.get("gender"),
        "n_conditions_candidates": len(p["conditions"]),
        "conditions_candidates_preview": "; ".join(p["conditions"][:n]),
        "n_medication_requests": len(p["medication_requests"]),
        "medreq_preview": "; ".join(p["medication_requests"][:n]),
    } for p in patients_prefilter])
    if not df_gold_prefilter.empty:
        df_gold_prefilter = df_gold_prefilter.sort_values("patient_id")

    return df_sample, patients_prefilter, df_gold_prefilter
