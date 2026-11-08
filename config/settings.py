"""Paramètres centralisés du pipeline BUPA.

Toutes les valeurs reprennent celles du notebook original
(`BUPA_benchmark_final.ipynb`, cellule « 1) Imports, chemins, configuration »).
Les chemins sont désormais ancrés à la racine du dépôt :

    data/raw/            -> zip Synthea + bundles FHIR extraits
    data/processed/      -> bundles filtrés, notes, reconstructions, scores, caches
    data/gold_standard/  -> gold source (un JSON par patient)

Les modules de `src/` lisent ces valeurs au moment de l'appel
(`settings.X`), on peut donc les modifier depuis un notebook avant de lancer
une étape, par exemple ``settings.N_PATIENTS = 40``.
"""

from __future__ import annotations

from pathlib import Path

# =========================
# Chemins
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold_standard"

# Données Synthea brutes
ZIP_DIR = RAW_DATA_DIR / "zip"
RAW_DIR = RAW_DATA_DIR / "extracted_raw"

# Sorties intermédiaires du pipeline
FILTERED_DIR = PROCESSED_DIR / "filtered_conditions_medreq"  # 1 Bundle JSON par patient (Patient + Condition + MedicationRequest)
NOTES_DIR = PROCESSED_DIR / "notes"
RAW_NOTES_DIR = NOTES_DIR / "raw"
RECON_DIR = PROCESSED_DIR / "reconstruction"
SCORES_DIR = PROCESSED_DIR / "scores"
LLM_ALIGN_DIR = SCORES_DIR / "llm_align"
NOTE_ALIGN_DIR = SCORES_DIR / "llm_note_vs_recon"
FILTER_CACHE_DIR = PROCESSED_DIR / "filter_cache"  # cache du filtre hybride (niveau patient)
VERIFICATION_DIR = PROCESSED_DIR / "verification"
EXPORT_DIR = PROCESSED_DIR / "exports"
EXPORT_ZIP_PATH = PROCESSED_DIR / "exports_datasets_full.zip"

# Livrables versionnés dans Git
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "assets" / "figures"

ALL_WORK_DIRS = [
    ZIP_DIR, RAW_DIR, FILTERED_DIR, GOLD_DIR, NOTES_DIR, RAW_NOTES_DIR,
    RECON_DIR, SCORES_DIR, LLM_ALIGN_DIR, NOTE_ALIGN_DIR, FILTER_CACHE_DIR,
    VERIFICATION_DIR, EXPORT_DIR,
]


def ensure_dirs() -> None:
    """Crée tous les dossiers de travail (appelé au début des notebooks)."""
    for d in ALL_WORK_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# =========================
# Source des données
# =========================
SYNTHEA_ZIP_URLS = [
    "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_nov2021.zip",
    "https://github.com/synthetichealth/synthea-sample-data/raw/refs/heads/main/downloads/synthea_sample_data_fhir_r4_nov2021.zip",
]
ZIP_PATH = ZIP_DIR / "synthea_fhir_r4.zip"

# =========================
# Modèle et échantillon
# =========================
MODEL_NAME = "google/medgemma-4b-it"
N_PATIENTS = 40          
RANDOM_SEED = 155

# =========================
# Filtrage hybride
# =========================
# Étape A : préfiltre déterministe (conservateur), rapide et reproductible.
PREFILTER_ENABLED = True
PREFILTER_DROP_SOCIAL_HISTORY = True

# On ne garde que les conditions codées dans ces systèmes (stable, retire le bruit)
ALLOWED_CONDITION_SYSTEMS = {
    "http://snomed.info/sct",
    "http://hl7.org/fhir/sid/icd-10",
    "http://hl7.org/fhir/sid/icd-10-cm",
}

# Sous-chaînes « SDOH / administratif » très évidentes pour le préfiltre.
# Liste volontairement courte pour ne pas jeter de vraies conditions médicales.
PREFILTER_DROP_SUBSTRINGS = [
    "criminal record",
    "victim of",
    "intimate partner",
    "domestic violence",
    "violence in the environment",
    "reports of violence",
    "housing unsatisfactory",
    "not in labor force",
    "employment",
    "education",
    "social",
    "stress",
    "risk activity",
    "primary school",
]

# Plafonds de candidats AVANT le filtre LLM (plus hauts que les plafonds finaux)
PREFILTER_MAX_CONDITION_CANDIDATES = 80
PREFILTER_MAX_MEDREQ_CANDIDATES = 40

# Étape B : filtre LLM au niveau patient, sur l'échantillon seulement (haute précision).
USE_LLM_PATIENT_FILTER = True
LLM_FILTER_MAX_NEW_TOKENS = 512
LLM_FILTER_TEMPERATURE = 0.0
LLM_FILTER_DO_SAMPLE = False
LLM_FILTER_RETRIES = 1

# Plafonds finaux utilisés pour la note, la reconstruction et le scoring
MAX_CONDITIONS_PER_PATIENT = 12
MAX_MEDICATION_REQUESTS_PER_PATIENT = 12

PREVIEW_N = 6

# =========================
# Budgets de génération
# =========================
NOTE_GENERATION_MAX_NEW_TOKENS = 320
RECON_EXTRACTION_MAX_NEW_TOKENS = 220
RECON_FIX_JSON_MAX_NEW_TOKENS = 256
SEMANTIC_MATCH_MAX_NEW_TOKENS = 96
NOTE_CONSISTENCY_MAX_NEW_TOKENS = 256
VERIFICATION_MAX_NEW_TOKENS = 256

# Températures / échantillonnage
NOTE_GENERATION_TEMPERATURE = 0.8
NOTE_GENERATION_DO_SAMPLE = True
RECON_TEMPERATURE = 0.0
RECON_DO_SAMPLE = False
SCORING_TEMPERATURE = 0.0
SCORING_DO_SAMPLE = False
VERIFICATION_TEMPERATURE = 0.0
VERIFICATION_DO_SAMPLE = False
