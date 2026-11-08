"""Fixtures partagées. Aucun test n'a besoin de GPU ni de MedGemma : on utilise un faux LLM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402


class FakeLLM:
    """Remplace MedGemma : renvoie des réponses prédéfinies, dans l'ordre, et garde les prompts reçus."""

    model_name = "fake-llm"

    def __init__(self, responses=None, available=True):
        self.responses = list(responses or [])
        self._available = available
        self.prompts = []

    def available(self) -> bool:
        return self._available

    def generate(self, prompt, max_new_tokens, temperature, do_sample):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "{}"


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Redirige tous les dossiers de travail vers un dossier temporaire."""
    mapping = {
        "RAW_DIR": tmp_path / "raw",
        "FILTERED_DIR": tmp_path / "processed" / "filtered",
        "GOLD_DIR": tmp_path / "gold",
        "NOTES_DIR": tmp_path / "processed" / "notes",
        "RAW_NOTES_DIR": tmp_path / "processed" / "notes" / "raw",
        "RECON_DIR": tmp_path / "processed" / "reconstruction",
        "SCORES_DIR": tmp_path / "processed" / "scores",
        "LLM_ALIGN_DIR": tmp_path / "processed" / "scores" / "llm_align",
        "NOTE_ALIGN_DIR": tmp_path / "processed" / "scores" / "llm_note_vs_recon",
        "FILTER_CACHE_DIR": tmp_path / "processed" / "filter_cache",
        "VERIFICATION_DIR": tmp_path / "processed" / "verification",
        "EXPORT_DIR": tmp_path / "processed" / "exports",
        "EXPORT_ZIP_PATH": tmp_path / "processed" / "exports.zip",
        "RESULTS_DIR": tmp_path / "results",
        "FIGURES_DIR": tmp_path / "figures",
    }
    for name, path in mapping.items():
        monkeypatch.setattr(settings, name, path)
    return settings


def make_bundle(pid="p1", conditions=None, meds=None, name=("John",), family="Doe",
                birth="1980-01-02", gender="male"):
    """Construit un petit Bundle FHIR synthétique pour les tests."""
    entries = [{"resource": {
        "resourceType": "Patient", "id": pid, "birthDate": birth, "gender": gender,
        "name": [{"given": list(name), "family": family}],
    }}]
    for c in conditions or []:
        res = {"resourceType": "Condition",
               "code": {"coding": [{"system": c.get("system", "http://snomed.info/sct"),
                                    "code": c["code"], "display": c["display"]}]}}
        if c.get("category"):
            res["category"] = [{"coding": [{"code": c["category"]}]}]
        entries.append({"resource": res})
    for m in meds or []:
        entries.append({"resource": {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                                      "code": m["code"], "display": m["display"]}]},
        }})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


@pytest.fixture
def bundle_factory():
    return make_bundle
