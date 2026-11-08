import json

import pandas as pd

from src import fhir


def test_patient_demographics():
    demo = fhir.patient_demographics({"name": [{"given": ["Ana", "Maria"], "family": "Lopez"}],
                                      "birthDate": "1990-05-06", "gender": "female"})
    assert demo["full_name"] == "Ana Maria Lopez"
    assert demo["birthDate"] == "1990-05-06"
    assert demo["gender"] == "female"


def test_is_medical_condition_rules():
    def cond(display, system="http://snomed.info/sct", code="1", category=None):
        c = {"code": {"coding": [{"system": system, "code": code, "display": display}]}}
        if category:
            c["category"] = [{"coding": [{"code": category}]}]
        return c

    assert fhir.is_medical_condition(cond("Hypertension"))
    assert not fhir.is_medical_condition(cond("Full-time employment (finding)"))   # sous-chaîne SDOH
    assert not fhir.is_medical_condition(cond("Asthma", category="social-history"))  # catégorie sociale
    assert not fhir.is_medical_condition(cond("Asthma", system="http://example.org"))  # système non autorisé
    assert not fhir.is_medical_condition(cond("Asthma", code=""))                    # non codée


def test_dedupe_limit_conditions_and_medreqs():
    conds = [{"code": {"coding": [{"system": "s", "code": c, "display": c}]}} for c in ["a", "b", "a", "c"]]
    assert [fhir.cond_display(c) for c in fhir.dedupe_limit_conditions(conds, 2)] == ["a", "b"]

    meds = [
        {"medicationCodeableConcept": {"coding": [{"system": "r", "code": "1", "display": "X"}]}},
        {"medicationCodeableConcept": {"coding": [{"system": "r", "code": "1", "display": "X bis"}]}},
        {"medicationCodeableConcept": {"text": "Free text drug"}},  # repli sur le libellé
    ]
    out = fhir.dedupe_limit_medreqs(meds, 10)
    assert [fhir.med_display(m) for m in out] == ["X", "Free text drug"]


def test_build_filtered_dataset_and_sample(tmp_settings, bundle_factory):
    tmp_settings.RAW_DIR.mkdir(parents=True)
    b1 = bundle_factory("p1", conditions=[{"code": "1", "display": "Hypertension"},
                                          {"code": "2", "display": "Stress (finding)"}],
                        meds=[{"code": "10", "display": "Lisinopril 10 MG Oral Tablet"}])
    b2 = bundle_factory("p2", conditions=[{"code": "3", "display": "Asthma"}])
    (tmp_settings.RAW_DIR / "a.json").write_text(json.dumps(b1))
    (tmp_settings.RAW_DIR / "b.json").write_text(json.dumps(b2))
    (tmp_settings.RAW_DIR / "not_a_bundle.json").write_text("{}")

    df_index = fhir.build_filtered_dataset()
    assert sorted(df_index["patient_id"]) == ["p1", "p2"]
    row = df_index.set_index("patient_id").loc["p1"]
    assert row["n_conditions"] == 1           # « Stress » retiré par le préfiltre
    assert row["n_medication_requests"] == 1

    # Second appel : relu depuis le cache (marqueur .filtered_ok)
    assert len(fhir.build_filtered_dataset()) == 2

    _, patients, df_prev = fhir.select_sample(df_index, n_patients=2, seed=1)
    assert isinstance(df_prev, pd.DataFrame) and len(patients) == 2
    p1 = next(p for p in patients if p["patient_id"] == "p1")
    assert p1["full_name"] == "John Doe"
    assert p1["conditions"] == ["Hypertension"]
    assert p1["medication_requests"] == ["Lisinopril 10 MG Oral Tablet"]
