import pandas as pd
import pytest

from src import evaluation as ev
from src import verification as vf
from src.llm import LLMResult


def test_metrics_from_counts_edge_cases():
    assert ev.metrics_from_counts(0, 0, 0)["f1"] == 1.0
    m = ev.metrics_from_counts(2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert ev.metrics_from_counts(0, 0, 3)["precision"] == 0.0


def test_metrics_from_sets():
    m = ev.metrics_from_sets({"a", "b"}, {"b", "c"})
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)


def test_demographic_exact():
    assert ev.name_exact("John  Doe", "john doe") == 1
    assert ev.name_exact("", "") == 0
    assert ev.dob_exact("1980-01-02", "1980-01-02") == 1
    assert ev.gender_exact("Male", "male") == 1


def test_exact_items_supported_by_note():
    assert ev.exact_items_supported_by_note("Pt has Hypertension and gout.", ["hypertension", "asthma"]) == ["hypertension"]


@pytest.mark.parametrize("kwargs, expected", [
    (dict(gold_n=0, pred_n=0, f1_exact=1, f1_semantic=1, note_f1=1), "no_target_and_no_prediction"),
    (dict(gold_n=3, pred_n=0, f1_exact=0, f1_semantic=0, note_f1=0), "complete_omission"),
    (dict(gold_n=3, pred_n=3, f1_exact=1.0, f1_semantic=1.0, note_f1=1.0), "perfect_reconstruction"),
    (dict(gold_n=3, pred_n=3, f1_exact=0.5, f1_semantic=1.0, note_f1=1.0), "exact_wording_gap_only"),
    (dict(gold_n=3, pred_n=3, f1_exact=0.5, f1_semantic=0.6, note_f1=0.95), "source_to_note_gap_or_generation_loss"),
])
def test_classify_error_type(kwargs, expected):
    base = dict(note_precision=1, note_recall=1, fp_semantic=0, fn_semantic=0, note_unsupported_n=0, note_missing_n=0)
    out = ev.classify_error_type(**base, **kwargs)
    assert out["overall"] == expected
    assert set(out) == {"overall", "stage", "severity", "explanation"}


def test_full_scoring_chain_with_fake_llm(tmp_settings, fake_llm):
    gold = {"patient_id": "p1", "note_style": "telegraphic_note", "note_style_label": "Telegraphic note",
            "full_name": "John Doe", "birthDate": "1980-01-02", "gender": "male",
            "conditions": ["Hypertension", "Asthma"], "medication_requests": ["Lisinopril 10 MG Oral Tablet"]}
    recon_clean = {"p1": {"conditions": ["Hypertension", "High blood sugar"],
                          "medication_requests": ["Lisinopril 10 MG Oral Tablet"]}}
    recon_raw = {"p1": {"patient_name": "John Doe", "date_of_birth": "1980-01-02", "gender": "male",
                        "conditions": ["Hypertension", "High blood sugar"], "medication_requests": ["Lisinopril 10 MG Oral Tablet"]}}
    note = "John Doe, 1980-01-02, male. Hypertension, asthma. Lisinopril 10 MG Oral Tablet."
    notes = {"p1": LLMResult(note, True, "fake")}

    # Ordre des appels LLM : note_vs_recon conditions, source_vs_recon conditions (1 item non exact),
    # note_vs_recon médicaments (pas d'appel source : match exact)
    fake = fake_llm([
        '{"supported_indexes": [0], "missing_items": ["Asthma"]}',
        '{"match_index": -1, "confidence": 0.9}',
        '{"supported_indexes": [0], "missing_items": []}',
    ])
    df_scores = ev.score_all([gold], recon_clean, recon_raw, notes, llm=fake)
    row = df_scores.iloc[0]
    assert row["score_source_vs_recon_conditions_exact_tp"] == 1
    assert row["score_source_vs_recon_conditions_exact_f1"] == pytest.approx(0.5)
    assert row["score_note_vs_recon_conditions_semantic_fn_llm"] == 1
    assert row["score_source_vs_recon_medication_requests_exact_f1"] == 1.0
    assert row["name_exact_raw"] == 1

    fake_v = fake_llm(['{"verdict": "minor_issues", "confidence": 0.8, "unsupported_conditions": ["High blood sugar"], '
                       '"missing_conditions_from_reconstruction": ["Asthma"], "unsupported_medication_requests": [], '
                       '"missing_medication_requests_from_reconstruction": [], "summary": "ok"}'])
    df_verif, df_verif_style = vf.run_verification([gold], recon_clean, notes, llm=fake_v)
    assert df_verif.iloc[0]["verification_verdict"] == "minor_issues"
    assert df_verif.iloc[0]["verification_issue_total_n"] == 2
    assert not df_verif_style.empty

    df_scores = vf.merge_verification_into_scores(df_scores, df_verif)
    tables = ev.run_error_analysis(df_scores)
    assert "difficulty_score" in tables["df_scores"].columns
    assert len(tables["df_error_analysis"]) == 1

    overall, by_style = ev.run_summaries(tables["df_scores"], df_verif)
    assert set(overall["metric_family"]) == {"demographics", "note_vs_recon", "source_vs_recon", "verification"}
    rate = overall.set_index("metric_name").loc["verification_minor_issues_rate", "mean_value"]
    assert rate == 1.0
    assert (by_style["note_style"] == "telegraphic_note").all()
    assert isinstance(by_style, pd.DataFrame)
