from src import export
from src.utils import write_json


def test_rebuild_export_and_publish(tmp_settings):
    write_json(tmp_settings.GOLD_DIR / "p1.json", {
        "patient_id": "p1", "note_style": "telegraphic_note", "note_style_label": "Telegraphic note",
        "full_name": "John Doe", "conditions": ["Asthma"], "medication_requests": ["X"]})
    tmp_settings.NOTES_DIR.mkdir(parents=True)
    (tmp_settings.NOTES_DIR / "p1.txt").write_text("John Doe has asthma, takes X.", encoding="utf-8")
    write_json(tmp_settings.RECON_DIR / "p1.json", {"patient_id": "p1", "conditions": ["asthma"], "medication_requests": []})
    write_json(tmp_settings.RECON_DIR / "p1.raw.json", {"patient_name": "John Doe", "conditions": ["asthma"]})

    # Pas de JSON de scores -> repli sur les métriques exactes (bug corrigé du notebook)
    exports = export.rebuild_datasets_from_cache()
    s = exports["df_scores"].iloc[0]
    assert s["score_source_vs_recon_conditions_exact_f1"] == 1.0
    assert s["score_source_vs_recon_medication_requests_exact_f1"] == 0.0
    assert len(exports["df_gold_conditions_long"]) == 1

    review = export.export_datasets(exports)
    assert "df_scores" in set(review["dataset_name"])
    assert tmp_settings.EXPORT_ZIP_PATH.exists()

    written = export.publish_results()
    assert (tmp_settings.RESULTS_DIR / "summaries" / "metrics" / "df_scores.csv") in written

    removed = export.reset_benchmark_outputs()
    assert any(r.endswith("df_scores.csv") for r in removed)
