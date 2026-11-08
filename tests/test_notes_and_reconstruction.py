from src import notes, reconstruction


PATIENT = {"patient_id": "p1", "full_name": "John Doe", "birthDate": "1980-01-02", "gender": "male",
           "conditions": ["Hypertension", "Asthma"], "medication_requests": ["Lisinopril 10 MG Oral Tablet"]}


def test_assign_note_styles_cycles():
    out = notes.assign_note_styles([{"patient_id": str(i)} for i in range(5)])
    assert [p["note_style"] for p in out] == notes.NOTE_STYLE_NAMES + notes.NOTE_STYLE_NAMES[:1]
    assert out[0]["note_style_label"] == "Short consultation note"


def test_note_prompt_contains_all_items():
    prompt = notes.note_prompt(dict(PATIENT, note_style="telegraphic_note"))
    for item in ["John Doe", "1980-01-02", "Hypertension", "Asthma", "Lisinopril 10 MG Oral Tablet"]:
        assert item in prompt
    assert "short fragments" in prompt


def test_clean_generated_note_removes_artifacts():
    raw = ("Clinical note: John Doe, born 1980-01-02, male, has hypertension. He takes lisinopril.\n"
           "Final Answer: The final answer is \\boxed{...}\n```python\nprint('x')\n```")
    clean = notes.clean_generated_note(raw)
    assert clean.startswith("John Doe")
    assert "Final Answer" not in clean and "```" not in clean


def test_clean_generated_note_dedupes_sentences():
    assert notes.clean_generated_note("He has asthma. He has asthma. He takes X daily.") == "He has asthma. He takes X daily."


def test_generate_notes_falls_back_to_template(tmp_settings, fake_llm):
    _, out = notes.generate_notes([PATIENT], llm=fake_llm(available=False))
    assert out["p1"].used_llm is False
    assert "John Doe" in out["p1"].note
    assert (tmp_settings.NOTES_DIR / "p1.txt").exists()
    _, reloaded = notes.load_notes()
    assert reloaded["p1"].note == out["p1"].note


def test_postprocess_list_cleans_dedupes_and_filters_noise():
    values = ["- Hypertension", "1. hypertension", "Employment", "Asthma ", "x", "Victim of abuse"]
    assert reconstruction.postprocess_list(values, 10, filter_noisy=True) == ["Hypertension", "Asthma"]
    # Sans filtre bruit, « Employment » est conservé
    assert "Employment" in reconstruction.postprocess_list(values, 10, filter_noisy=False)


def test_clip_to_gold():
    assert reconstruction.clip_to_gold(["Asthma", "Flu"], ["asthma"]) == ["Asthma"]


def test_parse_llm_json_without_retry():
    assert reconstruction.parse_llm_json('```json\n{"conditions": ["A",]}\n```', retries=0) == {"conditions": ["A"]}
    assert reconstruction.parse_llm_json("nothing", retries=0) is None


def test_reconstruct_all_writes_three_views(tmp_settings, fake_llm):
    from src.llm import LLMResult

    fake = fake_llm(['{"patient_name": "John Doe", "date_of_birth": "1980-01-02", "gender": "male", '
                     '"conditions": ["Hypertension", "Flu", "Employment"], "medication_requests": ["Lisinopril 10 MG Oral Tablet"]}'])
    raw, clean, clipped = reconstruction.reconstruct_all([PATIENT], {"p1": LLMResult("note", True, "fake")}, llm=fake)
    assert clean["p1"]["conditions"] == ["Hypertension", "Flu"]
    assert clipped["p1"]["conditions"] == ["Hypertension"]
    assert raw["p1"]["patient_name"] == "John Doe"
    for suffix in (".json", ".raw.json", ".clipped.json"):
        assert (tmp_settings.RECON_DIR / f"p1{suffix}").exists()
    r_raw, r_clean, _ = reconstruction.load_reconstructions()
    assert r_clean["p1"]["conditions"] == ["Hypertension", "Flu"]
