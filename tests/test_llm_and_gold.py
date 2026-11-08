from src import gold, llm


def test_extract_first_json_with_fences_and_text():
    txt = 'Here you go:\n```json\n{"a": {"b": 1}, "c": [1, 2]}\n```\nthanks'
    assert llm.extract_first_json(txt) == '{"a": {"b": 1}, "c": [1, 2]}'
    assert llm.extract_first_json("no json here") is None


def test_repair_and_lenient_load():
    assert llm.load_json_lenient('{"a": [1, 2,], }') == {"a": [1, 2]}
    assert llm.load_json_lenient("{'a': 'b'}") == {"a": "b"}
    assert llm.load_json_lenient("garbage") is None


def test_llm_filter_keeps_selected_indexes_and_caches(tmp_settings, fake_llm):
    cands = [{"system": "s", "code": "1", "display": "Hypertension"},
             {"system": "s", "code": "2", "display": "Housing problem"},
             {"system": "s", "code": "3", "display": "Asthma"}]
    fake = fake_llm(['{"keep_indexes": [0, 2, 99], "drop_indexes": [1]}'])
    assert gold.llm_keep_condition_indexes("p1", cands, llm=fake) == [0, 2]
    # Deuxième appel : vient du cache, le LLM n'est pas rappelé
    assert gold.llm_keep_condition_indexes("p1", cands, llm=fake) == [0, 2]
    assert len(fake.prompts) == 1


def test_llm_filter_keeps_everything_on_bad_output(tmp_settings, fake_llm):
    cands = [{"system": "s", "code": "1", "display": "A"}, {"system": "s", "code": "2", "display": "B"}]
    fake = fake_llm(["not json", "still not json"])
    assert gold.llm_keep_condition_indexes("p2", cands, llm=fake) == [0, 1]


def test_apply_filter_and_save_gold(tmp_settings, fake_llm):
    patients = [{"patient_id": "p1", "conditions_candidates": [
        {"system": "s", "code": "1", "display": "Hypertension"},
        {"system": "s", "code": "2", "display": "Housing problem"}],
        "medication_requests": ["X"]}]
    out = gold.build_gold(patients, llm=fake_llm(['{"keep_indexes": [0]}']))
    assert out[0]["conditions"] == ["Hypertension"]
    assert gold.load_gold_patients()[0]["conditions"] == ["Hypertension"]
