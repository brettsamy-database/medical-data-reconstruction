# Guide des notebooks

## Organisation

Dans le dépôt original, tout le pipeline tenait dans un seul notebook d'environ 3 400 lignes.
Le code est désormais dans `src/` et les notebooks ne font plus qu'enchaîner les étapes et afficher les résultats.

| Notebook | Étapes | Modules utilisés | GPU / MedGemma |
|---|---|---|---|
| `01_fhir_exploration.ipynb` | téléchargement Synthea, préfiltrage, index, échantillon | `src/fhir.py` | non |
| `02_note_generation.ipynb` | gold source (filtre LLM), styles, génération des notes, reconstruction | `src/gold.py`, `src/notes.py`, `src/reconstruction.py` | oui |
| `03_benchmark_results.ipynb` | scoring, vérification, analyse des erreurs, synthèses, exports, figures | `src/evaluation.py`, `src/verification.py`, `src/export.py`, `src/plots.py` | oui (scores sémantiques et vérification) |

## Correspondance avec l'ancien notebook

| Section de `BUPA_benchmark_final.ipynb` | Nouvel emplacement |
|---|---|
| 0) Installation, 0.1) Authentification HF | `requirements.txt`, `src/llm.py` (`hf_login`) |
| 1) Imports, chemins, configuration | `config/settings.py` |
| 2) Télécharger et extraire Synthea | `src/fhir.py` (`download_synthea`) |
| 3A-3C) Préfiltrage, dataset filtré, index | `src/fhir.py` (`build_filtered_dataset`) |
| 4A-4B) Extraction patient, échantillon | `src/fhir.py` (`select_sample`) |
| 4C.1) Chargement de MedGemma | `src/llm.py` (`MedGemma`, `get_llm`) |
| 4D) Filtre LLM du gold source | `src/gold.py` (`build_gold`) |
| 5) Génération des notes | `src/notes.py` (`generate_notes`) |
| 6) Reconstruction | `src/reconstruction.py` (`reconstruct_all`) |
| 7) Scoring | `src/evaluation.py` (`score_all`) |
| 8) Vérification | `src/verification.py` (`run_verification`) |
| 9) Analyse des erreurs | `src/evaluation.py` (`run_error_analysis`) |
| 9.1) Synthèses | `src/evaluation.py` (`run_summaries`) |
| 10.1) Reset des sorties | `src/export.py` (`reset_benchmark_outputs`) |
| 11) Export depuis le cache | `src/export.py` (`rebuild_datasets_from_cache`, `export_datasets`) |

## Règles d'exécution

- exécuter les notebooks dans l'ordre 01 → 02 → 03 ;
- garder la même valeur de `settings.N_PATIENTS` dans 01 et 02 (l'échantillon est tiré avec `RANDOM_SEED`) ;
- ne pas modifier les paramètres au milieu d'un run final ;
- avant un benchmark final, passer `RESET_BENCHMARK_OUTPUTS = True` dans le notebook 03 ;
- ne passer `PUBLISH = True` que pour un run qu'on veut publier : cela remplace `results/` et `assets/figures/`.

Les sorties de chaque étape sont écrites sur disque et les appels LLM sont en cache
(`data/processed/filter_cache`, `scores/llm_align`, `scores/llm_note_vs_recon`, `verification/`) :
relancer une étape ne rappelle le modèle que si ses entrées ont changé.

## Fichiers exportés

`data/processed/exports/` contient : `df_index`, `df_gold`, `df_notes`, `df_recon`, les tables longues
(`df_gold_conditions_long`, `df_gold_medreq_long`, `df_recon_conditions_long`, `df_recon_medreq_long`),
`df_scores`, `df_scores_overall_summary`, `df_scores_summary_by_note_style`, `df_error_analysis`,
`df_error_types_by_note_style_conditions`, `df_error_types_by_note_style_medication_requests`,
`df_hard_cases_by_note_style`, `df_verification`, `df_verification_by_note_style` et `df_export_review`,
plus une archive `exports_datasets_full.zip`.

## Voir aussi

- [Pipeline détaillée](pipeline_explained.md)
- [Guide des résultats](results_guide.md)
