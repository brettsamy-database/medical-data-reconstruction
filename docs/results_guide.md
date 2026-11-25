# Lire les résultats

Tous les CSV de `results/` sont régénérés par le notebook 03 (dernière section). Les chiffres commentés sont dans le [rapport du benchmark](benchmark_report.md).

## Par où commencer

1. [`df_scores_overall_summary.csv`](../results/summaries/metrics/df_scores_overall_summary.csv) : une ligne par métrique, moyenne sur les 40 patients
2. [`df_scores_summary_by_note_style.csv`](../results/summaries/metrics/df_scores_summary_by_note_style.csv) : les mêmes métriques, par style de note
3. [`df_error_analysis.csv`](../results/summaries/errors/df_error_analysis.csv) : où chaque patient a perdu de l'information
4. [`examples/`](../results/examples/) : trois patients suivis de bout en bout

## Noms des métriques

Les colonnes de score suivent toutes le même schéma :

```text
score_{référence}_vs_recon_{type}_{mode}_{mesure}[_llm]
```

| Partie | Valeurs |
|---|---|
| référence | `note` (ce qui est écrit dans la note) ou `source` (le gold) |
| type | `conditions` ou `medication_requests` |
| mode | `exact` (même libellé normalisé) ou `semantic` (jugé par le LLM, d'où le suffixe `_llm`) |
| mesure | `tp`, `fp`, `fn`, `precision`, `recall`, `f1` |

Par exemple, `score_source_vs_recon_conditions_semantic_f1_llm` est le F1 sémantique des conditions, entre le gold et la reconstruction.

Un cas mérite une précision. En **note vs recon exact**, la référence n'est pas toute la note, mais les éléments du gold recopiés mot pour mot dans la note. On ne peut pas faire mieux sans LLM : la note est du texte libre.

## Autres colonnes de `df_scores.csv`

| Colonnes | Contenu |
|---|---|
| `name_exact_raw`, `dob_exact_raw`, `gender_exact_raw` | 1 si la donnée extraite de la note est identique au gold |
| `gold_cond_n`, `pred_cond_n`, `raw_cond_n` (et `_med_`) | nombre d'éléments dans le gold, dans la reconstruction nettoyée, dans le JSON brut |
| `error_type_{type}_{overall,stage,severity,explanation}` | classement de l'erreur (voir ci-dessous) |
| `verification_*` | verdict et nombre de problèmes relevés par la vérification |
| `difficulty_score` | moyenne des quatre F1 sémantiques ; sert à trier les cas difficiles |

## Analyse des erreurs

Pour chaque patient et chaque type, `classify_error_type` ([src/evaluation.py](../src/evaluation.py)) compare les scores entre eux pour situer la perte :

| `stage` | Signification | Signal |
|---|---|---|
| `none` | rien à signaler | tous les F1 à 1 |
| `canonicalization_exact_matching` | reformulation sans perte de sens | exact < 1 mais sémantique à 1 |
| `source_to_note` | l'information manquait déjà dans la note | note vs recon ≥ 0.9, source vs recon < 0.9 |
| `note_to_reconstruction` | la note était complète, l'extraction a raté ou inventé | l'inverse |
| `mixed` | erreurs à plusieurs étapes | aucun des cas précédents |

`severity` dépend du plus faible des deux F1 sémantiques : `mild` au-dessus de 0.85, `moderate` au-dessus de 0.55, `severe` en dessous.

## Les autres fichiers

| Dossier | Fichiers | Usage |
|---|---|---|
| `intermediates/` | `df_gold`, `df_notes`, `df_recon` | le gold, la note (brute et nettoyée) et la reconstruction de chaque patient, côte à côte |
| `other_results/` | tables `*_long` | une ligne par condition ou prescription, pratique pour filtrer dans un tableur |
| `other_results/` | `df_index` | les 555 patients après préfiltrage, avec leur nombre de candidats |
| `other_results/` | `df_export_review` | liste des tables exportées, avec leur nombre de lignes et de colonnes |
| `summaries/errors/` | `df_error_types_by_note_style_*`, `df_hard_cases_by_note_style` | types d'erreurs comptés par style, et les 3 patients les plus difficiles de chaque style |
| `summaries/verification/` | `df_verification` | verdict, confiance et problèmes relevés pour chaque patient |
