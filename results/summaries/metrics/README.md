# Scores

| Fichier | Contenu |
|---|---|
| `df_scores_overall_summary.csv` | une ligne par métrique : moyenne, médiane, écart-type, min et max sur les 40 patients |
| `df_scores_summary_by_note_style.csv` | les mêmes statistiques pour chaque style de note (10 patients par style) |
| `df_scores.csv` | une ligne par patient, avec toutes les métriques, le classement de l'erreur et le verdict de vérification |

Les noms de colonnes suivent le schéma `score_{note|source}_vs_recon_{conditions|medication_requests}_{exact|semantic}_{tp|fp|fn|precision|recall|f1}[_llm]`, détaillé dans le [guide des résultats](../../../docs/results_guide.md#noms-des-métriques).
