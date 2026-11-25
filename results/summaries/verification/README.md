# Vérification

`df_verification.csv` contient, pour chaque patient, le verdict du modèle (`consistent`, `minor_issues`, `major_issues` ou `unknown`), sa confiance, le nombre d'éléments qu'il juge non justifiés ou manquants, et un résumé en une phrase.

À lire avec recul : c'est le même modèle qui a produit la reconstruction, et il se montre indulgent (37 `consistent` sur 40, y compris des cas où il liste lui-même des problèmes). Voir le [rapport du benchmark](../../../docs/benchmark_report.md#vérification).
