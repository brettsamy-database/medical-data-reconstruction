# Metrics

Ce dossier contient les métriques principales du benchmark.

## Contenu

- `df_scores.csv` : scores détaillés par patient
- `df_scores_overall_summary.csv` : synthèse globale des scores du benchmark
- `df_scores_summary_by_note_style.csv` : synthèse des scores par style de note

## Rôle des fichiers

### `df_scores.csv`
Table détaillée patient par patient.

Elle permet d'analyser :
- les scores officiels,
- les scores secondaires,
- les métriques exactes,
- les métriques sémantiques,
- certains champs démographiques.

### `df_scores_overall_summary.csv`
Vue synthétique globale du benchmark.

Ce fichier permet de répondre rapidement aux questions suivantes :
- Quelle est la performance moyenne du pipeline ?
- Les conditions sont-elles plus difficiles que les medication requests ?
- Quelle est la différence entre exact et sémantique ?

### `df_scores_summary_by_note_style.csv`
Vue synthétique par style de note.

Ce fichier permet de répondre aux questions suivantes :
- Quels styles de notes sont les plus robustes ?
- Quels styles dégradent le plus la reconstruction ?
- Les difficultés varient-elles selon le type de note ?

## Utilisation conseillée

Si vous voulez comprendre rapidement la performance du projet :

1. commencez par `df_scores_overall_summary.csv`
2. lisez ensuite `df_scores_summary_by_note_style.csv`
3. consultez enfin `df_scores.csv` pour le niveau patient
