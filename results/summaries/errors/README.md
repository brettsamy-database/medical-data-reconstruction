# Errors

Ce dossier contient l'analyse qualitative des erreurs du benchmark.

## Contenu

- `df_error_analysis.csv` : taxonomie globale des erreurs
- `df_error_types_by_note_style_conditions.csv` : analyse des erreurs sur les conditions par style de note
- `df_error_types_by_note_style_medication_requests.csv` : analyse des erreurs sur les medication requests par style de note
- `df_hard_cases_by_note_style.csv` : cas difficiles identifiés par style de note

## Objectif

Ces fichiers permettent de répondre à une question différente des métriques :

> Le pipeline se trompe, mais de quelle manière ?

## Rôle des fichiers

### `df_error_analysis.csv`
Vue globale des catégories d'erreurs observées sur le benchmark.

### `df_error_types_by_note_style_conditions.csv`
Analyse des erreurs sur les conditions, en fonction du style de note.

### `df_error_types_by_note_style_medication_requests.csv`
Analyse des erreurs sur les medication requests, en fonction du style de note.

### `df_hard_cases_by_note_style.csv`
Table des cas difficiles, utile pour identifier les patients ou styles qui concentrent les plus gros écarts.

## Utilisation conseillée

Ce sous-dossier est particulièrement utile pour :

- comprendre les limites du pipeline,
- préparer une restitution orale,
- identifier les axes d'amélioration futurs,
- illustrer les différences entre styles de note.
