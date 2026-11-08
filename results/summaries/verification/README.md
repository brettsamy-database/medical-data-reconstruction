# Verification

Ce dossier contient les résultats de la chaîne de vérification LLM.

## Contenu

- `df_verification.csv` : verdicts de vérification patient par patient

## Rôle de la vérification

La vérification ajoute une couche supplémentaire de relecture automatique.

Elle ne remplace pas les scores.
Elle sert à évaluer si la reconstruction semble cohérente avec la note générée.

## Ce que ce fichier permet de voir

- les verdicts de cohérence,
- la confiance associée,
- les patients qui paraissent plus fragiles ou plus ambigus.

## Utilisation conseillée

Ce fichier est utile pour :

- compléter la lecture des scores,
- repérer des cas discutables,
- enrichir l'analyse des résultats au-delà des seules métriques numériques.
