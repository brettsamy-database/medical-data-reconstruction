# Results

Ce dossier regroupe les résultats produits par le benchmark BUPA.

L’objectif est de séparer clairement :

1. les résultats intermédiaires de la pipeline
2. les résultats finaux de performance
3. les résultats secondaires utiles pour une lecture plus détaillée
4. les exemples patients permettant de suivre la pipeline complète sur des cas individuels

## Organisation

- [intermediates/](intermediates/) : résultats intermédiaires principaux de la pipeline
- [summaries/](summaries/) : résultats finaux du benchmark
- [other_results/](other_results/) : résultats secondaires utiles pour une lecture plus détaillée
- [examples/](examples/) : exemples patients illustrant la pipeline de bout en bout

## Ordre de lecture conseillé

Si vous découvrez le projet, l’ordre conseillé est :

1. lire `summaries/`
2. consulter `intermediates/`
3. explorer `examples/`
4. utiliser `other_results/` si une lecture plus détaillée est nécessaire

## Pipeline résumée

La logique générale du projet est la suivante :

**filtered -> gold -> note générée -> reconstruction -> scoring -> vérification -> analyse des erreurs**

Ce dossier montre cette logique à deux niveaux :

- niveau global avec les tables de synthèse
- niveau patient avec les exemples individuels
