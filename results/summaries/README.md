# Summaries

Ce dossier contient les résultats finaux du benchmark BUPA.

Il est organisé en trois sous-parties :

- [`metrics/`](metrics/) : métriques de performance et synthèses globales
- [`errors/`](errors/) : analyse des erreurs et cas difficiles
- [`verification/`](verification/) : résultats de la chaîne de vérification LLM

## Objectif

L'objectif de cette organisation est de séparer clairement :

1. la **performance quantitative**,
2. l'**interprétation qualitative des erreurs**,
3. la **relecture critique automatique** des reconstructions.

## Ordre de lecture conseillé

1. commencer par `metrics/`
2. poursuivre avec `errors/`
3. terminer par `verification/`

Cette séquence permet d'aller :
- des scores globaux,
- vers les causes des erreurs,
- puis vers la lecture critique automatique des sorties.
