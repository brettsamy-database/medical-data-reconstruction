# Pipeline détaillée

## Vue d’ensemble

La pipeline suit cette logique :

**filtered -> gold -> note -> reconstruction -> scoring -> vérification -> analyse des erreurs**

Chaque étape a un rôle précis.

## 1. Filtered

Cette étape correspond au premier filtrage des bundles FHIR.

Le but est de réduire le bruit et de garder les éléments les plus pertinents pour la suite.

Cette étape sert à :

- alléger les données
- éviter de surcharger le pipeline
- retirer une partie des éléments non utiles à la génération de notes

## 2. Gold source

Le gold source est la source structurée de référence retenue après filtrage.

C’est cette version qui sert de base à la génération de notes.

Elle contient principalement :

- l’identité du patient
- les conditions retenues
- les medication requests retenues

Cette étape est centrale, car elle définit la référence utilisée pour la génération et l’évaluation.

## 3. Note générée

À partir du gold source, le pipeline génère une note clinique non structurée.

Le projet utilise plusieurs styles de notes afin de tester la robustesse du pipeline face à des formulations variées.

Exemples de styles :

- short consultation note
- health check summary
- medical history note
- telegraphic note

## 4. Reconstruction

À partir de la note générée, le pipeline tente de reconstruire une structure médicale.

On distingue en général :

- une reconstruction brute
- une reconstruction nettoyée

L’objectif est de retrouver :

- les conditions
- les medication requests
- certains champs démographiques

## 5. Scoring

Le scoring mesure la qualité de la reconstruction.

Le projet distingue deux lectures complémentaires :

- **note vs recon** : comparaison entre la note générée et la reconstruction
- **source vs recon** : comparaison entre la source structurée de référence et la reconstruction

On utilise plusieurs métriques :

- precision
- recall
- F1
- exact match
- matching sémantique

Cette double lecture permet de distinguer :

- ce qui est bien reconstruit à partir de la note
- ce qui a déjà pu être perdu entre la source structurée et la note générée

## 6. Vérification

La vérification ajoute une couche de relecture automatique.

Elle sert à évaluer si la reconstruction semble cohérente avec la note générée.

Elle ne remplace pas les scores, mais elle ajoute une lecture critique supplémentaire.

## 7. Analyse des erreurs

L’analyse des erreurs sert à interpréter les écarts observés dans les scores.

Elle permet de comprendre :

- où ça casse
- quels types d’erreurs dominent
- si le problème vient plutôt de la transmission source -> note
- de la reconstruction depuis la note
- ou du matching exact et de la canonicalisation

## 8. Résultats finaux

La pipeline produit ensuite :

- des résultats intermédiaires
- des métriques de synthèse
- une vérification
- des cas difficiles
- des exemples patients

## Voir les fichiers associés

- [Guide des résultats](results_guide.md)
- [Carte du dépôt](repository_map.md)
