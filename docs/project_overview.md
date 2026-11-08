# Vue d’ensemble du projet

## Objectif

L’objectif du projet est d’évaluer la capacité d’un pipeline à reconstruire une information médicale structurée à partir de notes cliniques non structurées.

Le projet part de données FHIR synthétiques, puis construit une chaîne complète :

- filtrage
- structuration de référence
- génération de notes
- reconstruction
- scoring
- vérification
- analyse d’erreurs

## Problème étudié

Dans un dossier médical, l’information peut exister sous plusieurs formes :

- **forme structurée** : ressources FHIR, listes de conditions, médicaments
- **forme non structurée** : texte libre rédigé comme une note clinique

Le projet cherche à mesurer à quel point on peut revenir du texte libre vers une structure exploitable.

## Ressources FHIR étudiées

Le projet se concentre principalement sur :

- **Condition**
- **MedicationRequest**
- ainsi que certains champs démographiques utiles comme :
  - nom
  - date de naissance
  - genre

## Pourquoi ce projet est intéressant

Ce projet est utile parce qu’il traite une difficulté très concrète :

- les données structurées sont très utiles pour l’analyse
- mais beaucoup d’information médicale existe sous forme de texte libre
- il faut donc mesurer la qualité de la reconstruction de cette information

## Idée centrale

Le projet ne se contente pas de générer des notes.

Il benchmarke toute une chaîne de transformation :

1. structuré -> texte
2. texte -> structuré
3. comparaison avec la référence
4. interprétation des erreurs

## Résultat attendu

À la fin du pipeline, on ne veut pas seulement obtenir des scores.

On veut aussi comprendre :

- ce qui marche bien
- ce qui marche moins bien
- quels types de notes sont les plus faciles ou les plus difficiles
- quels types d’erreurs dominent

## Où aller ensuite

- [Pipeline détaillée](pipeline_explained.md)
- [Guide des notebooks](notebook_guide.md)
- [Guide des résultats](results_guide.md)
