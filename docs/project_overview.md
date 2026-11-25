# Le projet en bref

## La question

Une grande partie de l'information médicale est écrite en texte libre : comptes rendus, notes de consultation, courriers. Pour l'exploiter (statistiques, aide à la décision, interopérabilité), il faut la remettre sous forme structurée. Les modèles de langage médicaux promettent de le faire. **Mais que perd-on en route ?**

Pour le mesurer, il faut une vérité de référence, ce que de vraies notes n'ont pas. J'ai donc fait le trajet dans les deux sens :

1. je pars d'un dossier structuré (FHIR), dont je connais le contenu exact ;
2. MedGemma en tire une note clinique en texte libre ;
3. MedGemma reconstruit le dossier à partir de cette note seule ;
4. je compare la reconstruction au dossier de départ.

Comme la note est conservée, on peut aussi dire **où** une information s'est perdue : pendant l'écriture de la note, ou pendant sa relecture.

## Le périmètre

- **Données** : les 555 patients synthétiques de l'échantillon public Synthea (FHIR R4, novembre 2021). Rien de réel, donc aucune contrainte de confidentialité.
- **Ressources** : `Condition` (diagnostics) et `MedicationRequest` (prescriptions), plus le nom, la date de naissance et le genre.
- **Modèle** : `google/medgemma-4b-it`, la version 4B de MedGemma. Elle tourne sur un GPU grand public et a été entraînée sur des textes médicaux.
- **Variable étudiée** : le style de la note (consultation courte, antécédents, bilan, télégraphique), pour voir si la forme du texte change ce qu'on peut en extraire.

## Les choix de conception

- **Deux références pour chaque score.** Comparer à la note isole l'extraction ; comparer au dossier d'origine mesure tout le trajet. L'écart entre les deux localise la perte.
- **Deux façons de compter.** Le score exact est objectif mais punit chaque reformulation ; le score sémantique, jugé par le LLM, tolère les synonymes mais dépend du juge. Les deux sont rapportés.
- **Un cache partout.** Chaque appel au modèle est enregistré avec la signature de ses entrées. On peut modifier une métrique et relancer l'évaluation en quelques secondes, sans regénérer les notes.
- **Un pipeline testable sans GPU.** Le modèle est derrière une seule interface (`get_llm`), qu'un faux modèle remplace dans les tests.

## La suite

- [Le pipeline, étape par étape](pipeline_explained.md)
- [Le rapport du benchmark](benchmark_report.md)
