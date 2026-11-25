# Le pipeline, étape par étape

```text
FHIR → préfiltre → gold source → note → reconstruction → scores → vérification → analyse des erreurs
```

Chaque étape écrit ses sorties dans `data/`, et chaque réponse du modèle est mise en cache avec la signature de ses entrées. On peut donc arrêter et reprendre n'importe où : seules les étapes dont les entrées ont changé rappellent le modèle.

## 1. Préfiltre ([src/fhir.py](../src/fhir.py))

Un bundle Synthea contient tout l'historique d'un patient : consultations, examens, vaccins… On ne garde que la ressource Patient, les Conditions et les MedicationRequests.

Synthea range aussi des déterminants sociaux dans Condition (« Full-time employment », « Housing unsatisfactory »). Le préfiltre retire les cas évidents :

- les conditions de catégorie `social-history` ;
- celles dont le libellé contient une expression de `PREFILTER_DROP_SUBSTRINGS` ;
- celles qui ne sont pas codées en SNOMED ou ICD-10.

Une condition revient à chaque consultation, donc on ne garde qu'une occurrence par code. Le préfiltre est volontairement prudent : mieux vaut laisser passer un élément ambigu, que le LLM écartera ensuite, que jeter une vraie maladie.

## 2. Gold source ([src/gold.py](../src/gold.py))

Sur l'échantillon tiré, MedGemma relit les conditions candidates de chaque patient et renvoie les index de celles qui sont médicales. Trois garde-fous :

- il ne peut que **choisir** dans la liste, jamais ajouter ou reformuler ;
- le prompt lui demande de garder en cas de doute ;
- une réponse vide ou illisible revient à tout garder.

Les conditions retenues sont plafonnées à 12 par patient. Avec les prescriptions, elles forment le gold : la référence de toutes les comparaisons.

## 3. Notes ([src/notes.py](../src/notes.py))

MedGemma rédige une note clinique à partir du gold. Le prompt impose de citer le nom, la date de naissance, le genre et chaque élément, sans rien ajouter.

Quatre styles sont attribués à tour de rôle, pour mesurer l'effet de la forme sur la reconstruction : consultation courte, antécédents médicaux, bilan de santé, note télégraphique.

La sortie brute est ensuite nettoyée. Sans format de chat, le modèle ajoute parfois un « Final Answer », réécrit la note dans un bloc de code ou répète des phrases. On construit plusieurs versions candidates et on garde la plus plausible.

## 4. Reconstruction ([src/reconstruction.py](../src/reconstruction.py))

Le modèle reçoit **seulement la note** et renvoie un JSON : nom, date de naissance, genre, conditions, prescriptions. Si le JSON est illisible, on lui demande une fois de le réparer.

Les listes sont nettoyées (puces, numéros, doublons, éléments de plus de 160 caractères, bruit social), puis plafonnées. Les données démographiques sont scorées à part, mais celles du gold sont gardées dans la reconstruction finale, pour qu'une erreur sur un nom ne mélange pas deux patients dans les tables.

## 5. Scores ([src/evaluation.py](../src/evaluation.py))

Chaque reconstruction est comparée à deux références :

- **note vs recon** isole l'extraction : le modèle a-t-il retrouvé ce qui est écrit ?
- **source vs recon** mesure tout le trajet : que reste-t-il du dossier d'origine ?

Chacune est calculée en **exact** (même libellé après normalisation) et en **sémantique**. En sémantique, chaque élément reconstruit est d'abord cherché tel quel ; à défaut, le LLM désigne l'élément de référence qui lui correspond, ou aucun. Chaque élément de référence ne peut servir qu'une fois.

## 6. Vérification ([src/verification.py](../src/verification.py))

Une relecture globale : le modèle reçoit la note et la reconstruction, et rend un verdict (`consistent`, `minor_issues`, `major_issues`, `unknown`) avec la liste des éléments non justifiés ou manquants. Elle complète les scores, mais c'est le même modèle qui juge son propre travail : voir les limites dans le [rapport](benchmark_report.md#vérification).

## 7. Analyse des erreurs

En comparant les quatre scores d'un patient, on peut dire où l'information s'est perdue : à l'écriture de la note, à l'extraction, ou nulle part (simple reformulation). Les règles sont détaillées dans le [guide des résultats](results_guide.md#analyse-des-erreurs).

## Un mot sur la vitesse

Les appels qui attendent du JSON s'arrêtent dès que le premier objet JSON est fermé (`_StopAfterFirstJson` dans [src/llm.py](../src/llm.py)). Sans cela, le modèle répète sa réponse jusqu'à la limite de tokens. Le JSON lu est identique, et chaque appel de scoring devient 5 à 9 fois plus rapide : l'évaluation des 40 patients prend 18 minutes.
