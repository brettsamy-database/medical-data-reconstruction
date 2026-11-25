# Les notebooks

Le code est dans `src/` ; les notebooks enchaînent les étapes et affichent les tables intermédiaires.

| Notebook | Ce qu'il fait | GPU | Durée (40 patients, RTX 2000 Ada 16 Go) |
|---|---|---|---|
| [`01_fhir_exploration`](../notebooks/01_fhir_exploration.ipynb) | téléchargement Synthea, préfiltrage, tirage de l'échantillon | non | environ 1 min |
| [`02_note_generation`](../notebooks/02_note_generation.ipynb) | gold source, génération des notes, reconstruction | oui | environ 1 h |
| [`03_benchmark_results`](../notebooks/03_benchmark_results.ipynb) | scores, vérification, analyse des erreurs, exports, figures | oui | environ 18 min |

## Points d'attention

- Lancer les notebooks dans l'ordre. Chacun relit sur disque ce que le précédent a produit, on peut donc les lancer dans des sessions différentes.
- Garder la même valeur de `settings.N_PATIENTS` dans 01 et 02 : l'échantillon est retiré à chaque fois avec `RANDOM_SEED`.
- Relancer un notebook ne coûte presque rien : les réponses du modèle sont en cache (`data/processed/filter_cache/`, `scores/llm_align/`, `scores/llm_note_vs_recon/`, `verification/`) et ne sont recalculées que si leurs entrées changent.
- Les notes sont l'exception : la génération est aléatoire et n'est pas en cache. Relancer cette cellule produit de nouvelles notes, et donc de nouvelles reconstructions et de nouveaux scores.
- Dans le notebook 03, `RESET_BENCHMARK_OUTPUTS = True` efface les scores et la vérification pour repartir de zéro (après avoir modifié une métrique, par exemple), et `PUBLISH = True` remplace `results/` et `assets/figures/`.

## Sans GPU

Sans `torch`, le pipeline s'exécute quand même de bout en bout : le gold garde toutes les conditions candidates, les notes sont remplacées par un gabarit et les reconstructions restent vides. C'est utile pour vérifier qu'une modification ne casse rien, pas pour produire des résultats.
