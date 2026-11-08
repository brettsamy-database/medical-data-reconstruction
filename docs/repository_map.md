# Carte du dépôt

## Racine

- `README.md` : page d'accueil du projet
- `requirements.txt` : dépendances (MedGemma via `transformers` / `torch`)
- `.env.example` : modèle du fichier `.env` qui contient `HF_TOKEN` (le `.env` n'est jamais versionné)
- `.gitignore` : exclut `data/`, `.venv/`, `.env`, caches

## `config/`

- `settings.py` : chemins, modèle, taille de l'échantillon, règles de préfiltrage, plafonds, budgets de tokens, températures

## `src/`

| Module | Rôle |
|---|---|
| `utils.py` | normalisation de texte, lecture/écriture JSON |
| `fhir.py` | téléchargement Synthea, préfiltrage, extraction patient, échantillon |
| `llm.py` | MedGemma, authentification Hugging Face, parsing JSON des sorties LLM |
| `gold.py` | filtre LLM des conditions, écriture/lecture du gold source |
| `notes.py` | styles de notes, prompts, génération, nettoyage |
| `reconstruction.py` | prompt d'extraction, parsing, post-traitement, 3 vues de reconstruction |
| `evaluation.py` | métriques exactes et sémantiques, analyse des erreurs, synthèses |
| `verification.py` | vérification LLM de cohérence note / reconstruction |
| `export.py` | reset, reconstruction des datasets depuis le cache, exports, publication dans `results/` |
| `plots.py` | figures de `assets/figures/` |

## `notebooks/`

- `01_fhir_exploration.ipynb`, `02_note_generation.ipynb`, `03_benchmark_results.ipynb` (voir le [guide des notebooks](notebook_guide.md))

## `tests/`

Tests `pytest` de chaque module, avec un faux LLM (aucun GPU nécessaire).

## `data/` (non versionné)

- `raw/` : zip et bundles Synthea extraits
- `processed/` : bundles filtrés, notes, reconstructions, scores, vérification, caches LLM, exports
- `gold_standard/` : gold source, un JSON par patient

## `results/` (versionné)

- `intermediates/` : source structurée, notes, reconstruction
- `summaries/` : métriques, erreurs, vérification
- `other_results/` : tables longues, index, revue des exports
- `examples/` : 3 patients suivis de bout en bout

## `docs/` et `assets/figures/`

Documentation Markdown et figures du benchmark.

## Parcours conseillé

1. `README.md`
2. `docs/project_overview.md`
3. `docs/pipeline_explained.md`
4. `docs/results_guide.md`
5. `results/summaries/`
6. `results/examples/`
