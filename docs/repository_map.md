# Organisation du dépôt

```text
config/settings.py       tous les paramètres : chemins, modèle, échantillon, filtres, plafonds, tokens, températures
src/
  utils.py               normalisation de texte, JSON sur disque, signatures de cache
  llm.py                 chargement de MedGemma, arrêt de génération sur JSON, lecture tolérante du JSON
  fhir.py                téléchargement Synthea, préfiltre, échantillon
  gold.py                tri des conditions par le LLM → gold source
  notes.py               styles, prompt de note, nettoyage des sorties
  reconstruction.py      prompt d'extraction, réparation du JSON, nettoyage des listes
  evaluation.py          métriques exactes et sémantiques, analyse des erreurs, synthèses
  verification.py        relecture globale par le LLM
  export.py              tables finales depuis le disque, publication dans results/
  plots.py               figures de assets/figures/
notebooks/               les trois étapes à lancer dans l'ordre
tests/                   tests pytest, avec un faux LLM (aucun GPU nécessaire)
results/                 résultats publiés : CSV et exemples patients
assets/figures/          figures du README et de la documentation
docs/                    cette documentation
data/                    données de travail, non versionnées (régénérables)
  raw/                   archive et bundles Synthea
  processed/             bundles filtrés, notes, reconstructions, scores, caches LLM, exports
  gold_standard/         gold source, un JSON par patient
```

Les modules suivent l'ordre du pipeline, et chacun ne dépend que des précédents. `evaluation.py` réutilise par exemple le nettoyage de `reconstruction.py`, pour que le gold et la reconstruction soient normalisés de la même façon avant comparaison.
