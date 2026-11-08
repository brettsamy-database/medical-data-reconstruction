# Figures

Figures du benchmark (40 patients), utilisées dans le README et dans `docs/`.

| Fichier | Contenu |
|---|---|
| `overall_f1.png` | F1 exacts et sémantiques globaux (note vs recon, source vs recon) |
| `style_exact_f1.png` | F1 exacts par style de note |
| `error_categories.png` | types d'erreurs dominants (conditions, medication requests) |
| `verification_by_style.png` | verdicts de vérification par style de note |

Pour les régénérer à partir des CSV de `results/` : `from src.plots import make_all_figures; make_all_figures()`
(ou la dernière section de `notebooks/03_benchmark_results.ipynb`).
