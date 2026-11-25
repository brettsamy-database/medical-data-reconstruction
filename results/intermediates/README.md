# Tables intermédiaires

Les trois états successifs du dossier de chaque patient :

| Fichier | Contenu |
|---|---|
| `df_gold.csv` | le gold source : identité, style de note attribué, conditions et prescriptions de référence |
| `df_notes.csv` | la note générée, en version brute et en version nettoyée (la seule que voit la reconstruction) |
| `df_recon.csv` | la reconstruction : JSON brut, listes nettoyées (scorées) et intersection exacte avec le gold (pour déboguer) |

Les listes sont stockées deux fois : en JSON (`*_json`), pour être relues par du code, et en aperçu lisible (`*_preview`, les 6 premiers éléments).
