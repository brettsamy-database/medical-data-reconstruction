# Intermediates

Ce dossier contient les résultats intermédiaires principaux du pipeline.

Il permet de suivre la chaîne centrale du projet :

**source structurée -> note générée -> reconstruction**

## Contenu

- `df_gold.csv` : source structurée filtrée utilisée pour générer les notes
- `df_notes.csv` : notes générées, avec versions brutes et nettoyées
- `df_recon.csv` : reconstruction structurée obtenue à partir des notes

## Pourquoi ces fichiers sont importants

Ces tables permettent de comprendre concrètement :

- ce que le pipeline retient comme information médicale de référence,
- à quoi ressemblent les notes cliniques générées,
- ce que le système arrive à reconstruire à partir de ces notes.

## Utilisation conseillée

Ce sous-dossier est utile pour :

- comprendre la mécanique du pipeline,
- comparer visuellement la source, la note et la reconstruction,
- illustrer le projet dans une soutenance ou une démonstration.
