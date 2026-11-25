# Exemples patients

Les scores moyens ne montrent pas *comment* le pipeline réussit ou échoue. Ces trois patients, tirés du benchmark, le montrent sur des cas concrets :

| Dossier | Style de note | Ce qu'il illustre |
|---|---|---|
| [patient_01](patient_01/) | Note télégraphique | Un cas presque parfait |
| [patient_02](patient_02/) | Antécédents médicaux | Score exact à 0, score sémantique à 1 : le piège des reformulations |
| [patient_03](patient_03/) | Consultation courte | Un cas difficile, où l'information se perd à deux endroits |

Chaque dossier suit le pipeline dans l'ordre :

| Fichier | Contenu |
|---|---|
| `01_filtered.json` | Le dossier FHIR après filtrage |
| `02_gold.json` | La gold source : ce qu'on cherche à retrouver |
| `03_note.txt` | La note clinique générée par MedGemma |
| `04_reconstruction.json` | Ce que MedGemma a extrait de la note |
| `05_score_and_07_errors.json` | Les scores et la classification des erreurs |
| `06_verification.json` | Le verdict de la vérification |
