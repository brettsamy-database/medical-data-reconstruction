# Patient 01 : un cas presque parfait

**Style de note :** télégraphique · **10 conditions, 1 prescription** dans la gold source

La note télégraphique ressemble à une liste : chaque condition et la prescription y sont recopiées à peu près telles quelles. Rien ne se perd entre la source et la note, et MedGemma retrouve tout.

```text
Geri861 Padberg411, 1944-11-18, female today. Acute viral pharyngitis. Body mass index 30+ - obesity.
Prediabetes. Miscarriage in first trimester. [...] Hyperlipidemia. Viral sinusitis. Simvastatin 10 MG Oral Tablet.
```

| | F1 exact | F1 sémantique |
|---|---:|---:|
| Conditions (source → reconstruction) | 0.70 | **1.00** |
| Prescriptions (source → reconstruction) | 1.00 | **1.00** |

**Pourquoi le score exact n'est pas à 1 sur les conditions ?** Le modèle a fait un peu de ménage : « Acute viral pharyngitis (disorder) » devient « Acute viral pharyngitis », et « Body mass index 30+ - obesity (finding) » devient « Obesity ». Le sens est intact, ce que confirme le score sémantique à 1. L'analyse des erreurs classe ce cas en `exact_wording_gap_only` : un simple écart de formulation.

La vérification juge la reconstruction cohérente, avec une confiance de 1.0.
