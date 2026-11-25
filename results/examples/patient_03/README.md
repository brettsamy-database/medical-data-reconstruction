# Patient 03 : un cas difficile

**Style de note :** consultation courte · **6 conditions, 5 prescriptions** dans la gold source

Ici, l'information se perd à deux endroits différents, et la vérification ne s'en aperçoit pas.

## 1. Pendant l'écriture de la note

MedGemma rédige une vraie note de consultation, avec des symptômes, une température et une mère qui accompagne l'enfant. Mais en route, il oublie deux diagnostics (*Otitis media*, *Streptococcal sore throat*) et une prescription (*Penicillin V Potassium*). Il ne fait plus qu'évoquer le streptocoque :

> *Given the potential for Streptococcus, I am also considering Amoxicillin 500 MG Oral Tablet.*

Ces éléments ne pourront jamais être reconstruits, puisqu'ils ne figurent plus dans la note.

## 2. Pendant la reconstruction

La note dit :

> *[...] also has a history of child attention deficit disorder. **I suspect** acute viral pharyngitis, viral sinusitis, and acute bronchitis.*

MedGemma ne garde que le trouble de l'attention, le seul diagnostic posé comme acquis. Les trois diagnostics « suspectés » sont écartés. On peut défendre ce choix sur le plan clinique, mais il coûte cher au score :

| | F1 exact | F1 sémantique |
|---|---:|---:|
| Conditions (source → reconstruction) | 0.29 | **0.29** |
| Prescriptions (source → reconstruction) | 0.89 | 0.89 |

Les prescriptions s'en sortent mieux : les quatre présentes dans la note sont toutes retrouvées. Seule manque la pénicilline, perdue dès l'étape 1.

## Ce que la vérification n'a pas vu

La vérification déclare la reconstruction **cohérente, avec une confiance de 1.0**, alors que trois diagnostics de la note sont absents. Le même modèle produit et juge le résultat, et il se montre ici trop indulgent. C'est l'une des limites du projet.

L'analyse des erreurs, elle, ne s'y trompe pas : elle classe les conditions en `recall_dominant_pipeline_error`, de sévérité **severe**.
