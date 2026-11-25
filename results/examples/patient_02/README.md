# Patient 02 : le piège des reformulations

**Style de note :** antécédents médicaux · **4 conditions, 4 prescriptions** dans la gold source

Ce patient montre pourquoi le score exact seul ne suffit pas pour juger les conditions.

La note reprend fidèlement les quatre diagnostics, mais en langage naturel :

> *Genny123 Rath779 [...] has a history of anemia, acute bacterial sinusitis, chronic sinusitis, and osteoporosis.*

MedGemma les retrouve tous, mais les écrit comme dans la note :

| Gold source | Reconstruction |
|---|---|
| Anemia (disorder) | anemia |
| Acute bacterial sinusitis (disorder) | acute bacterial sinusitis |
| Chronic sinusitis (disorder) | chronic sinusitis |
| Osteoporosis (disorder) | osteoporosis |

| | F1 exact | F1 sémantique |
|---|---:|---:|
| Conditions | **0.00** | **1.00** |
| Prescriptions | 1.00 | 1.00 |

Le score exact compte zéro bonne réponse, alors que la reconstruction est parfaite. Ce cas résume l'écart entre le F1 exact (0.29) et le F1 sémantique (0.84) observé sur les conditions dans l'ensemble du benchmark.

Les prescriptions, elles, ont des libellés techniques (« Alendronic acid 10 MG Oral Tablet ») que personne ne reformule : elles sont recopiées mot pour mot.
