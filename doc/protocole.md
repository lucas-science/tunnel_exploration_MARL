# Protocole de recherche : exploration multi-agent de tunnels

## 1. Problématique de l'étude

- **Q1 :** sans mémoire et sans communication, l'exploration MARL peut-elle être meilleure que l'aléatoire ?
- **Q2 :** une communication limitée (signaux) peut-elle rivaliser avec une communication par partage de carte ?

## 2. Socle commun

### 2.1 Environnement

| Élément | Valeur |
| --- | --- |
| Actions | rester, avancer, gauche, droite, arrière, relatives à l'orientation du drone ; déplacements vers un mur masqués |
| Observation de base | 4 rayons de 3 cases (`blocked`, `drone`) vers l'avant, la gauche, la droite et l'arrière + drapeau de case partagée |
| Orientation | direction du dernier déplacement effectué, tirée au hasard pour chaque drone en début d'épisode |
| Récompense | d'équipe ou groupé, nombre de nouvelles cases découvertes, pénalité par pas |
| Critique | CNN sur la carte globale (murs, cases explorées, drones), utilisé à l'entraînement seulement |
### 2.2 Paramètres 
On fera varier le nombre d'agent dans la simulation. 
| Nombre de pas maximum | 2000, sans arrêt anticipé | savoir si la politique résout la carte ou se bloque |

### 2.2 Données

| Ensemble | Cartes | Usage |
| --- | --- | --- |
| `envs/train/` | 73 | entraînement |
| `envs/val/` | 14 | réglage des hyperparamètres et choix entre variantes |
| `envs/test/` | 16 | évaluation finale, une seule fois, aucun réglage |



### 2.4 Évaluation commune

| Élément | Définition | Ce qu'il indique |
| --- | --- | --- |
|Graine |5 run associé à une graine différentes chacun| Permet une analyse statistique |
| Modes | **Greedy** et **Stochastique** |
| Couverture finale | part des cases atteignables explorées au bout de 2000 pas | performance globale |
| Proportion blocage | part des épisodes atteignant 100 % de couverture |
| Courbe de couverture | couverture moyenne en fonction du pas, avec son intervalle de confiance | évolution de l'exploration |

### 2.6 Analyse statistique

- Unité d'analyse : moyenne par run et par carte.
- Résultats rapportés globalement, puis par famille de cartes.


## 3. Expérience Q1 : apprendre fait-il mieux que l'aléatoire ?

**Hypothèse.** Oui, mais de façon limitée. L'observation ne contient ni les cases explorées ni l'action précédente : dans un couloir droit, les deux sens paraissent identiques. Une politique sans mémoire peut seulement y adopter un biais de direction fixe, et un identifiant permet à chaque drone d'avoir le sien.
De plus, les agents seront capable de remarquer la présence d'un agent dans une branche du tunnel qui lui est adjacente, de cette façon il vont pouvoir adopter un comportement adaptatif.  

**Conditions**

| Condition | Définition 
| --- | --- |
| Aléatoire (référence) | tirage uniforme parmi les déplacements légaux, sans « rester », sauf si aucun déplacement n'est possible |
| MLP sans ID | politique apprise, réactive |
| MLP + ID | politique apprise avec identifiant du drone |

**Sans communication :** Aucun message n'est échangé. En exécution, un drone perçoit toutefois les autres à une distance au maximum de `view_range`. Le critique voit la carte globale pendant l'entraînement uniquement.

**Critère de décision.** Une condition bat l'aléatoire si son gain de couverture finale dépasse 5 points sur tout l'intervalle de confiance à 95 %. On conclut séparément pour chaque mode et chaque groupe de cartes.


## 4. Expérience Q2 : les signaux peuvent-ils rivaliser avec le partage de carte ?

Cette expérience demande d'implémenter la communication et ne commence qu'après Q1.

**Hypothèse.** s Les signaux est une information très faible, cependant elle peut faire apparaître des politiques complexes de coopération.

**Biais à contrôler.** Une carte apporte une mémoire en plus de la communication. La baseline à carte individuelle mesure l'apport de la mémoire seule, celle à carte partagée l'apport de la mémoire et de la communication.

**Conditions**

| Condition | Type | Définition |
| --- | --- | --- |
| Sans communication (meilleure condition de Q1) | RL | politique réactive sans signal, référence basse |
| Signaux + MLP | RL | politique réactive qui émet et reçoit des signaux |
| Frontières, carte individuelle | baseline non RL | chaque drone garde la carte de ce qu'il a vu et se dirige vers la frontière (case connue voisine d'une case inconnue) la plus proche |
| Frontières, carte partagée | baseline non RL | même algorithme, avec une carte commune à tous les drones |


**Paramètres**

| Paramètre | Valeurs | Usage |
| --- | --- | --- |
| Nombre d'états du signal `m` | 2, 4, 6 | un entraînement séparé par valeur, car `m` fixe la taille des entrées et des sorties du réseau. La meilleure valeur sur `val` est seule comparée aux autres conditions sur `test`, les autres servent à la courbe de performance en fonction de `m` |

**Implémentation prévue**

- **Signaux :** chaque drone émet à chaque pas un signal parmi `m` états possibles, reçu par les autres au pas suivant, avec `m` valant 2, 4 ou 6 L'espace d'actions passe à 5 (nombre de déplacement possibles) × `m`.
- **Baselines à carte :** non apprises, elles utilisent toute la carte connue, sans fenêtre. Elles partent du même point, avec la même vue, les mêmes déplacements et les mêmes règles de collision que les drones RL.
- **Canal :** parfait mais de portée limitée : un signal n'est reçu que par les drones situés à `r` cases au plus de l'émetteur (distance de Manhattan). `r` est fixé avant les expériences et.

**Critère de décision.** Les signaux rivalisent avec le partage de carte si leur retard de couverture finale reste sous 5 points sur tout l'intervalle de confiance à 95 %. Une absence de différence significative ne suffit pas, et l'ablation doit montrer que les signaux sont vraiment utilisés.



