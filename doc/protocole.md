# Protocole de recherche : exploration multi-agent de tunnels

## 1. Problématique de l'étude

- **Q1 :** sans mémoire et sans communication, l'exploration MARL peut-elle être meilleure que l'aléatoire biaisé ?
- **Q2 :** une communication limitée (signaux) peut-elle rivaliser avec une communication par partage de carte ?

## 2. Socle commun

### 2.1 Environnement

| Élément | Valeur |
| --- | --- |
| Actions | rester, avancer, gauche, droite, arrière, relatives à l'orientation du drone |
| Observation de base | 4 rayons (devant, derrière, gauche, droite) de 3 cases pour *blocked* et *drone* + drapeau de case partagée |
| Orientation | direction du dernier déplacement effectué, tirée aléatoirement en début d'épisode pour chaque drone  |
| Récompense | d'équipe ou groupé, nombre de nouvelles cases découvertes, pénalité par pas |
| Critique | CNN sur la carte globale (murs, cases explorées, drones), utilisé à l'entraînement seulement |
### 2.2 Paramètres 
On fera varier le nombre d'agent dans la simulation. 

| Paramètre | Valeur | Raison |
| --- | --- | --- |
Budget d'entraînement	5 120 000 frames par run (--frames 5120000), identique pour toutes les conditions	à 3 M de frames la couverture sur val progressait encore ; un budget fixe rend les conditions comparables à effort égal| Fin d'un épisode d'évaluation | toutes les cases atteignables explorées, ou aucune nouvelle case pendant 2 × le diamètre de la carte (plafond de sécurité 10 000 pas) | l'épisode dure tant que la politique découvre encore quelque chose, quelle que soit la taille de la carte |
| Budget d'entraînement | 5 120 000 frames par run par défaut (*--frames 5120000*), augmenté si la couverture sur *val* progresse encore sur les dernières frames | on s'assure que la politique a convergé avant de l'évaluer |
| Collecte | *--n-envs 512 --frames-per-batch 25600*, soit 200 itérations | le budget est un multiple exact du batch, sinon BenchMARL l'arrondit au batch supérieur |
| Checkpoint évalué | le dernier, à 5 120 000 frames | pas de choix du meilleur checkpoint sur *val*, qui biaiserait la comparaison |

### 2.2 Données

Deux jeux de cartes, *small_maps* (toutes en 15x15) et *big_maps* (toutes en 50x50), avec les mêmes familles et le même découpage :

| Ensemble | Cartes (*small* / *big*) | Usage |
| --- | --- | --- |
| *train* | 70 / 72 | entraînement |
| *val* | 14 / 14 | réglage des hyperparamètres et choix entre variantes |
| *test* | 15 / 15 | évaluation finale, une seule fois, aucun réglage. Les 7 familles de *train* et 8 familles jamais vues, une carte chacune |



### 2.4 Évaluation commune

| Élément | Définition | Ce qu'il indique |
| --- | --- | --- |
|Graine |5 run associé à une graine différentes chacun| Permet une analyse statistique |
| Modes | **Greedy** et **Stochastique** |
| Couverture finale | part des cases atteignables explorées à la fin de l'épisode | performance globale |
| Épisodes à 100 % | part des épisodes atteignant 90 % des cases atteignables | fiabilité |
| Vitesse d'exploration | nombre de pas pour atteindre 90 % des cases atteignables, moyenne sur les épisodes qui y arrivent | rapidité de découverte, même sans couverture complète |
| Courbe de couverture | couverture moyenne en fonction du pas, avec son intervalle de confiance | évolution de l'exploration |


Résultats rapportés séparément pour les petites cartes (*small_maps*, 15x15) et les grandes (*big_maps*, 50x50), globalement puis par famille de cartes.


## 3. Expérience Q1 : apprendre fait-il mieux que l'aléatoire ?

**Hypothèse.** Oui, mais de façon limitée. L'observation ne contient ni les cases explorées ni l'action précédente : dans un couloir droit, les deux sens paraissent identiques. Une politique sans mémoire peut seulement y adopter un biais de direction fixe, et un identifiant permet à chaque drone d'avoir le sien.
De plus, les agents seront capable de remarquer la présence d'un agent dans une branche du tunnel qui lui est adjacente, de cette façon il vont pouvoir adopter un comportement adaptatif.  

**Conditions**

| Condition | Définition 
| --- | --- |
| Aléatoire (référence) | tirage uniforme parmi les déplacements légaux, sans « rester », sauf si aucun déplacement n'est possible |
| MLP sans ID | politique apprise, réactive |
| MLP + ID | politique apprise avec identifiant du drone |
| MLP sans ID + reward mixte| politique apprise, réactive |
| MLP + ID +  reward mixte| politique apprise avec identifiant du drone |

**Sans communication :** Aucun message n'est échangé. En exécution, un drone perçoit toutefois les autres à une distance au maximum de *view_range*. Le critique voit la carte globale pendant l'entraînement uniquement.

**Critère de décision.** Une condition bat l'aléatoire si son gain de couverture finale dépasse 5 points sur tout l'intervalle de confiance à 95 %. On conclut séparément pour chaque mode et chaque groupe de cartes.

**Baselines de comparaison.** Algorithmes non appris de *algo/*, sans mémoire de la carte ni communication, comme nos politiques. Ils utilisent les mêmes 4 drones, le même départ, la même observation locale (masque d'actions relatif au cap), les mêmes déplacements et les mêmes règles de collision. 

| Baseline | Définition |
| --- | --- |
| Aléatoire  | déplacement légal tiré uniformément, sans « rester » |
| Aléatoire vers l'avant  | avance si possible, sinon tourne à gauche ou à droite au hasard, demi-tour en dernier recours |
| Main gauche  | avance jusqu'au premier mur, puis le longe à gauche (gauche, avant, droite, arrière par ordre de priorité) |
| Main droite | même règle en longeant le mur droit |

## 4. Expérience Q2 : les signaux peuvent-ils rivaliser avec le partage de carte ?

**Hypothèse.** Les signaux est une information très faible, cependant elle peut faire apparaître des politiques complexes de coopération. Néanmoins on va sûrement avoir des résultats limité malgrès la communication, il va falloir donner plus de vision, par rapport aux simples 4 rayons.


**Conditions**

| Condition | Définition |
| --- | --- |
| Signaux $\in$ *m* | politique réactive qui émet et reçoit des signaux |
|Signaux $\in$ *m* + vision sur les futures intersections|politique réactive qui émet et reçoit des signaux, avec des agents qui voient lesfutures intersection dans une direction|

**Baselines de comparaison.**

| Baseline | Définition |
| --- | --- |
| Sans communication (meilleure condition de Q1) | politique réactive sans signal, référence basse |
| Frontières, carte individuelle (*frontier*) | chaque drone garde la carte de ce qu'il a vu et se dirige vers la frontière (case connue voisine d'une case inconnue) la plus proche |
| Frontières, carte partagée (*frontier_shared*) | même algorithme, avec une carte commune à tous les drones |

**Paramètres**

| Paramètre | Valeurs | Usage |
| --- | --- | --- |
| Nombre d'états du signal *m* | 2, 4, 6 | un entraînement séparé par valeur, car *m* fixe la taille des entrées et des sorties du réseau. La meilleure valeur sur *val* est seule comparée aux autres conditions sur *test*, les autres servent à la courbe de performance en fonction de *m* |

**Implémentation prévue**
- **Signaux :** chaque drone émet à chaque pas un signal parmi *m* états possibles, reçu par les autres au pas suivant, avec *m* valant 2, 4 ou 6 L'espace d'actions passe à 5 (nombre de déplacement possibles) × *m*.
- **Baselines à carte :** non apprises, elles utilisent toute la carte connue, sans fenêtre. Elles partent du même point, avec la même vue, les mêmes déplacements et les mêmes règles de collision que les drones RL.
- **Canal :** parfait mais de portée limitée : un signal n'est reçu que par les drones situés à *r* cases. Le nombre de case sera compté en terme de propagation dans le tunnel réel. On ajoutera un affaiblissement du signal sur les tunnels formant un angle droit. 

**Critère de décision.** Les signaux rivalisent avec le partage de carte si leur retard de couverture finale reste sous 5 points sur tout l'intervalle de confiance à 95 %. Une absence de différence significative ne suffit pas, et l'ablation doit montrer que les signaux sont vraiment utilisés.



