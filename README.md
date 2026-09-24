# Challenge Data : Prédiction des Performances d'Allocations d'Actifs

**Organisé par :** Qube Research & Technologies (QRT)

## Informations Générales

* **Catégorie :** Sciences économiques, Finance
* **Type de problème :** Classification, Séries temporelles
* **Taille du dataset :** 10Mo à 1Go
* **Niveau :** Intermédiaire
* **Date de début :** 22 janvier 2026

---

## 1. Contexte : Faire confiance, ou parier contre ?

Dans le monde du trading systématique, les allocations d'actifs sont omniprésentes, mais la qualité des signaux fait toute la différence. Chaque jour, les traders reçoivent de nombreuses allocations candidates.

La question centrale de ce challenge est : **Pouvez-vous prédire si une allocation d'actifs donnée mérite d'être suivie (performance positive), ou si à l'inverse il vaut mieux parier contre (performance négative) ?**

### Qu'est-ce qu'une allocation d'actifs ?

C'est une méthode systématique de construction de portefeuille. Chaque allocation est définie par un vecteur de poids (positifs ou négatifs), définis chaque jour et tenus pour toute une session de trading. D'un jour à l'autre, une allocation peut rééquilibrer ses poids (turnover). La performance journalière représente les performances agrégées des positions pondérées.

---

## 2. Définitions Mathématiques

Pour un jour $t$, une allocation $S$, et $M$ actifs dans un univers de trading :

* **Poids de l'allocation $S$ au jour $t$ :**

$$w_{S,t} = (w_{S,t,1}, w_{S,t,2}, \dots, w_{S,t,M})$$


* **Performance (rendement) d'un actif $i$ du jour $t$ au jour $t+1$ :**
$r_{i,t+1}$
* **Rendement réalisé de l'allocation $S$ à $t+1$ :**

$$r_{S,t+1} = \sum_{i=1}^{M} w_{S,t,i} \times r_{i,t+1}$$



---

## 3. But du Challenge

Chaque ligne du dataset représente **une journée et une allocation d'actifs**.
Les données décrivent le comportement historique de l'allocation lors des 20 jours précédents (performances, liquidité, turnover).

L'objectif est de prédire le **signe du rendement futur** de cette allocation :

* `1` : Performance positive (faire confiance à l'allocation).
* `0` : Performance négative (parier contre l'allocation).

---

## 4. Métrique d'évaluation (Accuracy)

La métrique évalue la capacité du modèle à prédire correctement la direction (le signe) de la performance future, et non sa magnitude.

$$Accuracy = \frac{1}{T \times M} \sum_{t=1}^{T} \sum_{S=1}^{M} \mathbb{1}[\text{sign}(\hat{r}_{S,t+1}) = \text{sign}(r_{S,t+1})]$$

* $N$ : nombre total de lignes ($N = T \times M$).
* $\text{sign}(x) = 1$ si $x > 0$, sinon $0$.
* $\mathbb{1}$ : Fonction indicatrice (1 si condition vraie, 0 sinon).
* $r_{S,t+1}$ : Vrai rendement futur.
* $\hat{r}_{S,t+1}$ : Prédiction du rendement futur.

---

## 5. Description des Données

Le dataset est une série temporelle avec un multi-index `(date, allocation)`.

* **Entraînement (`X_train.csv`) :** 527 073 observations. Les vraies performances futures sont données dans `y_train.csv`.
* **Test (`X_test.csv`) :** 31 870 observations.

### Dictionnaire des colonnes

* `TS` : Timestamp du snapshot (dates anonymisées, ex: DATE_0001). Pas de garantie de continuité absolue.
* `ALLOCATION` : Identifiant de l'allocation (ex: ALLOCATION_01).
* `RET_{i}` ($i \in [1, 20]$) : Rendement de l'allocation au jour passé $i$.
* `SIGNED_VOLUME_{i}` ($i \in [1, 20]$) : Volume signé de l'allocation au jour passé $i$.
* `MEDIAN_DAILY_TURNOVER` : Turnover médian de l'allocation sur les 20 derniers jours.
* `GROUP` : Groupe anonymisé auquel appartient l'allocation.
* `TARGET` : Rendement futur de l'allocation (variable à prédire dans `y_train.csv`).

### Précisions Techniques (Volumes, Poids et Turnover)

**Contrainte des poids :**
À chaque jour $t$, chaque allocation $S$ respecte :


$$\sum_{i=1}^{M} \vert{}w_{S,i,t}\vert{} = 1$$

**Volume signé (`SIGNED_VOLUME`) :**
Où $V_{i,t}$ est le volume total échangé sur le marché pour l'instrument $i$.


$$V_{S,t} = \sum_{i=1}^{M} w_{S,t,i} \times V_{i,t}$$


*(Note : Ces volumes ont été rescalés de manière "roulante" pour assurer la comparabilité).*

**Turnover et Turnover Médian (`MEDIAN_DAILY_TURNOVER`) :**
Le turnover journalier est la somme absolue des rééquilibrages de poids :


$$TO_{S,t} = \sum_{i=1}^{M} \vert{}w_{S,t,i} - w_{S,t-1,i}\vert{}$$


Le turnover médian fourni en feature correspond à :


$$MDT_{S,t} = \text{median}(TO_{S,t}, TO_{S,t-1}, \dots, TO_{S,t-20})$$

---

## 6. Fichiers Fournis

Tous les fichiers sont indexés par un `ROW_ID` unique, représentant le tuple `(date, allocation)`.

* `X_train.csv` : Variables explicatives (Features) d'entraînement.
* `y_train.csv` : Variables cibles (Targets) d'entraînement.
* `X_test.csv` : Variables explicatives de test.
* `sample_submission.csv` : Exemple de soumission au format attendu.
* `benchmark_submission.ipynb` : Notebook générant la baseline du leaderboard.

---

## 7. Baseline / Benchmark

Un notebook de référence est fourni. Il construit des features additionnelles :

1. Moyenne journalière des performances passées des allocations (différents horizons).
2. Moyenne journalière des performances passées de *toutes* les allocations.
3. Volatilité passée sur les 20 derniers jours.
4. Moyenne des volatilités passées.

**Modèles testés dans la baseline :**

* Régression Ridge calibrée sur toutes les features.
* Modèle LightGBM calibré sur toutes les features avec cross-validation. (Score public de la baseline LGBM : **0.5079**).