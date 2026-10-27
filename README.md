# Lab 1 – Deep Learning (MBD) : Régression et classification avec des MLP PyTorch

Université Abdelmalek Essaadi – FST Tanger – Master MBD – Pr. Elaachak Lotfi

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `Lab1_Part1_Regression_NYSE.ipynb` | Partie 1 : régression sur le dataset NYSE |
| `Lab1_Part2_Classification_Maintenance.ipynb` | Partie 2 : classification multi-classes (Predictive Maintenance) |
| `README.md` | Rapport : méthode, résultats, interprétation et synthèse |

Exécution : Google Colab ou Kaggle (les notebooks trouvent le dataset automatiquement ; mode léger automatique sur CPU).
Les résultats ci-dessous proviennent d'une exécution sur **CPU, mode léger**, avec **une seule graine** (`SEED = 42`).

---

## Partie 1 – Régression (NYSE, 851 264 lignes, 501 actions, 2010-2016)

**Objectif :** prédire le cours de clôture du lendemain `close(t+1)`.

1. **EDA** : aucune valeur manquante ni doublon ; prix très asymétriques (1,6 $ à 1 578 $) → échelle log ; open/high/low/close corrélés à ~0,999 ; rendement journalier moyen 0,06 %, écart-type 1,81 % (queues épaisses).
2. **Features** : open, high, low, close, log(volume), rendement, moyennes mobiles 5/10 j, volatilité 5 j.
3. **Split chronologique** : train 601 664 / validation 69 218 / test 175 372 (test à partir du 2015-08-11) ; standardisation ajustée sur le train.
4. **Modèle** : MLP PyTorch, perte MSE. Baseline « persistance » (`close(t+1) = close(t)`) : **RMSE test = 2,212 $**.
5. **GridSearch** (`GridSearchCV` + `TimeSeriesSplit`, 24 combinaisons, 72 fits, sous-échantillon de 20 000 lignes).
   Meilleurs hyper-paramètres : **MLP (64, 32), Adam, lr = 0,01, 20 epochs, batch 2048**. Adam domine RMSprop dans le top 10 ; SGD n'a pas été testé en mode léger.
6. **Accuracy** de régression = part des prédictions à moins de 2 % du vrai cours.

### Modèle 1 (meilleurs hyper-paramètres)

| | RMSE ($) | MAE ($) | R² | Acc (<2 %) |
|---|---|---|---|---|
| Train | 1,512 | 0,733 | 0,9994 | 0,787 |
| Test | 2,403 | 1,119 | 0,9995 | 0,772 |
| *Baseline persistance (test)* | *2,212* | – | – | – |

**Interprétation des courbes.** La loss d'entraînement baisse vite puis plafonne. La loss de test est ~2,5 fois celle du train (0,00134 vs 0,00053) et sa valeur minimale est atteinte à l'epoch 9/20. Un pic d'instabilité apparaît à l'epoch 10 (loss test 0,0040, accuracy 0,61), puis le modèle se rétablit : c'est l'effet d'un learning rate élevé (0,01) avec Adam. L'écart train/test traduit en partie un **changement de régime** (le test est la période la plus récente, split chronologique) et pas seulement du surapprentissage. Le point important : **le MLP (RMSE 2,40 $) ne bat pas la baseline de persistance (2,21 $)** ; le R² de 0,9995 est trompeur car il est dominé par l'échelle des prix. Le marché est quasi efficient, les prix du jour contiennent déjà l'essentiel de l'information.

### Régularisation (même architecture / optimiseur / lr)

| Modèle | Epochs | RMSE train | RMSE test | MAE test | R² test | Acc test | Écart RMSE |
|---|---|---|---|---|---|---|---|
| Modèle 1 (aucune) | 20 | 1,512 | 2,403 | 1,119 | 0,9994 | 0,772 | 0,891 |
| Dropout 0,2 | 20 | 2,236 | 3,365 | 1,770 | 0,9989 | 0,572 | 1,129 |
| L2 (wd = 1e-4) | 20 | 1,641 | 2,638 | 1,238 | 0,9993 | 0,721 | 0,997 |
| L1 (1e-5) | 20 | 1,480 | 2,352 | 1,077 | 0,9995 | 0,769 | 0,872 |
| BatchNorm | 20 | 2,789 | 4,711 | 2,157 | 0,9979 | 0,527 | 1,922 |
| **Early stopping (p=5)** | 14 | 1,458 | **2,285** | 1,053 | 0,9995 | 0,778 | **0,827** |
| Dropout + L2 + ES | 11 | 1,795 | 2,817 | 1,381 | 0,9992 | 0,678 | 1,022 |
| BN + Dropout + L2 + ES | 8 | 1,915 | 2,684 | 1,481 | 0,9993 | 0,598 | 0,769 |

**Comparaison.** Seuls l'**early stopping** (RMSE test 2,285 $) et, de peu, **L1** (2,352 $) améliorent le modèle 1 (2,403 $) ; aucune variante ne passe sous la baseline de persistance (2,212 $). Dropout et BatchNorm **dégradent** nettement le modèle : la relation à apprendre est quasi l'identité (prix demain ≈ prix aujourd'hui), le bruit injecté par le dropout ou la normalisation par lot nuit à la précision (sous-apprentissage). Les écarts entre early stopping, L1 et modèle 1 sont faibles et obtenus avec une seule graine, avec un entraînement instable (pic à l'epoch 10) : ils sont à prendre avec prudence.

---

## Partie 2 – Classification multi-classes (Predictive Maintenance, 10 000 lignes)

**Objectif :** prédire le type de panne (`Failure Type`).

1. **Pré-traitement** : pas de manquants ni de doublons ; 27 lignes incohérentes entre `Target` et `Failure Type` supprimées (18 *Random Failures* avec Target = 0, 9 *No Failure* avec Target = 1), ce qui retire la classe *Random Failures* → **5 classes, 9 973 lignes**. `UDI`, `Product ID` et `Target` (fuite d'information) exclus ; ajout de `Temp_diff` et `Power` ; one-hot de `Type`. Outliers IQR : 418 sur la vitesse de rotation, 70 sur le couple (conservés : mesures physiques plausibles).
2. **EDA** : très fort déséquilibre : No Failure 96,69 %, Heat Dissipation 1,12 %, Power 0,95 %, Overstrain 0,78 %, Tool Wear 0,45 %.
3. **Split stratifié** train 6 781 / validation 1 197 / test 1 995 ; standardisation sur le train ; **SMOTE sur le train uniquement** : 6 781 → 32 785 lignes (classes équilibrées).
4. **GridSearch** (`imblearn.Pipeline(scaler → SMOTE → MLP)`, CV stratifiée 3 plis, 36 combinaisons, 108 fits, F1-macro). Meilleurs hyper-paramètres : **MLP (128, 64), Adam, lr = 0,01, 30 epochs, batch 512** (F1-macro CV = 0,677 ± 0,021). Modèle final entraîné 40 epochs.

### Modèle 1

| Jeu | Accuracy | Précision (macro) | Sensibilité (macro) | F1 (macro) | F1 (pondéré) |
|---|---|---|---|---|---|
| Train (original) | 0,9926 | 0,853 | 0,995 | 0,904 | 0,994 |
| Test | 0,9714 | 0,626 | 0,723 | 0,666 | 0,975 |

Par classe (test) :

| Classe | Sensibilité | Spécificité | Précision | F1 | Support |
|---|---|---|---|---|---|
| Heat Dissipation | 0,864 | 0,996 | 0,731 | 0,792 | 22 |
| No Failure | 0,980 | 0,758 | 0,992 | 0,986 | 1 929 |
| Overstrain | 0,812 | 0,996 | 0,650 | 0,722 | 16 |
| Power | 0,737 | 0,996 | 0,667 | 0,700 | 19 |
| Tool Wear | 0,222 | 0,990 | 0,091 | 0,129 | 9 |

**Interprétation des courbes.** La loss d'entraînement (données SMOTE) descend jusqu'à ~0,008 et l'accuracy train atteint 0,998, alors que la loss de test est minimale vers l'epoch 5 (~0,11) puis **remonte** (0,25-0,30) : le modèle surapprend, en particulier les échantillons synthétiques. L'accuracy de test reste élevée (~0,96-0,97) mais, avec 96,7 % de *No Failure*, elle est peu informative : un classifieur qui prédit toujours « No Failure » obtiendrait ~96,7 %. Il faut juger sur le F1-macro (0,666) et la sensibilité par classe. Le point faible est **Tool Wear Failure** (sensibilité 0,22, F1 0,13, seulement 9 exemples de test) : cette panne dépend de l'usure de l'outil, mais beaucoup de lignes normales ont aussi une usure élevée, donc les exemples synthétiques de SMOTE ne suffisent pas à la séparer. La spécificité de *No Failure* (0,758) indique que ~24 % des vraies pannes sont classées « No Failure ».

### Régularisation

| Modèle | Epochs | Acc train | Acc test | F1-macro train | F1-macro test | Sensibilité-macro test | Écart F1 |
|---|---|---|---|---|---|---|---|
| Modèle 1 (aucune) | 40 | 0,9926 | 0,9714 | 0,904 | **0,666** | 0,723 | 0,238 |
| Dropout 0,3 | 40 | 0,9679 | 0,9539 | 0,745 | 0,651 | 0,812 | 0,094 |
| L2 (wd = 1e-3) | 40 | 0,9617 | 0,9489 | 0,702 | 0,621 | 0,824 | 0,081 |
| L1 (1e-5) | 40 | 0,9900 | 0,9684 | 0,879 | 0,665 | 0,734 | 0,214 |
| BatchNorm | 40 | 0,9677 | 0,9464 | 0,752 | 0,625 | 0,846 | 0,128 |
| Early stopping (p=8) | 13 | 0,9844 | 0,9739 | 0,782 | 0,626 | 0,630 | 0,156 |
| Dropout + L2 + ES | 16 | 0,9451 | 0,9363 | 0,650 | 0,594 | 0,850 | 0,057 |
| BN + Dropout + L2 + ES | 13 | 0,9406 | 0,9293 | 0,638 | 0,600 | 0,878 | 0,039 |

**Comparaison.** Les régularisations **réduisent l'écart train/test** (de 0,238 à 0,04-0,13 en F1-macro) mais **ne font pas progresser le F1-macro de test** : le modèle 1 (0,666) et L1 (0,665) restent en tête. En revanche, Dropout, L2, BatchNorm et leurs combinaisons augmentent nettement la **sensibilité-macro** (0,81 à 0,88 contre 0,72) au prix d'une précision plus faible (plus de fausses alarmes), un compromis intéressant en maintenance prédictive où une panne manquée coûte plus cher qu'une alerte inutile. L'early stopping seul obtient la meilleure accuracy (0,974) mais la plus faible sensibilité (0,63) : il illustre bien que l'accuracy est trompeuse ici. Limites : les classes rares n'ont que 9 à 22 exemples de test, donc les scores par classe sont très bruités ; une seule graine est utilisée ; le « meilleur modèle » est choisi sur le jeu de test (un choix sur la validation serait plus rigoureux).

---

## Synthèse personnelle (brouillon à reformuler avec vos mots)

- J'ai appris à construire un MLP en PyTorch (`nn.Module`, boucle d'entraînement, optimiseurs) et à l'encapsuler dans l'API sklearn pour utiliser `GridSearchCV`.
- En régression de séries financières, un split **chronologique** et une **baseline simple** sont indispensables : mon MLP (R² = 0,9995) ne bat pas la persistance (RMSE 2,40 $ vs 2,21 $), ce qui montre qu'un R² très élevé ne prouve rien tout seul.
- En classification déséquilibrée, l'**accuracy est trompeuse** : j'ai utilisé F1-macro, sensibilité et spécificité par classe, SMOTE appliqué **uniquement au train** (dans le pipeline du GridSearch) pour éviter toute fuite de données, et retiré la variable `Target` qui fuitait l'étiquette.
- La régularisation n'améliore pas toujours la performance globale : elle réduit le surapprentissage mais peut déplacer le compromis précision/sensibilité ; le bon choix dépend du coût métier des erreurs.
