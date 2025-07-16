# Energy Demand Forecasting – Time Series Modeling

Prédiction de la consommation énergétique journalière par des approches statistiques, de machine learning et de deep learning – dans un objectif **académique**, **autodidacte** et **pédagogique**.


## Introduction : positionnement et objectifs

Ce projet s’inscrit dans une démarche de formation personnelle avancée sur la modélisation de séries temporelles appliquée à l’énergie. Objectifs principaux :

- Approfondir les concepts mathématiques liés aux modèles de séries temporelles (ARIMA, réseaux de neurones séquentiels, modèles probabilistes…)
- Maîtriser les étapes clés du prétraitement des données temporelles, la création de variables exogènes, et la validation croisée temporelle
- Comprendre les limites, les biais et les hypothèses associées à chaque méthode
- Comparer des modèles dans un cadre expérimental rigoureux

> Ce projet repose sur des données publiques simplifiées et **ne vise pas une application industrielle directe**, mais bien une **exploration pédagogique** des outils de modélisation.



## Données

### Sources

Les données proviennent de plusieurs jeux publics et dérivés :

- [RTE France Open Data](https://opendata.rte-france.com) : consommation électrique réelle (journalière / horaire)
- [OpenWeatherMap API](https://openweathermap.org/api) & jeux [Kaggle](https://www.kaggle.com/) : données météo (température, humidité, etc.)
- Sources manuelles dérivées : calendrier français (jours fériés, week-ends, vacances scolaires)

### Contenu et structure

Les données sont structurées en séries temporelles multivariées, avec la structure suivante :

| Catégorie     | Variables                              | Type / Exemple                        |
|---------------|----------------------------------------|----------------------------------------|
| **Cible**     | `load_MW`                              | Consommation électrique (MW)          |
| **Temporelles** | `date`, `hour` / `day`               | Timestamp (horaire/journalière)       |
| **Météo**     | `temperature`, `humidity`              | °C, % (exogènes)                      |
| **Calendrier**| `day_of_week`, `is_weekend`, `is_holiday` | Booléens / catégories             |
| **Dérivées**  | `lag_1`, `lag_7`, `rolling_mean_24h`   | Lags, moyennes mobiles                |

### Fréquence et période

- **Fréquence** : horaire ou journalière (selon expérimentation)
- **Période couverte** : 2016 à 2022
- **Taille des jeux** : de 5 000 à 50 000 lignes selon granularité


## Objectif technique

L’objectif est de construire un estimateur de la forme :
ŷ{t+h} = f(y_t, y_{t-1}, …, y_{t-p}, X_t, X_{t-1}, …)

Où :

- `y_t` est la consommation à l’instant `t`
- `X_t` regroupe les variables exogènes (météo, calendrier)
- `h` est l’horizon de prévision (quelques heures à plusieurs jours)

Les performances sont évaluées selon des métriques classiques suivantes :

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## Formulation mathématique

Chaque notebook présente les équations et hypothèses associées à chaque modèle :

| Modèle        | Description |
|---------------|-------------|
| **ARIMA**     | Modèle auto-régressif intégré à moyenne mobile, hypothèse de stationnarité |
| **XGBoost**   | Régression supervisée par boosting d’arbres, sur des features dérivées |
| **LSTM / DeepAR** | RNN séquentiels pour modéliser les dépendances temporelles longues |


## Modèles implémentés  

| Méthode       | Type                         | Librairie     |
|---------------|------------------------------|---------------|
| ARIMA / SARIMA| Modèle linéaire              | `statsmodels` |
| Prophet       | Modèle additif robuste       | `prophet`     |
| XGBoost       | Apprentissage supervisé       | `xgboost`     |
| LSTM          | Deep learning séquentiel     | `Keras`, `PyTorch` |
| DeepAR        | RNN probabiliste             | `GluonTS`     |


## Structure du dépôt
