# Energy Demand Forecasting – Multi-Scale & Multi-Horizon Time Series Modeling

Prédiction de la consommation énergétique horaire à deux échelles géographiques (France, Paris) sur des horizons allant de 1 heure à 7 jours, via des approches statistiques, de machine learning et de deep learning. Projet à visée pédagogique, exploratoire et professionnalisante.


## Introduction – Positionnement et objectifs

Ce projet s’inscrit dans une démarche de spécialisation personnelle sur la modélisation de séries temporelles appliquée à l’énergie, avec une attention portée à :
	• L’intégration de données hétérogènes (météo, calendrier…)
	• La structuration multi-échelle (France vs Paris)
	• La prédiction à court et moyen terme (1h → 7j)

Objectifs pédagogiques et techniques :
	• Approfondir les concepts théoriques et pratiques de modélisation séquentielle (ARIMA, LSTM, DeepAR…)
	• Travailler le prétraitement de séries temporelles réelles complexes
	• Comparer des modèles sur plusieurs horizons et zones géographiques
	• Explorer les effets de variables exogènes sur la qualité des prévisions

> Ce projet ne constitue pas un produit industriel mais un projet exploratoire avancé centré sur la modélisation et l’analyse.



## Données

### Sources utilisées

Source	Description
RTE France Open Data	Consommation électrique réelle de la France (horaire)
OpenWeatherMap API	Données météo historiques pour Paris et régions françaises
Kaggle	Jeux alternatifs de consommation énergétique régionale
OpenData Paris	Données publiques énergétiques ou environnementales de la ville
Librairies Python	Génération de calendriers (week-ends, jours fériés, saisons, etc.)


### Structure des jeux de données

Deux datasets sont construits :
	• national_dataset.csv : consommation énergétique horaire en France entière
	• paris_dataset.csv : consommation ou proxy énergétique pour Paris

Catégorie	Variables	Exemple
Cible	load_MW	Consommation (MW)
Temporelles	datetime, hour, day_of_week, month	Composantes horaires
Calendrier	is_weekend, is_holiday, season	Booléens / catégoriels
Météo	temperature, humidity, wind_speed, irradiance	Données exogènes
Techniques	lag_t-1, lag_t-24, rolling_mean_24h, etc.	Lags, moyennes mobiles
- Fréquence : horaire
- Période couverte : 2018 à 2023 (variable selon source)
- Horizon de prédiction :
- Court terme : +1h, +6h, +24h
- Moyen terme : +2j à +7j



## Objectif technique

Construire un estimateur de la forme :

ŷ{t+h} = f(y_t, y_{t-1}, …, y_{t-p}, X_t, X_{t-1}, …)

Avec :
- 'y_t' : la consommation à l’instant t
- 'X_t' : variables exogènes (météo, calendrier, effets saisonniers)
- 'h' : l’horizon de prévision

Objectif : prédire 'load_MW' pour t+h, avec 'h' ∈ {1h, 6h, 1j, 7j}

Évaluer les performances selon :
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)



## Modélisation – Formulation mathématique

Chaque modèle est implémenté dans un notebook dédié avec :
- Équations fondamentales
- Hypothèses (stationnarité, non-linéarité, indépendance)
- Méthode d’entraînement et de validation temporelle

Modèle	Description
ARIMA/SARIMA	Modèle linéaire avec composantes saisonnières
XGBoost	Modèle supervisé sur features dérivées (lags, encodage)
Prophet	Modèle additif (tendance, saison, jour férié)
LSTM	RNN pour dépendances longues
DeepAR	Réseaux probabilistes avec estimation d’incertitude


## Implémentation – Modèles et librairies

Méthode	Type	Librairie
ARIMA / SARIMA	Modèle linéaire	statsmodels
Prophet	Modèle additif robuste	prophet
XGBoost	Régression supervisée	xgboost
LSTM	Deep learning séquentiel	Keras, PyTorch
DeepAR	RNN probabiliste	GluonTS



## Évaluation & Expérimentations

Le projet inclut :
- Des splits temporels glissants (rolling windows)
- Une évaluation à horizons multiples (+1h, +24h, +7j)
- Une comparaison entre France et Paris pour tester la généralisation
- L’analyse de l’apport réel des variables météo




