# Soil Fertilizer Predictor
This repository contains a machine learning pipeline for predicting the top 3 most suitable fertilizers based on environmental and agricultural features such as soil type, crop type, temperature, humidity, and nutrient levels. The project utilizes the CatBoost classifier for its superior handling of categorical features and robust performance on structured datasets.
The data used for this pipeline has been sourced from [Kaggle](https://www.kaggle.com/competitions/playground-series-s5e6/data).

## Methodology
Model: CatBoostClassifier (multi-class classification)

Evaluation Metric: MAP@3 (Mean Average Precision at 3)

Hyperparameter Tuning: Performed using Optuna for optimal learning rate, depth, and number of iterations

Categorical Handling: Automatic via CatBoost's internal encoding strategy

Input Features: Combination of numerical and categorical features (e.g. temperature, humidity, crop/soil types)

## How to run
Launch main.py within its directory.

## Results
The final model outputs the top 3 fertilizer recommendations ranked by confidence scores. CatBoost's native support for categorical features provided a leaner pipeline and stronger predictive power compared to conventional encoders.
