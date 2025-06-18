import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from optuna.pruners import MedianPruner
from optuna.integration import CatBoostPruningCallback

# Ensure reproducibility
np.random.seed(42)

# Load training and test datasets from local path
train = pd.read_csv('dane/playground-series-s5e6/train.csv')
test = pd.read_csv('dane/playground-series-s5e6/test.csv')

train = train.rename(columns={"Temparature":"Temperature", "Phosphorous":"Phosphorus"})
test = test.rename(columns={"Temparature":"Temperature", "Phosphorous":"Phosphorus"})

# Preserve original ID columns for later use
train_id = train['id'].copy()
test_id = test['id'].copy()

# Drop ID columns before doing any feature engineering or model training
train_data = train.drop(columns=['id'])
test_data = test.drop(columns=['id'])



from utils.check_duplicates import check_duplicates_report
from utils.print_missing import missing_values_report
from utils.print_uniques import print_unique_values

def run_diagnostics():
    # Dictionary of datasets to check
    datasets = {
        "Training Data": train_data,
        "Test Data": test_data
    }

    # Collect summary info for later use or logging
    duplicate_summary = {}
    for name, data in datasets.items():
        check_duplicates_report(data, name)
        duplicate_summary[name] = {
            'duplicates': data.duplicated().sum(),
            'total_rows': len(data)
        }
        print()
        print("\n" + "=" * 50)


    for name, data in datasets.items():
        missing_values_report(data, name)
        print()
        print("\n" + "=" * 50)

    # List of categorical columns including target for train, excluding target for test
    train_cat_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
    test_cat_cols = ['Soil Type', 'Crop Type']

    # Explore distribution of categories
    print_unique_values(train_data, train_cat_cols, "Train Data")
    print_unique_values(test_data, test_cat_cols, "Test Data")


    # Identify numeric columns, excluding explicitly known categorical features
    numeric_columns = [col for col in train_data.columns
                    if col not in ['Soil Type', 'Crop Type', 'Fertilizer Name']
                    and train_data[col].dtype in [np.int64, np.float64]]

    # Categorical columns are those not numeric and not the target
    categorical_columns = [col for col in train_data.columns 
                        if col not in numeric_columns + ['Fertilizer Name']]

    # Define target variable for supervised learning
    target_column = 'Fertilizer Name'

    # Basic shape overview
    print("\n" + "=" * 50)
    print("Training set shape:", train.shape)
    print("Test set shape:", test.shape)
    print("\n" + "=" * 50)

    # Display schema and data types of training set
    print("Train Data Info:")
    train.info()
    print("\n" + "=" * 50)

    # Display schema and data types of test set
    print("\nTest Data Info:")
    test.info()
    print("\n" + "=" * 50)

    # Summary statistics for all numerical columns in the training set
    print("\nStatistical summary of training data:")
    print(train.describe().T.round(2))
    print("\n" + "=" * 50)

    print(f"\nNumber of numeric features: {len(numeric_columns)}")
    print(f"Number of categorical features: {len(categorical_columns)}")
    print(f"Target column: {target_column}")
    print("\n" + "=" * 50)


    # List of numerical columns to visualize and analyze
    numerical_features = [
        "Temperature",    # Note: typo in original column name? Should be "Temperature"?
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorus", 
    ]

    # For each numeric feature, plot histogram and boxplot + print stats
    for feature in numerical_features:
        plt.figure(figsize=(12, 5))
        binsize = max(train[feature]) - min(train[feature]) + 1
        # Histogram + KDE to view distribution shape
        plt.subplot(1, 2, 1)
        sns.histplot(train[feature], kde=True, bins=binsize)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        # Boxplot to detect outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(x=train_data[feature])
        plt.title(f"Box Plot of {feature}")

        plt.tight_layout()
        plt.show()

        # Print skewness for the feature
        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {train_data[feature].skew():.2f}")  # Helps assess normality

    # Pie charts for categorical features: show category distribution visually
    for feature in ["Soil Type", "Crop Type"]:
        counts = train_data[feature].value_counts()

        # Pie chart to show class balance or imbalance
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f"Distribution of {feature}")
        plt.axis("equal")  # Equal aspect ratio to ensure pie is round
        plt.show()


    correlation_matrix = train_data[numerical_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

print("\n" + "=" * 50)
run_diag = input("Run diagnostics before training? (y/n): ").strip().lower()
print("\n" + "=" * 50)
# Run diagnostics if user confirms
if run_diag == 'y':
    run_diagnostics()
else:
    print("Skipping diagnostics...\n")


# Feature engineering: interactions and ratios for domain insight
train_data['Temp_humidity'] = train_data['Temperature'] * train_data['Humidity'] / 100    # Relative interaction
train_data['Temp_moisture'] = train_data['Temperature'] * train_data['Moisture'] / 100      # Cross feature
train_data['Humidity_ratio'] = train_data['Humidity'] / (train_data['Temperature'] + 1e-6)  # Avoid div-by-zero

# Aggregate nutrient content (useful for total fertility)
train_data['npk_total'] = train_data['Nitrogen'] + train_data['Phosphorus'] + train_data['Potassium']

# Nutrient ratios to capture relative abundance
train_data['n_to_p'] = train_data['Nitrogen'] / (train_data['Phosphorus'] + 1e-6)
train_data['k_to_p'] = train_data['Potassium'] / (train_data['Phosphorus'] + 1e-6)
train_data['n_to_k'] = train_data['Nitrogen'] / (train_data['Potassium'] + 1e-6)

# Row-wise nutrient variability (dispersion)
train_data['npk_std'] = train_data[['Nitrogen', 'Phosphorus', 'Potassium']].std(axis=1)

# Bucket nutrient levels into quartiles for categorical modeling
train_data['n_quartile'] = pd.qcut(train_data['Nitrogen'], 4, labels=["small", "medium", "large", "very large"])
train_data['k_quartile'] = pd.qcut(train_data['Potassium'], 4, labels=["small", "medium", "large", "very large"])
train_data['p_quartile'] = pd.qcut(train_data['Phosphorus'], 4, labels=["small", "medium", "large", "very large"])

# Combined soil & crop type — useful categorical combo for modeling
train_data['soil_crop'] = train_data['Soil Type'].astype(str) + "_" + train_data['Crop Type'].astype(str)

# Apply same feature engineering to test data
test_data['Temp_humidity'] = test_data['Temperature'] * test_data['Humidity'] / 100
test_data['Temp_moisture'] = test_data['Temperature'] * test_data['Moisture'] / 100
test_data['Humidity_ratio'] = test_data['Humidity'] / (test_data['Temperature'] + 1e-6)

test_data['npk_total'] = test_data['Nitrogen'] + test_data['Phosphorus'] + test_data['Potassium']
test_data['n_to_p'] = test_data['Nitrogen'] / (test_data['Phosphorus'] + 1e-6)
test_data['k_to_p'] = test_data['Potassium'] / (test_data['Phosphorus'] + 1e-6)
test_data['n_to_k'] = test_data['Nitrogen'] / (test_data['Potassium'] + 1e-6)
test_data['npk_std'] = test_data[['Nitrogen', 'Phosphorus', 'Potassium']].std(axis=1)

test_data['n_quartile'] = pd.qcut(test_data['Nitrogen'], 4, labels=["small", "medium", "large", "very large"])
test_data['k_quartile'] = pd.qcut(test_data['Potassium'], 4, labels=["small", "medium", "large", "very large"])
test_data['p_quartile'] = pd.qcut(test_data['Phosphorus'], 4, labels=["small", "medium", "large", "very large"])

test_data['soil_crop'] = test_data['Soil Type'].astype(str) + "_" + test_data['Crop Type'].astype(str)

# Prepare features and target
drop_cols = ['Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorus', 'Potassium']
X = train_data.drop(columns=drop_cols + ['Fertilizer Name'])  # Final feature set
y = train_data['Fertilizer Name']                              # Target labels

X_test = test_data.drop(columns=drop_cols)  # Test features, no target




# Encode string labels into numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Categorical features to inform the CatBoost model (handles them natively)
cat_features = ['soil_crop', 'n_quartile', 'k_quartile', 'p_quartile']

# Stratified train-validation split to maintain class distribution
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Custom Mean Average Precision at 3 implementation
def mapk(actual, predicted, k=3):
    return np.mean([
        1.0 if a in p[:k] else 0.0
        for a, p in zip(actual, predicted)
    ])

# Optuna hyperparameter optimization for CatBoost
def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.12),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4, 8),
        'random_strength': trial.suggest_float('random_strength', 0.75, 1.5),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 64, 255),
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'verbose': 0,
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0',  # Adjust based on your GPU setup
        'cat_features': cat_features
    }

    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    model = CatBoostClassifier(**params)
    model.fit(X_train_, y_train_, eval_set=(X_valid_, y_valid_), early_stopping_rounds=50)

    # Predict probabilities
    probs = model.predict_proba(X_valid_)
    top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]

    return mapk(y_valid_, top3, k=3)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best MAP@3:", study.best_value)
print("Best Params:", study.best_params)

# Train final model
best_model = CatBoostClassifier(
    **study.best_params,
    iterations=1000,
    loss_function='MultiClass',
    eval_metric='TotalF1',
    cat_features=cat_features,
    random_seed=42,
    verbose=100
)
best_model.fit(X, y_encoded)

# Predict probabilities for all classes
probs = best_model.predict_proba(X_val)

# Get top 3 predicted class indices per sample
top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]

# Evaluate MAP@3 performance
map3_score = mapk(y_val, top3, k=3)
print(f"Validation MAP@3: {map3_score:.4f}")

# Visualize feature importances
#feature_importances = best_model.get_feature_importance(prettified=True)
#feature_importances.plot(kind='barh', x='Feature Id', y='Importances', figsize=(10, 6))
#plt.title("Feature Importances")
#plt.show()

# Predict on test set
probs_test = best_model.predict_proba(X_test)
top3_test = np.argsort(probs_test, axis=1)[:, ::-1][:, :3]
top3_labels = np.array([le.inverse_transform(row) for row in top3_test])

# Prepare submission DataFrame
submission = pd.DataFrame({
    'Id': test_id,
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})


# Save submission to CSV
submission.to_csv('submission.csv', index=False)