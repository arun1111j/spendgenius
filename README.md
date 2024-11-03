## spendgenius
# ML code
```

# improved version

import time

start_time = time.time()
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np
import shap

# Load the dataset
data = pd.read_csv('/content/Finance_data.csv')

# Checking for any missing values
print("Missing values in each column:\n", data.isnull().sum())

# Filling missing values (if any)
data.fillna(method='ffill', inplace=True)

# Encoding categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separating features and target
X = data.drop(columns=['Avenue'])  # Features
y = data['Avenue']                 # Target

# Display class distribution before SMOTE
print("Class distribution before SMOTE:\n", y.value_counts())

# Applying SMOTE to balance the classes
smote = SMOTE(random_state=42, k_neighbors=1)
X_res, y_res = smote.fit_resample(X, y)

# Display class distribution after SMOTE
print("Class distribution after SMOTE:\n", y_res.value_counts())

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define important feature indices based on your analysis
important_feature_indices = [0, 12, 16, 21, 6, 13, 8, 4, 2, 3, 5, 7]  # Adjusted according to feature importance
X_train_top = X_train[:, important_feature_indices]
X_test_top = X_test[:, important_feature_indices]

# Model selection: Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_top, y_train)

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X_train_top, y_train, cv=5, scoring='accuracy')
print(f"Random Forest Cross-Validation Accuracy: {np.mean(rf_cv_scores):.4f}")

# Making predictions
y_pred_rf = rf_model.predict(X_test_top)

# Evaluating the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Model selection: Gradient Boosting with hyperparameter tuning
gb_model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_top, y_train)

# Best model from GridSearchCV
best_gb_model = grid_search.best_estimator_

# Cross-validation for Gradient Boosting
gb_cv_scores = cross_val_score(best_gb_model, X_train_top, y_train, cv=5, scoring='accuracy')
print(f"Gradient Boosting Cross-Validation Accuracy: {np.mean(gb_cv_scores):.4f}")

# Making predictions and evaluating
y_pred_gb = best_gb_model.predict(X_test_top)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nGradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))

# Hyperparameter tuning for XGBoost using GridSearchCV
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), 
                                xgb_param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
xgb_grid_search.fit(X_train_top, y_train)

# Best model from GridSearchCV
best_xgb_model = xgb_grid_search.best_estimator_

# Cross-validation for the best XGBoost model
best_xgb_cv_scores = cross_val_score(best_xgb_model, X_train_top, y_train, cv=5, scoring='accuracy')
print(f"Best XGBoost Cross-Validation Accuracy: {np.mean(best_xgb_cv_scores):.4f}")

# Making predictions and evaluating with the best model
y_pred_best_xgb = best_xgb_model.predict(X_test_top)
print("Best XGBoost Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print("\nBest XGBoost Classification Report:\n", classification_report(y_test, y_pred_best_xgb))

# SHAP analysis for feature importance
explainer = shap.Explainer(best_xgb_model)
shap_values = explainer(X_test_top)

# Plotting SHAP values
shap.summary_plot(shap_values, X_test_top)

end_time = time.time()
print("Processing time:", end_time - start_time, "seconds")
```
