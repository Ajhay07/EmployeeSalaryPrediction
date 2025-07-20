# ---------------------------------------------
# 1. Import Libraries
# ---------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ---------------------------------------------
# 2. Load CSV
# ---------------------------------------------
data = pd.read_csv(r'C:\Users\Ajhay\adult 3.csv')
print("Data Loaded Successfully!")
print(data.head())

# ---------------------------------------------
# 3. Clean Data
# ---------------------------------------------
data = data.replace('?', pd.NA)
data = data.dropna()
print(f"Cleaned Data Shape: {data.shape}")

# ---------------------------------------------
# 4. Split Features and Target with Encoding
# ---------------------------------------------
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 0 if x == '<=50K' else 1)
print(f"Target classes encoded: {y.unique()}")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"Categorical Columns: {categorical_cols}")
print(f"Numeric Columns: {numeric_cols}")

# ---------------------------------------------
# 5. Preprocessing Pipeline
# ---------------------------------------------
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

# ---------------------------------------------
# 6. Train/Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train Set: {X_train.shape}, Test Set: {X_test.shape}")

# ---------------------------------------------
# 7. Define Model Pipeline with XGBoost
# ---------------------------------------------
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
])

# ---------------------------------------------
# 8. Hyperparameter Tuning with GridSearch
# ---------------------------------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2
)

print("Starting Model Training with GridSearch...")
grid_search.fit(X_train, y_train)
print("Model Training Complete!")

print("\nBest Parameters Found:")
print(grid_search.best_params_)

# ---------------------------------------------
# 9. Evaluate on Test Data
# ---------------------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ---------------------------------------------
# 10. Confusion Matrix Visualization
# ---------------------------------------------
disp = ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    display_labels=['<=50K', '>50K'],
    cmap=plt.cm.Blues,
    values_format='d'
)
plt.title("Confusion Matrix")
plt.show()

# ---------------------------------------------
# 11. Feature Importance Visualization
# ---------------------------------------------
# Extract feature names
onehot_encoder = best_model.named_steps['preprocessor'].transformers_[0][1]
onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
all_feature_names = list(onehot_feature_names) + numeric_cols

# Extract importances
importances = best_model.named_steps['classifier'].feature_importances_

# Plot top 20 features
sorted_idx = np.argsort(importances)[-20:]
plt.figure(figsize=(10,6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), [all_feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top 20 Important Features (XGBoost)")
plt.show()

# ---------------------------------------------
# 12. Predict Income for New Employee Example
# ---------------------------------------------
sample_employee = pd.DataFrame({
    'age': [35],
    'workclass': ['Private'],
    'fnlwgt': [200000],
    'education': ['Bachelors'],
    'educational-num': [13],
    'marital-status': ['Married-civ-spouse'],
    'occupation': ['Exec-managerial'],
    'relationship': ['Husband'],
    'race': ['White'],
    'gender': ['Male'],
    'capital-gain': [0],
    'capital-loss': [0],
    'hours-per-week': [45],
    'native-country': ['United-States']
})

predicted_income_num = best_model.predict(sample_employee)[0]
label_map = {0: "<=50K", 1: ">50K"}
print(f"\nPredicted Income for Sample Employee: {label_map[predicted_income_num]}")
