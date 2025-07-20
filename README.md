# 🧠 Employee Salary Prediction using XGBoost

This project aims to predict whether a person earns **more than $50K** or **less than or equal to $50K** annually using a range of demographic and employment-related features from the **UCI Adult dataset**. The machine learning pipeline is built using **XGBoost**, and enhanced with hyperparameter tuning using **GridSearchCV**.

📍 **GitHub Repository:** [Ajhay07/EmployeeSalaryPrediction](https://github.com/Ajhay07/EmployeeSalaryPrediction)

---

## 📁 Project Files

| File Name        | Description                                      |
|------------------|--------------------------------------------------|
| `FinalESP.py`    | Python script containing full ML pipeline, model training, evaluation, and prediction |
| `adult 3.csv`    | Cleaned version of the UCI Adult dataset         |
| `README.md`      | Project documentation (this file)                |

---

## 🛠️ Technologies & Libraries Used

- **Python 3.x**
- **pandas** – Data manipulation
- **numpy** – Numerical computing
- **matplotlib** – Visualizations
- **scikit-learn** – Preprocessing, model selection, evaluation
- **xgboost** – High-performance gradient boosting classifier

---

## 🎯 Project Objective

- Classify whether a person's income is `<=50K` or `>50K` using structured data
- Build a clean, modular ML pipeline using `Pipeline` and `ColumnTransformer`
- Optimize the model using **GridSearchCV**
- Evaluate using **accuracy**, **classification report**, and **confusion matrix**
- Visualize feature importances to interpret the model

---

## 🔢 Features Used

- `age`
- `workclass`
- `fnlwgt`
- `education`
- `educational-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `gender`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

---

## ⚙️ How to Run the Project

1. **Clone this repository**:
   ```bash
   git clone https://github.com/Ajhay07/EmployeeSalaryPrediction.git
   cd EmployeeSalaryPrediction
