# 🧠 Customer Churn Prediction (End-to-End ML Project)

---

## 📌 Project Overview

This project predicts customer churn (whether a customer will leave the bank) using machine learning models. The objective is to help businesses identify high-risk customers and reduce churn.

---

## 📊 Dataset Description

* Total samples: **10,000 customers**
* Target distribution:

  * Stayed: **79.63%**
  * Churned: **20.37%** (Imbalanced dataset)

### Features:

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

### Target:

* **Exited (0 = Stay, 1 = Churn)**

---

## 🧹 Data Preprocessing

* Removed irrelevant columns:

  * RowNumber
  * CustomerId
  * Surname
* No missing values detected
* Applied **One-Hot Encoding** for categorical variables
* Train/Test split: **80/20**

---

## 📊 Exploratory Data Analysis (EDA)

### Key Insights:

* 📈 **Age** has the strongest correlation with churn (**0.285**)
* 💤 **Inactive members** are more likely to churn
* 💰 Higher **balance → higher churn probability**
* 🌍 **Germany customers churn significantly more (~32%)**

### Statistical Tests:

* Chi-Square (Geography): **p < 0.001**
* ANOVA (Age, Balance): highly significant
* Point-Biserial correlation used for feature importance

---

## ⚙️ Model Building

---

### 🔥 Model 1: XGBoost (Primary Model)

#### Handling Imbalance:

* `scale_pos_weight = 3.91`

#### Best Hyperparameters (RandomizedSearchCV):

```python
{
 'subsample': 0.9,
 'n_estimators': 500,
 'min_child_weight': 3,
 'max_depth': 6,
 'learning_rate': 0.01,
 'gamma': 0,
 'colsample_bytree': 0.8
}
```

---

### 📊 XGBoost Performance:

* **ROC-AUC:** 0.864
* **Accuracy:** 0.81
* **F1 Score (Churn class):** 0.62

#### Classification Report:

* Precision (churn): 0.53
* Recall (churn): 0.74
  👉 Model is good at catching churners (high recall)

---

## 🌲 Model 2: Random Forest

#### Best Hyperparameters:

```python
{
 'n_estimators': 400,
 'min_samples_split': 10,
 'min_samples_leaf': 4,
 'max_features': 'log2',
 'max_depth': 15
}
```

---

### 📊 Random Forest Performance:

* **Accuracy:** 0.85
* **F1 Score (Churn class):** 0.63

#### Overfitting Check:

* Train F1: **0.84**
* Test F1: **0.63** ❌ (clear overfitting)

---

## ⚖️ Model Comparison

| Metric         | XGBoost   | Random Forest   |
| -------------- | --------- | --------------- |
| Accuracy       | 0.81      | **0.85**        |
| ROC-AUC        | **0.864** | ❌ Not optimized |
| F1 (Churn)     | 0.62      | **0.63**        |
| Overfitting    | Low ✅     | High ❌          |
| Recall (Churn) | **0.74**  | 0.62            |

---

## 🏆 Final Verdict

* **XGBoost is the better model overall**

  * Better generalization
  * Better ROC-AUC
  * Better at detecting churn (higher recall)

* **Random Forest**

  * Slightly better F1
  * But suffers from overfitting

👉 In real business scenarios: **XGBoost is preferred**

---

## 📉 Feature Importance (Top Drivers)

1. NumOfProducts
2. Age
3. IsActiveMember
4. Geography (Germany)
5. Balance

---

## 🎯 Key Takeaways

* Customer activity is a critical factor in retention
* German customers have significantly higher churn rates
* Imbalanced data must be handled carefully
* Model selection depends on business goal:

  * Recall → XGBoost
  * Accuracy → Random Forest

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SciPy
* Matplotlib, Seaborn

---

## 🚀 How to Run

```bash
git clone https://github.com/USERNAME/Churn-Prediction.git
cd Churn-Prediction
pip install -r requirements.txt
jupyter notebook
```

---

## 🔮 Future Improvements

* Deploy model using FastAPI
* Build interactive dashboard (Streamlit)
* Apply advanced feature engineering
* Try ensemble stacking

---

## 👨‍💻 Author

Mohamed Ahmed
