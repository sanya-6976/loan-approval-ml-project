# 🏦 Loan Approval Prediction — End-to-End Supervised ML Pipeline

An interview-grade, leakage-free machine learning pipeline to predict loan approval status using tree-based models, ensemble learning, domain feature engineering, and business threshold optimization.

## 🎯 Key Highlights & Engineering Decisions
- **Zero Data Leakage Architecture:** Executed stratified train-test split (`80/20`) prior to preprocessing, encapsulating median imputation and one-hot encoding inside a Scikit-Learn `Pipeline` and `ColumnTransformer`.
- **Overfitting Diagnosis & Ensemble Progression:** Benchmarked an unpruned Decision Tree (`1.00` Train vs `0.715` Test Acc) against pruned trees, Random Forest, and Gradient Boosting—eliminating variance and achieving **0.8307 ROC-AUC** and **0.9130 F1-Score**.
- **Domain Feature Engineering:** Engineered household `TotalIncome` (log-transformed) and `Loan_to_Income` debt-burden ratio, which emerged as the **#2 and #3 most predictive features** right after `Credit_History`.
- **Hyperparameter & Risk Threshold Tuning:** Optimized tree depth and leaf constraints via 5-Fold Stratified `GridSearchCV`, and demonstrated precision-recall trade-off analysis (`0.50` vs `0.60` probability cutoff) to minimize costly False Positive bad loans.

## 📊 Model Progression Summary
| Model | Train Acc | Test Acc | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Unpruned Decision Tree (Baseline) | 1.0000 | 0.7154 | — | — |
| Pruned Decision Tree (`max_depth=3`) | 0.8086 | 0.8455 | 0.8984 | 0.7375 |
| Gradient Boosting (`lr=0.05`) | 0.8411 | 0.8211 | 0.8804 | 0.7862 |
| **Tuned Random Forest + Feature Eng.** | **0.8106** | **0.8537** | **0.9130** | **0.8307** |

## 🛠️ Tech Stack
`Python` • `Pandas` • `NumPy` • `Scikit-Learn` • `Matplotlib` • `Seaborn` • `Google Colab`
