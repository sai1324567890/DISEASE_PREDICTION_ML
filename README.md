# Disease Prediction Using Machine Learning

Multi-model predictive system (**Logistic Regression, Decision Tree, SVM**) with imputation, scaling,
**feature selection (SelectKBest)**, **GridSearch hyper-parameter tuning** and **5-fold stratified
cross-validation**. The best model is saved and reused for predictions.

Runs out-of-the-box on the built-in Breast Cancer Wisconsin dataset, or on any CSV.

## Setup & run
```bash
pip install -r requirements.txt
python disease_prediction.py train
python disease_prediction.py predict --sample 3

# your own data (numeric features + a target column)
python disease_prediction.py train --csv heart.csv --target target
python disease_prediction.py predict --csv new_patients.csv
```
Outputs in `outputs/`: `best_model.joblib`, `meta.json` (all metrics), `model_comparison.png`, `confusion_matrix.png`.

> Educational project - not a medical device.
