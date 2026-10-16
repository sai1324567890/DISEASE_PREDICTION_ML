#!/usr/bin/env python3
"""
Train and evaluate models for disease prediction using the breast cancer dataset or a custom CSV file.

"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

OUT = Path("outputs")
MODEL_PATH = OUT / "best_model.joblib"
META_PATH = OUT / "meta.json"
RANDOM_STATE = 42


def load_data(csv=None, target=None):
    if csv:
        df = pd.read_csv(csv)
        if target not in df.columns:
            sys.exit(f"Target column '{target}' not found. Columns: {list(df.columns)}")
        y = df[target]
        X = df.drop(columns=[target]).select_dtypes(include=[np.number])
        if y.dtype == object:
            y = pd.factorize(y)[0]
        return X, pd.Series(y), "custom"
    ds = load_breast_cancer(as_frame=True)
    return ds.data, ds.target, "breast_cancer"


def build_models(n_features):
    k_grid = sorted({max(2, n_features // 2), n_features})

    def pipe(clf):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            ("clf", clf),
        ])

    return {
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=2000)),
            {"select__k": k_grid, "clf__C": [0.1, 1, 10]}),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            {"select__k": k_grid, "clf__max_depth": [3, 5, 8]}),
        "SVM": (
            pipe(SVC(probability=True, random_state=RANDOM_STATE)),
            {"select__k": k_grid, "clf__C": [0.5, 1, 5],
             "clf__kernel": ["rbf", "linear"]}),
    }


def train(args):
    OUT.mkdir(exist_ok=True)
    X, y, name = load_data(args.csv, args.target)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for model_name, (pipe, grid) in build_models(X.shape[1]).items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_acc = cross_val_score(best, X_tr, y_tr, cv=cv, scoring="accuracy")
        pred = best.predict(X_te)
        proba = best.predict_proba(X_te)
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y_te, proba[:, 1])
        else:
            auc = roc_auc_score(y_te, proba, multi_class="ovr")
        results[model_name] = {
            "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
            "cv_accuracy_std": round(float(cv_acc.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "test_f1_weighted": round(float(f1_score(y_te, pred, average="weighted")), 4),
            "test_roc_auc": round(float(auc), 4),
            "best_params": search.best_params_,
        }
        fitted[model_name] = (best, pred)
        print(f"\n=== {model_name} ===")
        print(json.dumps(results[model_name], indent=2, default=str))
        print(classification_report(y_te, pred))

    best_name = max(results, key=lambda m: results[m]["test_f1_weighted"])
    best_model, best_pred = fitted[best_name]
    joblib.dump(best_model, MODEL_PATH)
    META_PATH.write_text(json.dumps(
        {"dataset": name, "features": list(X.columns), "best_model": best_name,
         "results": results}, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(results.keys()), [r["test_accuracy"] for r in results.values()],
           color="#2b6cb0")
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, best_pred, ax=ax, cmap="Blues")
    ax.set_title(best_name)
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\nBest model: {best_name}  ->  saved to {MODEL_PATH}")


def predict(args):
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run: python disease_prediction.py train")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    if args.csv:
        X = pd.read_csv(args.csv)[meta["features"]]
    else:
        X, _, _ = load_data()
        X = X.iloc[[args.sample]]
    preds, probs = model.predict(X), model.predict_proba(X)
    for i, (p, pr) in enumerate(zip(preds, probs)):
        print(f"Row {i}: class={p}  confidence={pr.max():.2%}")
    if meta["dataset"] == "breast_cancer":
        print("(Built-in dataset: 0 = malignant, 1 = benign)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--csv")
    t.add_argument("--target")
    t.set_defaults(fn=train)
    p = sub.add_parser("predict")
    p.add_argument("--csv")
    p.add_argument("--sample", type=int, default=0)
    p.set_defaults(fn=predict)
    a = ap.parse_args()
    if a.cmd == "train" and a.csv and not a.target:
        ap.error("--target is required with --csv")
    a.fn(a)
