# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 04:04:13 2026

@author: MSI
"""

# train.py
import os
import json
import joblib
from palmerpenguins import load_penguins

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

def build_preprocess(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocess

def main():
    os.makedirs("models", exist_ok=True)

    df = load_penguins()
    target = "species"
    feature_cols = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    df = df[feature_cols + [target]].copy()

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    categorical_features = ["island", "sex"]

    preprocess = build_preprocess(numeric_features, categorical_features)

    candidates = {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    }

    scores = {}
    saved_paths = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        scores[name] = acc

        path = os.path.join("models", f"{name}.joblib")
        joblib.dump(pipe, path)
        saved_paths[name] = path

        print(f"Modelo: {name} | accuracy: {acc:.4f} | guardado en: {path}")

    best_model = max(scores, key=scores.get)
    meta = {
        "best_model": best_model,
        "scores": scores,
        "paths": saved_paths,
        "active_model": best_model,
        "features": feature_cols,
        "target": target
    }

    with open(os.path.join("models", "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nMeta guardada en models/meta.json")
    print(f"Modelo activo inicial: {best_model}")

if __name__ == "__main__":
    main()
