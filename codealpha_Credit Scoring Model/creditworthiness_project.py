"""
Creditworthiness classification project
Author: Generated for internship submission
Dataset: UCI - Default of Credit Card Clients (or replace with your dataset)

How to run:
1. Install dependencies: pip install -r requirements.txt
   Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, xlrd, openpyxl
2. Run: python src/creditworthiness_project.py

This script downloads the dataset (if not present), preprocesses data, does feature engineering,
trains Logistic Regression, Decision Tree, Random Forest, evaluates them (Precision, Recall, F1, ROC-AUC),
and saves the best model to 'models/best_model.pkl'.

Note: If your internship provides a different dataset, update the DATA_PATH variable.
"""

import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# CONFIG: paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# make sure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
DATA_FILENAME = "default_of_credit_card_clients.xls"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_OUTPUT = os.path.join(MODELS_DIR, "best_model.pkl")

# -----------------------
# UTIL: download dataset if needed
# -----------------------
def download_dataset(url, filename):
    if os.path.exists(filename):
        print(f"Dataset already exists at {filename}")
        return
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filename)
    print("Download completed.")

# -----------------------
# LOAD DATA (auto detect)
# -----------------------
def load_data(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".xls":
        df = pd.read_excel(path, header=1, engine="xlrd")
    elif ext == ".xlsx":
        df = pd.read_excel(path, header=1, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Rename target column to 'default'
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})
    elif 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default'})
    return df

# -----------------------
# FEATURE ENGINEERING
# -----------------------
def feature_engineering(df):
    df = df.copy()
    bill_cols = [c for c in df.columns if c.startswith('BILL_AMT')]
    pay_cols = [c for c in df.columns if c.startswith('PAY_AMT')]

    df['total_bill_amt_6m'] = df[bill_cols].sum(axis=1)
    df['total_pay_amt_6m'] = df[pay_cols].sum(axis=1)
    df['debt_to_limit'] = df['total_bill_amt_6m'] / (df['LIMIT_BAL'] + 1e-6)
    df['pay_ratio'] = df['total_pay_amt_6m'] / (df['total_bill_amt_6m'].replace(0, np.nan))
    df['pay_ratio'] = df['pay_ratio'].fillna(0)
    df['avg_bill'] = df[bill_cols].mean(axis=1)
    df['avg_pay'] = df[pay_cols].mean(axis=1)
    df['num_delays'] = (df[[c for c in df.columns if c.startswith('PAY_')]].fillna(0) > 0).sum(axis=1)

    df['debt_to_limit'] = df['debt_to_limit'].clip(0, 10)
    df['pay_ratio'] = df['pay_ratio'].clip(0, 10)

    return df

# -----------------------
# PREPARE PIPELINE
# -----------------------
def build_preprocessing_pipeline(df):
    target = 'default'
    if target in df.columns:
        features = df.drop(columns=[target]).columns.tolist()
    else:
        features = df.columns.tolist()

    numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features

# -----------------------
# TRAIN & EVALUATE
# -----------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    print(f"\n===== Evaluation: {model_name} =====")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred, y_proba

# -----------------------
# SAVE FEATURE IMPORTANCES
# -----------------------
def save_feature_importances(best_model, best_model_name, num_feats, cat_feats):
    try:
        ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(cat_feats)
        feature_names = np.concatenate([num_feats, cat_feature_names])
    except Exception:
        feature_names = num_feats + cat_feats

    if best_model_name in ['DecisionTree', 'RandomForest']:
        values = best_model.named_steps['classifier'].feature_importances_
    elif best_model_name == 'LogisticRegression':
        values = np.abs(best_model.named_steps['classifier'].coef_[0])
    else:
        print("No feature importances available for this model.")
        return

    fi = pd.Series(values, index=feature_names).sort_values(ascending=False).head(30)

    print("\nTop features:")
    print(fi)

    # Save as PNG
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title(f"Top features ({best_model_name})")
    plt.tight_layout()
    fi_path = os.path.join(OUTPUTS_DIR, "feature_importances.png")
    plt.savefig(fi_path)
    print(f"Saved feature importance plot to {fi_path}")

    # Save as CSV
    csv_path = os.path.join(OUTPUTS_DIR, "feature_importances.csv")
    fi.to_csv(csv_path, header=["importance"])
    print(f"Saved feature importance table to {csv_path}")

# -----------------------
# MAIN
# -----------------------
def main():
    if not os.path.exists(DATA_PATH):
        download_dataset(DATA_URL, DATA_PATH)

    df = load_data(DATA_PATH)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist()[:20])

    df_fe = feature_engineering(df)
    target = 'default'
    if target not in df_fe.columns:
        raise ValueError("Target column 'default' not found in dataset.")

    X = df_fe.drop(columns=[target])
    y = df_fe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor, num_feats, cat_feats = build_preprocessing_pipeline(pd.concat([X_train, y_train], axis=1))

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    }

    trained_models, results = {}, {}
    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        _, y_proba = evaluate_model(pipe, X_test, y_test, model_name=name)
        trained_models[name] = pipe
        results[name] = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    if any(v is not None for v in results.values()):
        best_model_name = max(results.items(), key=lambda kv: (kv[1] if kv[1] else -1))[0]
    else:
        accs = {name: trained_models[name].score(X_test, y_test) for name in trained_models}
        best_model_name = max(accs.items(), key=lambda kv: kv[1])[0]

    print(f"\nBest model selected: {best_model_name}")
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODEL_OUTPUT)
    print(f"Saved best model to {MODEL_OUTPUT}")

    # Save feature importances
    save_feature_importances(best_model, best_model_name, num_feats, cat_feats)

    # Save ROC curve
    try:
        y_proba_best = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba_best)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{best_model_name} (AUC={roc_auc_score(y_test, y_proba_best):.4f})')
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
        plt.savefig(roc_path)
        print(f"Saved ROC curve to {roc_path}")
    except Exception as e:
        print("Could not plot ROC curve:", e)

if __name__ == '__main__':
    main()
