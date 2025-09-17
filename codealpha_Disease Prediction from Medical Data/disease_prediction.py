# src/disease_prediction.py
"""
Disease prediction pipeline
- Datasets: breast_cancer (sklearn), diabetes (Pima), heart (heart disease)
- Models: Logistic Regression, SVM, RandomForest, XGBoost
- Saves predictions, results, confusion matrices, ROC curves, combined predictions,
  and stores the best performing model as models/best_model.pkl
Dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, openpyxl
"""

import os
import sys
import pickle
import warnings
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Optional: try import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ---------------------------
# Configuration / Constants
# ---------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

DATASETS = ["breast_cancer", "diabetes", "heart"]

DEFAULT_SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": RANDOM_SEED}
DEFAULT_RF_PARAMS = {"n_estimators": 100, "random_state": RANDOM_SEED}
DEFAULT_LOGREG_PARAMS = {"solver": "liblinear", "random_state": RANDOM_SEED}
DEFAULT_XGB_PARAMS = {"use_label_encoder": False, "eval_metric": "logloss", "random_state": RANDOM_SEED} if XGB_AVAILABLE else {}

# ---------------------------
# Utilities
# ---------------------------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved model: {path}")

def load_csv_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def try_download_file(url: str, dest: str) -> bool:
    try:
        from urllib.request import urlretrieve
        print(f"Attempting to download {url} to {dest} ...")
        urlretrieve(url, dest)
        print("Download succeeded.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# ---------------------------
# Data loading
# ---------------------------
def load_dataset(name: str) -> pd.DataFrame:
    name = name.lower()
    local_path = os.path.join(DATA_DIR, f"{name}.csv")
    df = load_csv_if_exists(local_path)
    if df is not None:
        print(f"Loaded {name} from {local_path} (shape={df.shape})")
        return df

    if name == "breast_cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        df.to_csv(local_path, index=False)
        print(f"Saved breast_cancer to {local_path}")
        return df

    if name == "diabetes":
        urls = [
            "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
            "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv",
        ]
    elif name == "heart":
        urls = [
            "https://raw.githubusercontent.com/ansh941/Machine-Learning-Projects/master/Heart%20Disease/heart.csv",
            "https://raw.githubusercontent.com/nareshpro/Heart-Disease-Analysis/master/heart.csv",
        ]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    for url in urls:
        if try_download_file(url, local_path):
            df = pd.read_csv(local_path)
            print(f"Downloaded and loaded {name} dataset (shape={df.shape})")
            return df

    raise FileNotFoundError(
        f"Dataset {name} not found at {local_path} and automatic download failed.\n"
        "Please place the CSV in the data/ directory with filename "
        f"'{name}.csv' and re-run."
    )

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_data(df: pd.DataFrame, dataset_name: str = "", label_col: str = "target") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df_proc = df.copy()

    # Drop duplicates
    df_proc = df_proc.drop_duplicates()

    # Handle label column alternatives
    if label_col not in df_proc.columns:
        alternatives = ["Outcome", "target", "Class", "diagnosis", "DEATH_EVENT", "y"]
        for alt in alternatives:
            if alt in df_proc.columns:
                df_proc = df_proc.rename(columns={alt: "target"})
                print(f"Renamed label column {alt} -> target")
                break

    # Dataset-specific preprocessing
    if dataset_name.lower() == "heart":
        # Replace '?' with NaN
        df_proc.replace('?', np.nan, inplace=True)
        # Convert all columns to numeric
        for col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
        # Fill missing numeric values
        df_proc.fillna(df_proc.median(), inplace=True)
        # Map target >0 to 1
        if 'target' in df_proc.columns:
            df_proc['target'] = df_proc['target'].apply(lambda x: 1 if x > 0 else 0)

    # Drop non-numeric columns except target
    non_numeric = df_proc.select_dtypes(include=["object", "category"]).columns.tolist()
    non_numeric = [c for c in non_numeric if c != "target"]
    if non_numeric:
        print(f"Warning: dropping non-numeric columns: {non_numeric}")
        df_proc = df_proc.drop(columns=non_numeric)

    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")
    if not numeric_cols:
        raise ValueError("No numeric feature columns found after preprocessing.")

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    df_proc[numeric_cols] = imputer.fit_transform(df_proc[numeric_cols])

    # Feature scaling
    scaler = StandardScaler()
    df_proc[numeric_cols] = scaler.fit_transform(df_proc[numeric_cols])

    # Separate X and y
    if "target" not in df_proc.columns:
        raise ValueError("No 'target' label column found. Please ensure dataset contains labels with header 'target' or one of common alternatives.")

    X = df_proc[numeric_cols].values
    y = df_proc["target"].values.astype(int)
    return X, y, df_proc

# ---------------------------
# Plotting helpers
# ---------------------------
def plot_and_save_confusion_matrix(cm: np.ndarray, labels: list, out_path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix: {out_path}")

def plot_and_save_roc(y_true: np.ndarray, y_proba: np.ndarray, out_path: str, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC = {auc:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve: {out_path}")

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, df_test_original: pd.DataFrame, dataset_name: str, use_hyperparam_tuning: bool = False) -> Dict:
    results = {}
    models = {}

    # Logistic Regression
    logreg = LogisticRegression(**DEFAULT_LOGREG_PARAMS)
    logreg.fit(X_train, y_train)
    models["LogisticRegression"] = logreg

    # SVM
    svm = SVC(**DEFAULT_SVM_PARAMS)
    if use_hyperparam_tuning:
        param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
        grid = GridSearchCV(SVC(probability=True, random_state=RANDOM_SEED), param_grid, cv=3, scoring="roc_auc")
        grid.fit(X_train, y_train)
        svm = grid.best_estimator_
        print("SVM best params:", grid.best_params_)
    else:
        svm.fit(X_train, y_train)
    models["SVM"] = svm

    # Random Forest
    rf = RandomForestClassifier(**DEFAULT_RF_PARAMS)
    if use_hyperparam_tuning:
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 6, 12]}
        grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_SEED), param_grid, cv=3, scoring="roc_auc")
        grid.fit(X_train, y_train)
        rf = grid.best_estimator_
        print("RF best params:", grid.best_params_)
    else:
        rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # XGBoost
    if XGB_AVAILABLE:
        xg = xgb.XGBClassifier(**DEFAULT_XGB_PARAMS)
        if use_hyperparam_tuning:
            param_grid = {"n_estimators": [100, 200], "max_depth": [3, 6]}
            grid = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_SEED), param_grid, cv=3, scoring="roc_auc")
            grid.fit(X_train, y_train)
            xg = grid.best_estimator_
            print("XGB best params:", grid.best_params_)
        else:
            xg.fit(X_train, y_train)
        models["XGBoost"] = xg
    else:
        print("XGBoost not installed — skipping XGBoost.")

    # Evaluate models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_score = model.decision_function(X_test)
            y_proba = (df_score - df_score.min()) / (df_score.max() - df_score.min() + 1e-9)
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, digits=4)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "model_obj": model,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

        print(f"{name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print("Classification report:\n", class_report)

    # Save outputs (confusion, ROC, predictions, best model)
    def model_score_key(r):
        val = r.get("roc_auc", np.nan)
        if np.isnan(val):
            val = r.get("f1", 0.0)
        return val

    best_model_name = max(results.keys(), key=lambda k: model_score_key(results[k]))
    best_result = results[best_model_name]
    print(f"\nBest model for dataset '{dataset_name}': {best_model_name}")

    cm_out = os.path.join(OUTPUTS_DIR, f"{dataset_name}_confusion_matrix.png")
    plot_and_save_confusion_matrix(best_result["confusion_matrix"], labels=["0", "1"], out_path=cm_out, title=f"{dataset_name} - {best_model_name} Confusion Matrix")

    roc_out = os.path.join(OUTPUTS_DIR, f"{dataset_name}_roc_curve.png")
    plot_and_save_roc(y_test, best_result["y_proba"], out_path=roc_out, title=f"{dataset_name} - {best_model_name} ROC Curve")

    # Results summary Excel
    metrics_rows = []
    for mname, res in results.items():
        metrics_rows.append({
            "model": mname,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "roc_auc": res["roc_auc"]
        })
    results_df = pd.DataFrame(metrics_rows).sort_values(by="roc_auc", ascending=False)
    results_out = os.path.join(OUTPUTS_DIR, f"{dataset_name}_results.xlsx")
    results_df.to_excel(results_out, index=False)
    print(f"Saved results summary: {results_out}")

    # Predictions Excel
    try:
        df_preds = df_test_original.copy().reset_index(drop=True)
        df_preds[f"{best_model_name}_pred"] = best_result["y_pred"]
        df_preds[f"{best_model_name}_prob"] = best_result["y_proba"]
    except Exception as e:
        print(f"Warning: couldn't attach predictions to original df_test: {e}")
        df_preds = pd.DataFrame({
            "y_true": y_test,
            f"{best_model_name}_pred": best_result["y_pred"],
            f"{best_model_name}_prob": best_result["y_proba"]
        })
    preds_out = os.path.join(OUTPUTS_DIR, f"{dataset_name}_predictions.xlsx")
    df_preds.to_excel(preds_out, index=False)
    print(f"Saved predictions: {preds_out}")

    # Save best model
    per_dataset_model_path = os.path.join(MODELS_DIR, f"{dataset_name}_best_model.pkl")
    save_pickle(best_result["model_obj"], per_dataset_model_path)

    summary = {
        "dataset": dataset_name,
        "best_model_name": best_model_name,
        "best_model_obj": best_result["model_obj"],
        "best_metrics": {
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "roc_auc": best_result["roc_auc"]
        },
        "results_df": results_df,
        "predictions_df": df_preds
    }
    return summary

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(use_hyperparam_tuning: bool = False):
    combined_predictions = []
    comparison_records = []
    overall_best = {"score": -np.inf, "dataset": None, "model": None}

    for ds in DATASETS:
        print("\n" + "="*80)
        print(f"Processing dataset: {ds}")
        try:
            df_raw = load_dataset(ds)
        except Exception as e:
            print(f"Error loading dataset '{ds}': {e}")
            continue

        try:
            X, y, df_processed = preprocess_data(df_raw, dataset_name=ds)
        except Exception as e:
            print(f"Error in preprocessing dataset '{ds}': {e}")
            continue

        X_train, X_test, y_train, y_test, df_train_proc, df_test_proc = train_test_split(
            X, y, df_processed, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y if len(np.unique(y)) > 1 else None
        )

        try:
            summary = train_and_evaluate(
                X_train, X_test, y_train, y_test,
                df_test_proc, dataset_name=ds, use_hyperparam_tuning=use_hyperparam_tuning
            )
        except Exception as e:
            print(f"Error training/evaluating on dataset '{ds}': {e}")
            continue

        preds_df = summary["predictions_df"].copy()
        preds_df["dataset"] = ds
        combined_predictions.append(preds_df)

        best_metrics = summary["best_metrics"]
        comparison_records.append({
            "dataset": ds,
            "best_model": summary["best_model_name"],
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
            "roc_auc": best_metrics["roc_auc"]
        })

        score = best_metrics.get("roc_auc", np.nan)
        if np.isnan(score):
            score = best_metrics.get("f1", 0.0)
        if score > overall_best["score"]:
            overall_best.update({"score": score, "dataset": ds, "model": summary["best_model_obj"], "model_name": summary["best_model_name"]})

    # Save combined predictions
    if combined_predictions:
        combined_df = pd.concat(combined_predictions, ignore_index=True)
        combined_out = os.path.join(OUTPUTS_DIR, "patient_predictions.xlsx")
        combined_df.to_excel(combined_out, index=False)
        print(f"\nSaved combined predictions: {combined_out}")

    # Model comparison chart
    if comparison_records:
        comp_df = pd.DataFrame(comparison_records).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(x="dataset", y="roc_auc", data=comp_df)
        plt.ylim(0, 1)
        plt.title("Model comparison by ROC-AUC (best model per dataset)")
        plt.tight_layout()
        comp_out = os.path.join(OUTPUTS_DIR, "model_comparison.png")
        plt.savefig(comp_out, dpi=200)
        plt.close()
        print(f"Saved model comparison plot: {comp_out}")

        comp_xlsx = os.path.join(OUTPUTS_DIR, "model_comparison_results.xlsx")
        comp_df.to_excel(comp_xlsx, index=False)
        print(f"Saved model comparison results: {comp_xlsx}")

    # Save overall best model
    if overall_best["model"] is not None:
        overall_best_path = os.path.join(MODELS_DIR, "best_model.pkl")
        save_pickle(overall_best["model"], overall_best_path)
        print(f"\nOverall best model: dataset='{overall_best['dataset']}', model='{overall_best.get('model_name')}', score={overall_best['score']:.4f}")

    print("\nPipeline finished.")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Disease prediction pipeline runner")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning (GridSearchCV).")
    args = parser.parse_args()
    run_pipeline(use_hyperparam_tuning=args.tune)
