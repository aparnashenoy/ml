"""Extracts logged experiment metrics (ROC AUC, accuracy, F1, ...) out of the
saved text/stream outputs in the healthcare notebooks, so the app can tell a
data-driven story instead of just showing static images.
"""
import json
import re
from pathlib import Path

METRIC_PATTERNS = [
    ("ROC AUC", re.compile(r"(?:^|\n)[ \t]*([A-Za-z][^\n:]{1,60}?)[:\-]\s*ROC AUC\s*=\s*([\d.]+)\s*%")),
    ("Accuracy", re.compile(r"Model Accuracy:\s*([\d.]+)")),
    ("F1 Score", re.compile(r"Test F1 Score\s*=\s*([\d.]+)\s*%")),
    ("CV Best Score", re.compile(r"Best Score:\s*([\d.]+)")),
    ("Mean ROC AUC", re.compile(r"Mean ROC AUC:\s*([\d.]+)")),
    ("Accuracy (labelled)", re.compile(r"Accuracy of ([A-Za-z0-9_ ]{2,40}?)(?: classifier)?(?: on test set)?:\s*([\d.]+)%?")),
]

# Ordered most-specific-first: filename substring -> human-readable technique/category.
CATEGORY_RULES = [
    ("rac_2022", "RAC 2022 Ensemble"),
    ("smote_undersample", "SMOTE + Resampling Ensemble"),
    ("smote_oversample", "SMOTE + Resampling Ensemble"),
    ("no_smote", "No-SMOTE Baseline"),
    ("undersampling", "Undersampling"),
    ("undersample", "Undersampling"),
    ("oversampling", "Oversampling"),
    ("oversample", "Oversampling"),
    ("smote", "SMOTE"),
    ("cost", "Cost-sensitive"),
    ("paper3", "Feature-engineered (paper3)"),
    ("imputation", "Imputation technique"),
    ("mice", "Imputation technique"),
    ("datawig", "Imputation technique"),
    ("mean__median", "Imputation technique"),
    ("kaggle_heartparameters", "Heart-parameters model"),
    ("heartparameters", "Heart-parameters model"),
    ("metabolicparameters", "Metabolic-parameters model"),
    ("cvs_resp", "CVS/Respiratory model"),
    ("cvs_cost_function", "CVS/Respiratory model"),
    ("sepsisprediction", "Sepsis prediction (full pipeline)"),
    ("class_imbalance", "Class-imbalance exploration"),
    ("boxplot_feature", "Feature engineering"),
    ("feature_engineering", "Feature engineering"),
    ("xgboost", "XGBoost model"),
    ("feature_importance", "EDA / Preprocessing"),
    ("data_analysis", "EDA / Preprocessing"),
    ("new analysis", "EDA / Preprocessing"),
    ("data_pre_processing", "EDA / Preprocessing"),
]


def category_for(filename: str) -> str:
    name = filename.lower()
    for key, label in CATEGORY_RULES:
        if key in name:
            return label
    return "Other"


def _cell_texts(nb: dict):
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                yield "".join(out.get("text", []))
            elif "text/plain" in out.get("data", {}):
                yield "".join(out["data"]["text/plain"])


def extract_metrics(notebook_paths: list[Path]) -> list[dict]:
    rows = []
    for path in notebook_paths:
        nb = json.loads(Path(path).read_text())
        category = category_for(path.name)
        for text in _cell_texts(nb):
            if not text:
                continue
            for metric_name, pattern in METRIC_PATTERNS:
                for m in pattern.finditer(text):
                    groups = m.groups()
                    model, raw_value = ("", groups[0]) if len(groups) == 1 else groups
                    try:
                        value = float(raw_value)
                    except ValueError:
                        continue
                    if value > 1.5:  # was reported as a percentage (0-100 scale)
                        value /= 100
                    if not (0 <= value <= 1.0001):
                        continue
                    rows.append(
                        {
                            "notebook": path.name,
                            "category": category,
                            "metric": metric_name,
                            "model": model.strip() or "(unlabeled)",
                            "value": value,
                        }
                    )
    return rows
