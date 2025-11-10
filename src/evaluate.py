"""Evaluate trained model artifacts on validation and test sets."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import json
import joblib
from sklearn import metrics

from .data_pipeline import prepare_datasets
from .utils import configure_logging, load_config, save_json

LOGGER = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained hospital readmission model.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--artifacts", default="artifacts/model", help="Directory of trained artifacts.")
    parser.add_argument("--output", default="artifacts/evaluation", help="Directory for evaluation reports.")
    return parser.parse_args()


def load_artifacts(artifacts_dir: pathlib.Path):
    model = joblib.load(artifacts_dir / "model.joblib")
    preprocessing = joblib.load(artifacts_dir / "preprocessing.joblib")
    feature_meta_path = artifacts_dir / "feature_names.json"
    feature_meta = None
    if feature_meta_path.exists():
        with feature_meta_path.open("r", encoding="utf-8") as fh:
            feature_meta = json.load(fh)
    return model, preprocessing, feature_meta


def compute_metrics(y_true, y_proba, threshold: float) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": metrics.roc_auc_score(y_true, y_proba),
        "average_precision": metrics.average_precision_score(y_true, y_proba),
        "f1": metrics.f1_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
    }


def build_confusion_matrix(y_true, y_proba, threshold: float) -> Dict[str, int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifacts_dir = pathlib.Path(args.artifacts)
    model, _, _ = load_artifacts(artifacts_dir)
    dataset, _ = prepare_datasets(config)

    y_val_proba = model.predict_proba(dataset.x_val)[:, 1]
    y_test_proba = model.predict_proba(dataset.x_test)[:, 1]

    threshold = config["evaluation"].get("threshold", 0.5)
    val_metrics = compute_metrics(dataset.y_val, y_val_proba, threshold)
    test_metrics = compute_metrics(dataset.y_test, y_test_proba, threshold)

    reports = {
        "threshold": threshold,
        "validation": val_metrics,
        "test": test_metrics,
        "confusion_matrix": build_confusion_matrix(dataset.y_val, y_val_proba, threshold),
    }

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(reports, output_dir / "metrics.json")

    LOGGER.info("Evaluation complete. ROC-AUC (validation): %.3f", val_metrics["roc_auc"])


if __name__ == "__main__":
    main()

