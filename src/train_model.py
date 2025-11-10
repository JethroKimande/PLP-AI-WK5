"""Train a LightGBM classifier for patient readmission risk prediction."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import lightgbm as lgb
import joblib

from .data_pipeline import DatasetSplits, prepare_datasets
from .utils import configure_logging, load_config, save_json, set_global_seed

LOGGER = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM model for hospital readmission risk.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--output", type=str, default="artifacts/model", help="Directory to store trained artifacts.")
    return parser.parse_args()


def train_model(config: Dict[str, Any], dataset: DatasetSplits) -> lgb.LGBMClassifier:
    """Instantiate and train a LightGBM classifier."""
    model_cfg = config["model"]
    params = model_cfg.get("params", {})
    LOGGER.info("Training LightGBM model with params: %s", params)
    model = lgb.LGBMClassifier(**params, random_state=config.get("random_seed", 42))
    model.fit(
        dataset.x_train,
        dataset.y_train,
        eval_set=[(dataset.x_val, dataset.y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def persist_artifacts(model: lgb.LGBMClassifier, preprocessing_pipeline, output_dir: pathlib.Path, feature_names: list[str], metrics: Dict[str, Any]) -> None:
    """Persist model, preprocessing pipeline, feature names, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(preprocessing_pipeline, output_dir / "preprocessing.joblib")
    save_json({"feature_names": feature_names}, output_dir / "feature_names.json")
    save_json(metrics, output_dir / "training_metrics.json")
    LOGGER.info("Artifacts saved to %s", output_dir.resolve())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config.get("random_seed", 42))

    dataset, preprocessing_pipeline = prepare_datasets(config)
    model = train_model(config, dataset)

    metrics = {
        "best_iteration": int(getattr(model, "best_iteration_", 0) or model.get_params().get("n_estimators", 0)),
        "validation_score": float(model.best_score_["valid_0"]["auc"]) if hasattr(model, "best_score_") else None,
    }

    output_dir = pathlib.Path(args.output)
    persist_artifacts(model, preprocessing_pipeline, output_dir, dataset.feature_names, metrics)


if __name__ == "__main__":
    main()

