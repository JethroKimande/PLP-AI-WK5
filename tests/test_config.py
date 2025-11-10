"""Basic configuration sanity checks."""

from __future__ import annotations

from src.utils import load_config


def test_split_ratios_sum_to_one():
    config = load_config("config/experiment.yaml")
    ratios = config["splits"]
    total = ratios["train_ratio"] + ratios["validation_ratio"] + ratios["test_ratio"]
    assert abs(total - 1.0) < 1e-6, "Split ratios must sum to 1.0"


def test_model_type_is_lightgbm():
    config = load_config("config/experiment.yaml")
    assert config["model"]["type"] == "lightgbm_classifier"

