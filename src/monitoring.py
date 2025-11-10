"""Monitoring utilities for post-deployment model oversight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    feature: str
    psi_value: float
    breached: bool


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between expected and actual distributions."""
    expected_percents, _ = np.histogram(expected, bins=bins, range=(expected.min(), expected.max()), density=True)
    actual_percents, _ = np.histogram(actual, bins=bins, range=(expected.min(), expected.max()), density=True)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 1e-6, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-6, actual_percents)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi)


def generate_drift_report(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.2,
    features: list[str] | None = None,
) -> Dict[str, DriftReport]:
    """Compare baseline and current feature distributions and return drift report."""
    if features is None:
        features = list(set(baseline.columns) & set(current.columns))

    report: Dict[str, DriftReport] = {}
    for feature in features:
        psi_value = population_stability_index(baseline[feature].to_numpy(), current[feature].to_numpy())
        breached = psi_value >= threshold
        report[feature] = DriftReport(feature=feature, psi_value=psi_value, breached=breached)
    return report

