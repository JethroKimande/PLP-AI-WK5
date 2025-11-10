"""Feature engineering utilities for the hospital readmission workflow."""

from __future__ import annotations

import pandas as pd


def compute_length_of_stay(df: pd.DataFrame) -> pd.Series:
    """Example feature: compute length of stay in days from admission/discharge columns, if available."""
    if {"admission_date", "discharge_date"}.issubset(df.columns):
        return (pd.to_datetime(df["discharge_date"]) - pd.to_datetime(df["admission_date"])).dt.days.clip(lower=0)
    raise KeyError("Columns 'admission_date' and 'discharge_date' required to compute length of stay.")


def normalize_vitals(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Create aggregated vitals score by averaging z-scored vital signs."""
    if not set(columns).issubset(df.columns):
        missing = set(columns) - set(df.columns)
        raise KeyError(f"Missing vital sign columns: {missing}")
    normed = df[columns].apply(lambda col: (col - col.mean()) / (col.std() or 1e-6))
    return normed.mean(axis=1)


def socio_economic_proxy(df: pd.DataFrame) -> pd.Series:
    """Generate a socio-economic status proxy combining zip-level income and education."""
    required_cols = {"median_income", "education_index"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing columns for socio-economic proxy: {missing}")
    return (0.6 * df["median_income"].rank(pct=True) + 0.4 * df["education_index"].rank(pct=True)) * 100

