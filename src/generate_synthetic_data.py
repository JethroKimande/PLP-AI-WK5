"""Generate synthetic patient readmission dataset for experimentation."""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .utils import configure_logging, set_global_seed

LOGGER = configure_logging(__name__)


GENDERS = ["Female", "Male"]
INSURANCE_TYPES = ["Medicare", "Medicaid", "Private", "Uninsured"]
DISCHARGE_OPTIONS = ["Home", "Rehab", "Nursing Facility"]
DIAGNOSES = ["CHF", "COPD", "Diabetes", "Renal Failure", "Stroke", "Pneumonia", "Sepsis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic hospital readmission dataset.")
    parser.add_argument("--rows", type=int, default=500, help="Number of synthetic patient records.")
    parser.add_argument("--output", type=str, default="data/synthetic_patients.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def simulate_patient_data(n_rows: int, seed: int) -> pd.DataFrame:
    set_global_seed(seed)

    base_date = datetime(2025, 1, 1)

    data = {
        "patient_id": [f"P{idx:05d}" for idx in range(1, n_rows + 1)],
        "discharge_date": [base_date + timedelta(days=int(np.random.randint(0, 60))) for _ in range(n_rows)],
        "gender": np.random.choice(GENDERS, size=n_rows, p=[0.52, 0.48]),
        "insurance_type": np.random.choice(INSURANCE_TYPES, size=n_rows, p=[0.35, 0.25, 0.3, 0.1]),
        "discharge_disposition": np.random.choice(DISCHARGE_OPTIONS, size=n_rows, p=[0.7, 0.2, 0.1]),
        "primary_diagnosis": np.random.choice(DIAGNOSES, size=n_rows),
        "age": np.random.normal(loc=62, scale=12, size=n_rows).round().clip(18, 95),
        "length_of_stay": np.random.poisson(lam=6, size=n_rows).clip(1, 30),
        "prior_admissions": np.random.poisson(lam=2, size=n_rows),
    }

    medication_ratio = np.random.beta(a=4, b=2, size=n_rows)
    vitals_score = np.random.normal(loc=0.7, scale=0.1, size=n_rows).clip(0.3, 0.95)
    comorbidity_index = np.random.randint(0, 7, size=n_rows)
    socio_economic_score = np.random.normal(loc=50, scale=20, size=n_rows).clip(5, 95)

    readmission_logits = (
        -2.5
        + 0.03 * (data["age"] - 60)
        + 0.4 * (np.array(data["length_of_stay"]) > 7)
        + 0.5 * (np.array(data["prior_admissions"]) >= 3)
        - 1.3 * medication_ratio
        - 1.1 * vitals_score
        + 0.6 * (np.array(comorbidity_index) >= 4)
        - 0.01 * socio_economic_score
    )
    readmission_probs = 1 / (1 + np.exp(-readmission_logits))
    readmitted = np.random.binomial(1, readmission_probs)

    summaries = []
    for idx in range(n_rows):
        diagnosis = data["primary_diagnosis"][idx]
        if readmitted[idx]:
            summary = f"Patient exhibits risk factors post-discharge for {diagnosis}; close follow-up required."
        else:
            summary = f"Stable discharge after treatment for {diagnosis}; routine follow-up scheduled."
        summaries.append(summary)

    df = pd.DataFrame(data)
    df["medication_possession_ratio"] = medication_ratio.round(2)
    df["avg_vitals_score"] = vitals_score.round(2)
    df["comorbidity_index"] = comorbidity_index
    df["socio_economic_score"] = socio_economic_score.round(0)
    df["readmitted"] = readmitted
    df["discharge_summary"] = summaries

    return df


def main() -> None:
    args = parse_args()
    LOGGER.info("Generating %s synthetic patient records", args.rows)
    df = simulate_patient_data(args.rows, args.seed)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Synthetic dataset saved to %s", output_path.resolve())


if __name__ == "__main__":
    main()

