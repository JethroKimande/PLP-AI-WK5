"""Data loading and preprocessing utilities for the AI workflow."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover - optional dependency handling
    SMOTE = None
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import configure_logging

LOGGER = configure_logging(__name__)


@dataclass
class DatasetSplits:
    """Container for dataset splits."""

    x_train: Any
    x_val: Any
    x_test: Any
    y_train: Any
    y_val: Any
    y_test: Any
    feature_names: list[str]


def _build_column_transformer(config: Dict[str, Any]) -> ColumnTransformer:
    """Create preprocessing column transformer based on configuration."""
    data_cfg = config["data"]
    preprocess_cfg = config["preprocessing"]

    numeric_features = data_cfg.get("numeric_features", [])
    categorical_features = data_cfg.get("categorical_features", [])
    text_features = data_cfg.get("text_features", [])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=preprocess_cfg.get("imputation_strategy", "median"))),
            ("scaler", StandardScaler() if preprocess_cfg.get("scale_numeric", True) else "passthrough"),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=preprocess_cfg.get("impute_categorical", "most_frequent"))),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if text_features:
        # Apply TF-IDF independently to each text column then concatenate.
        for feature in text_features:
            transformers.append(
                (
                    f"tfidf_{feature}",
                    Pipeline(
                        steps=[
                            ("tfidf", TfidfVectorizer(max_features=256, ngram_range=(1, 2))),
                        ]
                    ),
                    feature,
                )
            )

    if not transformers:
        raise ValueError("No transformers configured. Check feature definitions.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def _temporal_split(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically if temporal_split flag is set."""
    split_cfg = config["splits"]
    target = config["data"]["target_column"]
    datetime_col = config["data"].get("datetime_column")

    if not datetime_col or datetime_col not in df.columns:
        return _standard_split(df, config)

    LOGGER.info("Performing temporal split using column '%s'", datetime_col)
    df_sorted = df.sort_values(datetime_col).reset_index(drop=True)

    train_ratio = split_cfg["train_ratio"]
    val_ratio = split_cfg["validation_ratio"]

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    if split_cfg.get("stratify", False):
        LOGGER.warning("Stratified sampling is not supported with temporal splits; ignoring.")

    # Ensure each split contains both classes.
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if split_df[target].nunique() < 2:
            LOGGER.warning("Split '%s' has less than two classes; consider adjusting ratios.", split_name)

    return train_df, val_df, test_df


def _standard_split(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform standard random split."""
    split_cfg = config["splits"]
    target = config["data"]["target_column"]

    train_df, temp_df = train_test_split(
        df,
        train_size=split_cfg["train_ratio"],
        random_state=config.get("random_seed", 42),
        stratify=df[target] if split_cfg.get("stratify", False) else None,
    )

    val_ratio = split_cfg["validation_ratio"] / (split_cfg["validation_ratio"] + split_cfg["test_ratio"])
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=config.get("random_seed", 42),
        stratify=temp_df[target] if split_cfg.get("stratify", False) else None,
    )

    return train_df, val_df, test_df


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset from configured path."""
    data_path = pathlib.Path(config["data"]["raw_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    LOGGER.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    if config["data"].get("datetime_column") in df.columns:
        df[config["data"]["datetime_column"]] = pd.to_datetime(df[config["data"]["datetime_column"]])
    return df


def prepare_datasets(config: Dict[str, Any]) -> DatasetSplits:
    """Load data, split into train/validation/test, and apply preprocessing pipeline."""
    df = load_dataset(config)
    train_df, val_df, test_df = (
        _temporal_split(df, config)
        if config["splits"].get("temporal_split", False)
        else _standard_split(df, config)
    )

    target_col = config["data"]["target_column"]
    id_col = config["data"].get("id_column")

    x_train = train_df.drop(columns=[target_col] + ([id_col] if id_col and id_col in train_df.columns else []))
    y_train = train_df[target_col]

    x_val = val_df.drop(columns=[target_col] + ([id_col] if id_col and id_col in val_df.columns else []))
    y_val = val_df[target_col]

    x_test = test_df.drop(columns=[target_col] + ([id_col] if id_col and id_col in test_df.columns else []))
    y_test = test_df[target_col]

    column_transformer = _build_column_transformer(config)
    LOGGER.info("Fitting preprocessing pipeline on training data")
    preprocessing_pipeline = Pipeline(
        steps=[
            ("transformer", column_transformer),
        ]
    )
    x_train_processed = preprocessing_pipeline.fit_transform(x_train, y_train)
    x_val_processed = preprocessing_pipeline.transform(x_val)
    x_test_processed = preprocessing_pipeline.transform(x_test)

    feature_names = _extract_feature_names(preprocessing_pipeline, column_transformer)

    if config["preprocessing"].get("oversample_minority", False):
        LOGGER.info("Applying SMOTE oversampling to training data")
        if SMOTE is None:
            msg = "SMOTE oversampling requested but imbalanced-learn is not available or failed to import."
            raise ImportError(msg)
        smote = SMOTE(random_state=config.get("random_seed", 42))
        x_train_processed, y_train = smote.fit_resample(x_train_processed, y_train)

    return DatasetSplits(
        x_train=x_train_processed,
        x_val=x_val_processed,
        x_test=x_test_processed,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
    ), preprocessing_pipeline


def _extract_feature_names(preprocessing_pipeline: Pipeline, column_transformer: ColumnTransformer) -> list[str]:
    """Attempt to derive human-readable feature names from preprocessing pipeline."""
    feature_names: list[str] = []
    for name, transformer, cols in column_transformer.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            if isinstance(cols, str):
                cols = [cols]
            transformed_names = list(transformer.get_feature_names_out(cols))
            feature_names.extend(transformed_names)
        elif hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                if isinstance(cols, str):
                    cols = [cols]
                transformed_names = list(last_step.get_feature_names_out(cols))
                feature_names.extend(transformed_names)
            else:
                feature_names.extend(
                    [f"{name}__{col}" for col in ([cols] if isinstance(cols, str) else cols)]
                )
        else:
            feature_names.extend(
                [f"{name}__{col}" for col in ([cols] if isinstance(cols, str) else cols)]
            )
    return feature_names

