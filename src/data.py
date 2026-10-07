"""Dataset loading and cleaning utilities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_YEAR = 2026

COLUMN_ALIASES = {
    "price": ["selling_price", "sale_price", "price", "selling price", "resale_price"],
    "year": ["year", "manufacturing_year", "model_year", "yr_mfr", "mfg_year", "year_of_manufacture"],
    "km_driven": ["km_driven", "kilometers_driven", "kilometres_driven", "mileage_driven", "kms_run", "km_run", "kms"],
    "name": ["name", "car_name", "vehicle_name"],
    "make": ["make", "brand", "manufacturer"],
    "model": ["model", "car_model", "vehicle_model"],
    "fuel": ["fuel", "fuel_type"],
    "seller_type": ["seller_type", "seller", "seller type"],
    "transmission": ["transmission", "gearbox"],
    "owner": ["owner", "ownership", "owner_type"],
    "mileage": ["mileage", "mileage_kmpl", "mileage (kmpl)"],
    "engine": ["engine", "engine_cc", "engine (cc)"],
    "max_power": ["max_power", "power", "max power"],
    "seats": ["seats", "seating_capacity", "seating capacity"],
    "drivetrain": ["drivetrain", "drive_train", "drive"],
    "body_type": ["body_type", "body type", "type"],
    "exterior_color": ["exterior_color", "exterior color", "paint_color", "paint color"],
    "interior_color": ["interior_color", "interior color"],
    "condition": ["condition", "vehicle_condition"],
    "trim": ["trim", "variant"],
    "accident_history": ["accident_history", "accident history", "accidents"],
    "clean_title": ["clean_title", "clean title", "title_status", "title status"],
    "cylinders": ["cylinders", "cylinder"],
    "horsepower": ["horsepower", "hp"],
    "odometer": ["odometer", "mileage", "miles"],
}

CANONICAL_FEATURES = [
    "age", "km_driven", "mileage", "odometer", "engine", "max_power", "horsepower", "cylinders", "seats",
    "name", "make", "model", "fuel", "seller_type", "transmission", "owner", "drivetrain", "body_type",
    "exterior_color", "interior_color", "condition", "trim", "accident_history", "clean_title",
]


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {_clean_name(column): column for column in columns}
    for alias in aliases:
        if _clean_name(alias) in lookup:
            return lookup[_clean_name(alias)]
    return None


def _number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename recognized dataset fields to consistent internal names."""
    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _find_column(list(frame.columns), aliases)
        if found:
            rename_map[found] = canonical
    return frame.rename(columns=rename_map).copy()


def load_vehicle_data(path: str | Path) -> pd.DataFrame:
    """Load, validate, normalize, and clean a used vehicle CSV."""
    frame = standardize_columns(pd.read_csv(path))
    if "price" not in frame:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(
            "A target price column was not found. Expected selling_price or price. "
            f"Found: {available}"
        )
    if "year" in frame:
        frame["age"] = CURRENT_YEAR - pd.to_numeric(frame["year"], errors="coerce")
        frame.loc[(frame["age"] < 0) | (frame["age"] > 80), "age"] = np.nan
    for column in ["price", "km_driven", "mileage", "odometer", "engine", "max_power", "horsepower", "cylinders", "seats"]:
        if column in frame:
            frame[column] = frame[column].map(_number)
    for column in [
        "name", "make", "model", "fuel", "seller_type", "transmission", "owner", "drivetrain", "body_type",
        "exterior_color", "interior_color", "condition", "trim", "accident_history", "clean_title",
    ]:
        if column in frame:
            frame[column] = (
                frame[column].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})
                .fillna("Unknown").astype(object)
            )

    available_features = [column for column in CANONICAL_FEATURES if column in frame]
    if not available_features:
        raise ValueError("No recognized vehicle feature columns were found in the CSV.")
    frame = frame.dropna(subset=["price"]).drop_duplicates().copy()
    frame = frame[(frame["price"] > 0) & np.isfinite(frame["price"])]
    return frame[["price", *available_features]]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the numeric selling price target."""
    return frame.drop(columns="price"), frame["price"]
