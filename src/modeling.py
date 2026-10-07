"""Preprocessing, model comparison, and feature impact tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass
class TrainingResult:
    model: Pipeline
    model_name: str
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]


def _preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop",
    )


def _pipeline(estimator, numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> Pipeline:
    return Pipeline([
        ("preprocessor", _preprocessor(numeric_features, categorical_features, scale_numeric)),
        ("model", estimator),
    ])


def train_and_evaluate(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> TrainingResult:
    """Train the requested regressors and choose the lowest-RMSE model."""
    if len(frame) < 20:
        raise ValueError("At least 20 usable rows are required for model training.")
    x = frame.drop(columns="price")
    y = frame["price"]
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]
    if not numeric_features and not categorical_features:
        raise ValueError("The dataset has no usable features.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    candidates = {
        "Linear Regression": _pipeline(LinearRegression(), numeric_features, categorical_features, True),
        "Decision Tree": _pipeline(DecisionTreeRegressor(random_state=random_state, min_samples_leaf=3), numeric_features, categorical_features, False),
        "Random Forest": _pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1), numeric_features, categorical_features, False),
    }
    rows: list[dict[str, float | str]] = []
    trained: dict[str, Pipeline] = {}
    predicted: dict[str, np.ndarray] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        trained[name] = pipeline
        predicted[name] = y_pred
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
            "R2": r2_score(y_test, y_pred),
        })
    metrics = pd.DataFrame(rows).sort_values("RMSE", ignore_index=True)
    best_name = str(metrics.loc[0, "model"])
    best_model = trained[best_name]
    prediction_frame = pd.DataFrame({"actual_price": y_test.to_numpy(), "predicted_price": predicted[best_name]})
    return TrainingResult(
        model=best_model,
        model_name=best_name,
        metrics=metrics,
        feature_importance=feature_importance(best_model),
        predictions=prediction_frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return absolute coefficients or tree importances for transformed inputs."""
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        raise ValueError("The selected estimator does not expose feature importance.")
    output = pd.DataFrame({"feature": transformed_names, "importance": values})
    return output.sort_values("importance", ascending=False, ignore_index=True)
