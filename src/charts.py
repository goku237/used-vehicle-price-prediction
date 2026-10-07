"""Polished matplotlib charts shared by the training reports and the web interface."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from sklearn.pipeline import Pipeline

INK = "#172033"
MUTED = "#69748a"
GRID = "#e7ebf3"
PRIMARY = "#1d5aa6"
PRIMARY_MID = "#5f93cf"
PRIMARY_PALE = "#c8dbf0"
SELECTED = "#0e7a5f"
DANGER = "#cf4b3f"

DISPLAY_NAMES = {
    "age": "Vehicle age", "km_driven": "Distance driven", "mileage": "Fuel mileage", "odometer": "Odometer",
    "engine": "Engine capacity", "max_power": "Maximum power", "horsepower": "Horsepower", "cylinders": "Cylinders",
    "seats": "Seats", "make": "Make", "model": "Model", "fuel": "Fuel type", "seller_type": "Seller type",
    "transmission": "Transmission", "owner": "Owner history", "drivetrain": "Drivetrain", "body_type": "Body type",
    "exterior_color": "Exterior colour", "interior_color": "Interior colour", "condition": "Condition", "trim": "Trim",
    "accident_history": "Accident history", "clean_title": "Clean title", "name": "Vehicle name",
}


def display_label(column: str) -> str:
    """Human-friendly name for an input column."""
    return DISPLAY_NAMES.get(str(column), _default_label(str(column)))


def save_chart(figure: plt.Figure, path) -> None:
    """Write a figure to disk and release it from memory."""
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)



def _theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 10.5,
        "font.family": "DejaVu Sans",
        "axes.axisbelow": True,
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _default_label(feature: str) -> str:
    text = feature
    for prefix in ("numeric__", "categorical__"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.replace("_", " ").strip().title()


def _pretty_money(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:,.0f}k"
    return f"{value:,.0f}"


def model_comparison_chart(metrics: pd.DataFrame, selected: str | None = None) -> plt.Figure:
    """Accuracy and error panels for every candidate model, with the winner highlighted."""
    _theme()
    models = metrics["model"].astype(str).tolist()
    selected = selected or (models[0] if models else None)
    positions = np.arange(len(models))
    colors = [PRIMARY if name == selected else PRIMARY_PALE for name in models]

    figure, (accuracy_axis, error_axis) = plt.subplots(1, 2, figsize=(10.4, 3.9))

    scores = metrics["R2"].astype(float).to_numpy()
    bars = accuracy_axis.bar(positions, scores, width=0.58, color=colors, zorder=3)
    accuracy_axis.set_title("Accuracy by model (R²)")
    accuracy_axis.set_ylabel("Validation R²")
    accuracy_axis.set_ylim(0, min(1.0, max(scores) * 1.35) if scores.max() > 0 else 1)
    accuracy_axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    accuracy_axis.set_xticks(positions)
    accuracy_axis.set_xticklabels(models, rotation=12, ha="right")
    accuracy_axis.grid(axis="y", zorder=0)
    accuracy_axis.grid(axis="x", visible=False)
    for bar, score in zip(bars, scores):
        accuracy_axis.annotate(
            f"{score:.3f}",
            (bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", color=INK,
        )
    if selected:
        try:
            winner = models.index(selected)
        except ValueError:
            winner = -1
        if winner >= 0:
            accuracy_axis.annotate(
                "selected",
                (positions[winner], scores[winner]),
                xytext=(0, 22), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color=SELECTED,
            )

    width = 0.36
    mae = metrics["MAE"].astype(float).to_numpy()
    rmse = metrics["RMSE"].astype(float).to_numpy()
    error_axis.bar(positions - width / 2, mae, width, label="MAE", color=PRIMARY, zorder=3)
    error_axis.bar(positions + width / 2, rmse, width, label="RMSE", color=PRIMARY_MID, zorder=3)
    error_axis.set_title("Prediction error by model")
    error_axis.set_ylabel("Error (price units)")
    error_axis.set_xticks(positions)
    error_axis.set_xticklabels(models, rotation=12, ha="right")
    error_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _pretty_money(value)))
    error_axis.grid(axis="y", zorder=0)
    error_axis.grid(axis="x", visible=False)
    error_axis.set_ylim(0, max(mae.max(), rmse.max()) * 1.22)
    error_axis.legend(ncol=2, loc="upper right")
    for offset, values in ((-width / 2, mae), (width / 2, rmse)):
        for position, value in zip(positions + offset, values):
            error_axis.annotate(
                _pretty_money(value),
                (position, value),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK,
            )

    figure.suptitle("Model comparison on the held-out validation split", fontsize=11, color=MUTED, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def feature_impact_chart(
    impact: pd.DataFrame,
    top: int = 10,
    labeler=None,
    title: str = "Most influential inputs",
) -> plt.Figure:
    """Horizontal bars for the strongest input fields, shown as a share of total impact."""
    _theme()
    labeler = labeler or display_label
    data = impact.head(top).iloc[::-1]
    total = float(impact["importance"].sum()) or 1.0
    shares = data["importance"].astype(float) / total * 100
    labels = [labeler(str(feature)) for feature in data["feature"]]
    count = len(data)
    colors = [
        PRIMARY_PALE if count == 1 else
        "#{:02x}{:02x}{:02x}".format(
            int(0xC8 + (0x1D - 0xC8) * index / (count - 1)),
            int(0xDB + (0x5A - 0xDB) * index / (count - 1)),
            int(0xF0 + (0xA6 - 0xF0) * index / (count - 1)),
        )
        for index in range(count)
    ]

    figure, axis = plt.subplots(figsize=(7.8, 4.5))
    bars = axis.barh(labels, shares.to_numpy(), color=colors, height=0.68, zorder=3)
    axis.set_title(title)
    axis.set_xlabel("Share of total model impact (%)")
    axis.set_xlim(0, max(shares.max() * 1.22, 1.0))
    axis.grid(axis="x", zorder=0)
    axis.grid(axis="y", visible=False)
    for bar, share in zip(bars, shares):
        axis.annotate(
            f"{share:.1f}%",
            (share, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=9.5, fontweight="bold", color=INK,
        )
    figure.tight_layout()
    return figure


def price_distribution_chart(prices: pd.Series) -> plt.Figure:
    """Histogram of listing prices with median and mean markers."""
    _theme()
    values = pd.Series(prices).dropna().astype(float)
    median = float(values.median())
    mean = float(values.mean())
    upper = max(float(values.quantile(0.99)) * 1.02, 1.0)
    hidden = int((values > upper).sum())

    figure, axis = plt.subplots(figsize=(7.8, 4.3))
    axis.hist(values, bins=40, range=(0.0, upper), color=PRIMARY_MID, edgecolor="white",
              linewidth=0.7, zorder=3)
    axis.axvline(median, color=PRIMARY, linestyle="--", linewidth=1.8, zorder=4,
                 label=f"Median {_pretty_money(median)}")
    axis.axvline(mean, color=DANGER, linestyle=":", linewidth=1.8, zorder=4,
                 label=f"Mean {_pretty_money(mean)}")
    axis.set_title("Listing price distribution")
    axis.set_ylabel("Listings")
    axis.set_xlabel(f"Price · view up to {_pretty_money(upper)}" + (f" ({hidden} pricier listings not shown)" if hidden else ""))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _pretty_money(value)))
    axis.set_xlim(0, upper)
    axis.grid(axis="y", zorder=0)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def actual_vs_predicted_chart(predictions: pd.DataFrame, model_name: str, r2: float | None = None) -> plt.Figure:
    """Scatter of actual versus predicted prices with a perfect-prediction diagonal."""
    _theme()
    actual = predictions["actual_price"].astype(float).to_numpy()
    predicted = predictions["predicted_price"].astype(float).to_numpy()
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))

    figure, axis = plt.subplots(figsize=(7.6, 5.4))
    axis.scatter(actual, predicted, s=18, alpha=0.45, color=PRIMARY, edgecolors="none", zorder=3)
    axis.plot([low, high], [low, high], color=DANGER, linestyle="--", linewidth=1.6, zorder=4,
              label="Perfect prediction")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Actual vs predicted price ({model_name})")
    axis.set_xlabel("Actual price")
    axis.set_ylabel("Predicted price")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _pretty_money(value)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _pretty_money(value)))
    axis.grid(zorder=0)
    axis.legend(loc="upper left")
    if r2 is not None:
        axis.annotate(
            f"R² = {r2:.3f}",
            xy=(0.03, 0.95), xycoords="axes fraction",
            fontsize=11, fontweight="bold", color=INK, ha="left", va="top",
        )
    figure.tight_layout()
    return figure


def _block_labels(transformer, columns, output_count: int) -> list[str] | None:
    named_steps = getattr(transformer, "named_steps", {})
    onehot = named_steps.get("onehot") if hasattr(named_steps, "get") else None
    if onehot is not None and hasattr(onehot, "categories_"):
        counts = [len(categories) for categories in onehot.categories_]
        if sum(counts) == output_count:
            labels: list[str] = []
            for column, count in zip(columns, counts):
                labels.extend([str(column)] * count)
            return labels
    if output_count == len(columns):
        return [str(column) for column in columns]
    return None


def _source_labels(preprocessor) -> list[str] | None:
    """Map every transformed feature back to the original input column it came from."""
    labels: list[str] = []
    for _, transformer, columns in preprocessor.transformers_:
        if transformer in ("drop", "passthrough") or not hasattr(transformer, "get_feature_names_out"):
            return None
        try:
            output_count = len(list(transformer.get_feature_names_out()))
        except Exception:
            return None
        block = _block_labels(transformer, columns, output_count)
        if block is None:
            return None
        labels.extend(block)
    try:
        total = len(list(preprocessor.get_feature_names_out()))
    except Exception:
        return None
    return labels if len(labels) == total else None


def grouped_feature_importance(model: Pipeline, importance: pd.DataFrame) -> pd.DataFrame | None:
    """Combine one-hot impacts into one row per original input column."""
    preprocessor = model.named_steps["preprocessor"]
    labels = _source_labels(preprocessor)
    if labels is None:
        return None
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    if len(names) != len(labels) or len(importance) != len(names):
        return None
    mapping = dict(zip(names, labels))
    frame = importance.copy()
    frame["feature"] = frame["feature"].astype(str).map(mapping)
    if frame["feature"].isna().any():
        return None
    estimator = model.named_steps["model"]
    aggregate = "mean" if hasattr(estimator, "coef_") and not hasattr(estimator, "feature_importances_") else "sum"
    grouped = getattr(frame.groupby("feature", as_index=False)["importance"], aggregate)()
    return grouped.sort_values("importance", ascending=False, ignore_index=True)
