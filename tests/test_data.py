import pandas as pd

from src.data import load_vehicle_data


def test_loader_normalizes_common_columns(tmp_path):
    csv_path = tmp_path / "cars.csv"
    pd.DataFrame({
        "selling_price": [200000, 350000],
        "year": [2015, 2018],
        "km_driven": [80000, 45000],
        "fuel": ["Diesel", "Petrol"],
    }).to_csv(csv_path, index=False)
    result = load_vehicle_data(csv_path)
    assert {"price", "age", "km_driven", "fuel"}.issubset(result.columns)
    assert result["price"].tolist() == [200000.0, 350000.0]


def test_loader_supports_supplied_dataset_style_columns(tmp_path):
    csv_path = tmp_path / "supplied_dataset.csv"
    pd.DataFrame({
        "price": [12000, 16000],
        "make": ["Toyota", "Honda"],
        "model": ["Camry", "Civic"],
        "fuel_type": ["Gasoline", "Gasoline"],
        "drivetrain": ["FWD", "FWD"],
        "accident_history": ["None reported", "None reported"],
    }).to_csv(csv_path, index=False)
    result = load_vehicle_data(csv_path)
    assert {"make", "model", "fuel", "drivetrain", "accident_history"}.issubset(result.columns)


def test_loader_supports_kaggle_sale_price_target(tmp_path):
    csv_path = tmp_path / "kaggle_dataset.csv"
    pd.DataFrame({
        "sale_price": [200000, 350000],
        "yr_mfr": [2015, 2018],
        "kms_run": [80000, 45000],
        "fuel_type": ["Diesel", "Petrol"],
    }).to_csv(csv_path, index=False)
    result = load_vehicle_data(csv_path)
    assert result["price"].tolist() == [200000.0, 350000.0]
    assert {"age", "km_driven"}.issubset(result.columns)
    assert result["age"].tolist() == [11.0, 8.0]
