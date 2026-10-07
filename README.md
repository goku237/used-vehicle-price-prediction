# Used Vehicle Price Prediction and Feature Impact Analysis

This project predicts a used vehicle's selling price from vehicle details. It implements the work described in the review deck:

- cleans a Kaggle-style used-car CSV;
- explores price relationships;
- trains Linear Regression, Decision Tree, and Random Forest regressors;
- compares them with MAE, RMSE, and R²;
- saves the best model and shows feature impact;
- provides a Streamlit interface for individual price predictions.

## Dataset

Download the supplied [Used Car Price Prediction dataset](https://www.kaggle.com/datasets/vrajesh0sharma7/used-car-price-prediction), extract its CSV (for example, `split_file_1.csv`), and save it as `data/raw/used_cars.csv`. This project supports its fields, including `make`, `model`, `transmission`, `fuel_type`, `drivetrain`, `body_type`, `condition`, `trim`, and `accident_history`.

## Setup

Use Python 3.10 or later.

```powershell
cd C:\Users\gokul\Documents\Codex\2026-10-07\thi\outputs\used-vehicle-price-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the models

```powershell
python train.py --data data/raw/used_cars.csv
```

The command creates:

- `artifacts/vehicle_price_model.joblib` - the selected model and input metadata;
- `artifacts/model_metrics.csv` - validation results for all three models;
- `artifacts/feature_importance.csv` - ranked feature impact values;
- `reports/figures/` - model comparison, price distribution, actual-vs-predicted, and feature impact plots.

## Run the web app

Train the model first, then run:

```powershell
streamlit run app.py
```

Choose vehicle values in the main form and select **Estimate resale price**. The interface shows the prediction, a model-comparison table with accuracy and error charts, the ranked feature impacts, and the price distribution of the training data.

## Project structure

```text
app.py                    Streamlit prediction interface
train.py                  End-to-end training entry point
src/data.py               CSV normalization and data cleaning
src/modeling.py           Preprocessing, training, metrics, feature impact
src/charts.py             Shared charts for reports and the web interface
data/raw/                 Place the downloaded CSV here
artifacts/                Generated model and CSV outputs
reports/figures/          Generated visualizations
tests/                    Lightweight automated checks
```

## Notes

- Prices remain in the currency and units supplied by the dataset. Verify the Kaggle dataset's currency before presenting results.
- This is an estimate from historical records. It does not account for inspection condition, accident history, location, demand, or negotiation.
