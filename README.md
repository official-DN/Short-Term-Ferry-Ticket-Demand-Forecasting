# Short Ferry Forecasting Streamlit Application

This app integrates the original notebook-defined workflow from `dataProcess/` and KPI/score functions from `key_performence/`.

## Run

```bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app.py
```

## Project integration
- `main_01.ipynb` : just basic operation on csv files for next utilization.
- `main_02train_test.ipynb`: loading, cleaning, 15-minute resampling, feature engineering, targets, and chronological split.
- `main_03models.ipynb`: baselines, Linear Regression, Random Forest, Gradient Boosting, optional XGBoost, and project metrics.
- `main_04evaluation.ipynb`: they evaluation step to enhence the modals accuracy.
- `main_05uncertainty_ana.ipynb`: empirical prediction interval functions.
- `key_performence/indicator.py`: KPI calculation.
- `key_performence/show_score.py`: regression and peak-event scores.

The app runs the notebook setup/function cells, not notebook main routines or output visualizations. Forecasts are based on the provided historical dataset; no live ferry API is connected.
