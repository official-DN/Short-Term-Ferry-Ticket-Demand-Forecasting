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
- `assist_ai/connect.py` — Gemini SDK client, retry handling, and readable API errors.
- `assist_ai/script.py` — Streamlit chat interface and 25 project-focused starter prompts.
- `assist_ai/project_context.py` — project instructions and optional runtime context builder.

The app runs the notebook setup/function cells, not notebook main routines or output visualizations. Forecasts are based on the provided historical dataset; no live ferry API is connected.



SUGGESTED QUESTIONS = {
    "Project Overview": [
        "What is the main objective of this project?",
        "Explain the complete project architecture.",
        "What problems does this system solve?",
        "Which technologies are used?",
        "Explain demand forecasting vs crowd prediction."
    ],

    "Data Analysis": [
        "Explain the dataset and its features.",
        "What preprocessing steps are performed?",
        "How are missing intervals handled?",
        "Identify the busiest hours and days.",
        "How do weather and holidays affect demand?"
    ],

    "Forecasting Models": [
        "How does the system forecast demand?",
        "Why are multiple ML models compared?",
        "Which model performs best?",
        "Explain MAE, RMSE and MAPE.",
        "What is rolling-origin validation?"
    ],

    "Crowd & Queue Prediction": [
        "How are passenger arrivals estimated?",
        "Explain the queue prediction formula.",
        "How are crowd levels classified?",
        "How is passenger waiting time calculated?",
        "How are overcrowding alerts generated?"
    ],

    "AI Decision Support": [
        "What if ticket demand increases by 40%?",
        "What if arrivals exceed boarding capacity?",
        "How can ferry schedules be optimized?",
        "Why are confidence intervals important?",
        "Analyze the results and suggest improvements."
    ]
}
