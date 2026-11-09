"""Project-specific context builder for the Ferry Forecasting Assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_INSTRUCTIONS = """
You are FerryAI, the AI assistant for the Short-Term Ferry Ticket Demand
Forecasting & Predictive Decision Support System for Toronto Island ferry
operations.

Help explain the project's purpose, architecture, data preparation, time-series
forecasting, model comparison, MAE/RMSE/MAPE, rolling-origin validation,
uncertainty intervals, KPIs, passenger arrival estimation, queue prediction,
crowd levels, waiting-time estimates, and operational decision support.

Important accuracy rules:
- Use only facts and numerical values provided in the user's question or
  supplied project context.
- Never invent forecast values, model accuracy, passenger counts, queue sizes,
  waiting times, dataset columns, or results.
- If required project data is absent, say what is missing and explain how to
  obtain it.
- Clearly distinguish historical ticket sales, forecast ticket demand,
  estimated passenger arrivals, and observed physical passenger counts.
- Ticket sales alone do not prove when passengers arrive at the terminal.
  Arrival estimates require calibrated ticket-to-arrival behaviour or must be
  labelled as assumptions/scenarios.
- Queue balance is Q_next = max(0, Q_current + arrivals - boarded).
  Boarding is constrained by actual interval boarding capacity and ferry
  operations.
- Treat crowd thresholds and waiting-time outputs as estimates unless they
  have been calibrated against real terminal observations and schedules.
- Give practical, concise, step-by-step explanations suitable for a student
  project. Explain equations and assumptions when useful.
- Do not claim access to live ferry operations, ticketing systems, weather,
  or external project files unless those data are explicitly supplied.
"""

def build_project_context(
    project_summary: str = "",
    data_summary: str = "",
    forecast_summary: str = "",
    model_summary: str = "",
    crowd_summary: str = "",
) -> str:
    """Combine optional runtime summaries into a bounded context string."""
    sections = [
        ("PROJECT SUMMARY", project_summary),
        ("DATA SUMMARY", data_summary),
        ("FORECAST SUMMARY", forecast_summary),
        ("MODEL EVALUATION", model_summary),
        ("PASSENGER CROWD / QUEUE SUMMARY", crowd_summary),
    ]
    output = []
    for heading, value in sections:
        value = str(value or "").strip()
        if value:
            output.append(f"{heading}:\n{value[:6000]}")
    if not output:
        return (
            "No live project statistics or uploaded project files were supplied "
            "with this question. Answer general project questions, but do not "
            "invent current project metrics."
        )
    return "\n\n".join(output)


def read_text_if_exists(path: str | Path, max_chars: int = 5000) -> str:
    """Read a small text/CSV excerpt if a caller explicitly supplies a path."""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""