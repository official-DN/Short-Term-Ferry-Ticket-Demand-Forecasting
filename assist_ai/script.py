"""Streamlit chat interface for the Ferry AI Assistant.

Or import `render_assistant` into an existing Streamlit page.
"""

from __future__ import annotations

import streamlit as st

try:
    from .connect import GeminiClient, GeminiConnectionError
    from .project_context import PROJECT_INSTRUCTIONS, build_project_context
except ImportError:  # supports `streamlit run assist_AI/script.py`
    from connect import GeminiClient, GeminiConnectionError
    from project_context import PROJECT_INSTRUCTIONS, build_project_context


SUGGESTED_QUESTIONS = {
    "Project Overview": [
        "What is the main objective of this project?",
        "Explain the complete project architecture.",
        "What problems does this system solve for ferry operations?",
        "Which technologies and machine-learning models are used?",
        "What is the difference between ticket-demand forecasting and crowd prediction?",
    ],
    "Data Analysis": [
        "Explain the historical ferry ticket dataset.",
        "What preprocessing steps are required?",
        "How should missing 15-minute or 30-minute intervals be handled?",
        "How can busy hours and daily demand patterns be identified?",
        "How can weather and holidays affect ticket demand?",
    ],
    "Forecasting Models": [
        "How does the system forecast demand for 15 minutes to 2 hours?",
        "Why compare Linear Regression, Random Forest and Gradient Boosting?",
        "How do I identify the best-performing forecasting model?",
        "Explain MAE, RMSE and MAPE with a ferry example.",
        "What is rolling-origin validation?",
    ],
    "Crowd and Queue": [
        "How can passenger arrivals be estimated from forecast ticket demand?",
        "Explain the queue prediction equation.",
        "How should Low, Moderate, High and Critical crowd levels be defined?",
        "How is passenger waiting time estimated?",
        "How can the system generate overcrowding alerts?",
    ],
    "Decision Support": [
        "What should operators do if forecast demand increases by 40%?",
        "What happens when arrivals exceed boarding capacity?",
        "How can forecasts help with ferry staffing and scheduling?",
        "Why are prediction intervals useful for operations?",
        "Analyze the available results and suggest improvements.",
    ],
}


def _history_for_api() -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.get("ferry_ai_history", [])
        if item.get("role") in {"user", "assistant"}
    ]


def render_assistant(
    project_summary: str = "",
    data_summary: str = "",
    forecast_summary: str = "",
    model_summary: str = "",
    crowd_summary: str = "",
) -> None:
    """Render the assistant inside an existing Streamlit page."""
    st.subheader("Ferry AI Assistant")
    st.caption(
        "Ask questions about forecasting, model evaluation, passenger arrivals, "
        "queues, crowd levels and operational recommendations."
    )

    if "ferry_ai_history" not in st.session_state:
        st.session_state.ferry_ai_history = []

    context = build_project_context(
        project_summary=project_summary,
        data_summary=data_summary,
        forecast_summary=forecast_summary,
        model_summary=model_summary,
        crowd_summary=crowd_summary,
    )
    system_instruction = f"{PROJECT_INSTRUCTIONS}\n\nPROJECT CONTEXT:\n{context}"

    with st.expander("Suggested questions", expanded=False):
        for category, questions in SUGGESTED_QUESTIONS.items():
            st.markdown(f"**{category}**")
            cols = st.columns(2)
            for index, question in enumerate(questions):
                if cols[index % 2].button(
                    question,
                    key=f"ferry_ai_{category}_{index}",
                    use_container_width=True,
                ):
                    st.session_state.ferry_ai_pending_question = question

    for message in st.session_state.ferry_ai_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input(
        "Ask FerryAI about your project, models, results or crowd analysis..."
    )
    question = typed_question or st.session_state.pop(
        "ferry_ai_pending_question", None
    )

    if question:
        st.session_state.ferry_ai_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("FerryAI is preparing an answer..."):
                try:
                    client = GeminiClient()
                    try:
                        answer = client.generate(
                            prompt=question,
                            system_instruction=system_instruction,
                            history=_history_for_api()[:-1],
                        )
                    finally:
                        client.close()
                    st.markdown(answer)
                    st.session_state.ferry_ai_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except GeminiConnectionError as exc:
                    st.error(str(exc))
                    st.info(
                        "Check GEMINI_API_KEY, the selected model, internet "
                        "access, and your Google AI Studio quota/billing status."
                    )

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.ferry_ai_history = []
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Ferry AI Assistant",
        page_icon="⛴️",
        layout="wide",
    )
    render_assistant()