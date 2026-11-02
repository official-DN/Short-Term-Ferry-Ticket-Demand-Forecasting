"""
Streamlit multipage entry point for the Short-Term Ferry Forecasting project.

The navigation exposes:
- Forecasting Dashboard: existing forecasting application
- Passenger Crowd: queue, crowd pressure, and waiting-time scenario analysis
"""

import streamlit as st

st.set_page_config(
    page_title="Ferry Demand & Passenger Crowd Forecasting",
    page_icon="⛴️",
    layout="wide",
)

pages = [
    st.Page(
        "streamlitView/dashboard.py",
        title="Forecasting Dashboard",
        icon="⛴️",
        default=True,
    ),
    st.Page(
        "streamlitView/passengerCrowd.py",
        title="Passenger Crowd",
        icon="👥",
    ),
]

navigation = st.navigation(pages)
navigation.run()
