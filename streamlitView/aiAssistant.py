
import streamlit as st

from assist_ai.script import render_assistant

# Streamlit page configuration
st.set_page_config(
    page_title="Ferry AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Display AI Assistant
render_assistant()
