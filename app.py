"""envelope website. Run: streamlit run app.py

Pages (top navigation): Home, Assess your model (upload your own model's outputs),
Interactive example, Guided tour (five-stage walkthrough, also for screen recordings), API.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="envelope", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

pages = [
    st.Page("site/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("site/assess.py", title="Assess your model", icon=":material/upload_file:", url_path="assess"),
    st.Page("site/example.py", title="Interactive example", icon=":material/tune:", url_path="example"),
    st.Page("site/tour.py", title="Guided tour", icon=":material/slideshow:", url_path="tour"),
    st.Page("site/api_docs.py", title="API", icon=":material/api:", url_path="api"),
]
st.navigation(pages, position="top").run()
