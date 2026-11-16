"""Streamlit entry point.

    streamlit run src/frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import os  # noqa: E402

import streamlit as st  # noqa: E402

# On Streamlit Community Cloud, keys are entered as app secrets. Copy the top-level ones into
# the environment so the embedded API and the Refresh subprocess read them like a .env file.
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)) and key not in os.environ:
            os.environ[key] = str(value)
except Exception:  # no secrets file: local development uses .env
    pass

from src.frontend.components import api, ui  # noqa: E402

st.set_page_config(page_title="Retail Investor Intelligence", page_icon="📊", layout="wide")
ui.apply_style()

PAGES = Path(__file__).parent / "pages"
navigation = st.navigation({
    "Summary": [
        st.Page(PAGES / "briefing.py", title="Executive briefing", icon="📋", default=True),
    ],
    "Explore": [
        st.Page(PAGES / "overview.py", title="Sentiment and platforms", icon="📊"),
        st.Page(PAGES / "topics.py", title="Topics and aspects", icon="🧭"),
        st.Page(PAGES / "explorer.py", title="Post explorer", icon="🔎"),
        st.Page(PAGES / "assistant.py", title="Ask the data", icon="💬"),
    ],
    "About": [
        st.Page(PAGES / "method.py", title="Method and reliability", icon="✅"),
    ],
})

api.ensure_backend()
try:
    st.session_state["scope"] = ui.sidebar_filters()
except api.ApiError as exc:
    st.error(str(exc))
    st.info("The analytics service is not reachable. Check the database settings and try again shortly.")
    st.stop()

with st.sidebar:
    st.divider()
    ui.refresh_panel()
    st.divider()
    st.caption("Describes public conversation only. Not investment advice, and not a "
               "prediction of share prices.")
ui.show_refresh_toast()

navigation.run()
