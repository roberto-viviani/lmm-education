import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data
def load_messages():
    base_path = Path(__file__).resolve().parent.parent

    messages_path = base_path / "messages.csv"

    if not messages_path.exists():
        st.error(f"messages.csv nicht gefunden unter: {messages_path}")
        st.stop()

    df = pd.read_csv(messages_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["response_length"] = df["response"].astype(str).str.len()
    df["query_norm"] = (
        df["query"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df
