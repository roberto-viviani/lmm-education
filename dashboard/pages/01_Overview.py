import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data import load_messages
import pandas as pd
from datetime import timedelta
from translations import TRANSLATIONS

def get_text(key):
    """Get translated text for current language"""
    lang = st.session_state.get("language", "de")
    return TRANSLATIONS.get(lang, {}).get(key, key)

st.set_page_config(page_title="Overview", layout="wide")

# Language selector with buttons (centered, top right)
col1, col2, col3 = st.columns([9, 1.2, 1.2])
with col2:
    if st.button("DE", key="btn_de", use_container_width=True):
        st.session_state.language = "de"
        st.rerun()
with col3:
    if st.button("EN", key="btn_en", use_container_width=True):
        st.session_state.language = "en"
        st.rerun()

st.title(get_text("overview_title"))

df = load_messages()

min_date = df["date"].min()
max_date = df["date"].max()

# Filter options
col1, col2 = st.columns([3, 1])
with col1:
    period = st.selectbox(
        get_text("time_periods"),
        options=[
            get_text("last_day"),
            get_text("last_week"),
            get_text("last_month"),
            get_text("last_3_months"),
            get_text("last_6_months")
        ],
        index=2
    )

# Calculate start date based on selection
lang = st.session_state.get("language", "de")
if period == TRANSLATIONS[lang]["last_day"]:
    start = max_date - timedelta(days=1)
elif period == TRANSLATIONS[lang]["last_week"]:
    start = max_date - timedelta(weeks=1)
elif period == TRANSLATIONS[lang]["last_month"]:
    start = max_date - timedelta(days=30)
elif period == TRANSLATIONS[lang]["last_3_months"]:
    start = max_date - timedelta(days=90)
else:
    start = max_date - timedelta(days=180)

end = max_date

filtered = df[
    (df["date"] >= start) &
    (df["date"] <= end)
]

# KPIs
k1, k2, k4, k5 = st.columns(4)

total_questions = len(filtered)
unique_sessions = filtered["session_hash"].nunique()
avg_response_length = round(filtered["response_length"].mean(), 0)
avg_query_length = round(filtered["query"].astype(str).str.len().mean(), 0)

with k1:
    st.metric(get_text("total_questions"), f"{total_questions:,}")
    st.caption(get_text("questions_metric"))

with k2:
    st.metric(get_text("total_sessions"), f"{unique_sessions:,}")
    st.caption(get_text("sessions_metric"))


with k4:
    st.metric(get_text("avg_response"), f"{int(avg_response_length):,}")
    st.caption(get_text("response_length_metric"))

with k5:
    st.metric(get_text("avg_query"), f"{int(avg_query_length):,}")
    st.caption(get_text("avg_query_metric"))

st.divider()

# Questions over time
st.markdown(f"## {get_text('questions_over_time')}")
daily = filtered.groupby("date").size().reset_index(name="questions")
fig_time = px.line(
    daily, 
    x="date", 
    y="questions",
    markers=True,
    title=get_text("daily_volume"),
    labels={"date": get_text("date"), "questions": get_text("number_of_questions")}
)
fig_time.update_layout(hovermode="x unified", showlegend=False)
st.plotly_chart(fig_time, use_container_width=True)

st.divider()




# Hourly trends
st.markdown(f"## {get_text('hourly_activity')}")
filtered_copy = filtered.copy()
filtered_copy["hour"] = pd.to_datetime(filtered_copy["timestamp"]).dt.hour
hourly = filtered_copy.groupby("hour").size().reset_index(name="questions")
fig_hourly = px.bar(
    hourly,
    x="hour",
    y="questions",
    labels={"hour": get_text("hour_utc"), "questions": get_text("number_of_questions")}
)
fig_hourly.update_xaxes(type="category")
st.plotly_chart(fig_hourly, use_container_width=True)

st.divider()

