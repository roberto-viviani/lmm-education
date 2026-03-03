import streamlit as st
import pandas as pd
from data import load_messages
from translations import TRANSLATIONS

def get_text(key):
    """Get translated text for current language"""
    lang = st.session_state.get("language", "de")
    return TRANSLATIONS.get(lang, {}).get(key, key)

st.set_page_config(page_title="Session Explorer", layout="wide")

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

st.title("💬 Session Explorer")

# Load data
df = load_messages()

# Get all sessions sorted by newest first
@st.cache_data
def get_all_sessions():
    sessions = []
    for session_hash in df["session_hash"].unique():
        session_data = df[df["session_hash"] == session_hash].sort_values("timestamp")
        sessions.append({
            "hash": session_hash,
            "messages": len(session_data),
            "model": session_data["model_name"].mode()[0] if len(session_data["model_name"].mode()) > 0 else session_data["model_name"].iloc[0],
            "start_time": session_data["timestamp"].min(),
            "end_time": session_data["timestamp"].max(),
            "data": session_data
        })
    return sorted(sessions, key=lambda x: x["start_time"], reverse=True)

all_sessions = get_all_sessions()

# Sidebar filters

filtered_sessions = all_sessions.copy()

# Display sessions
st.divider()

# Sessions header with sort filter
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Sessions ({len(filtered_sessions)})")

with col2:
    # Sort option
    sort_option = st.selectbox(get_text("sort_by"), 
        [get_text("newest_first"), get_text("oldest_first"), get_text("most_messages"), get_text("least_messages")],
        label_visibility="collapsed")

# Apply sorting
if sort_option == get_text("oldest_first"):
    filtered_sessions = sorted(filtered_sessions, key=lambda x: x["start_time"], reverse=False)
elif sort_option == get_text("most_messages"):
    filtered_sessions = sorted(filtered_sessions, key=lambda x: x["messages"], reverse=True)
elif sort_option == get_text("least_messages"):
    filtered_sessions = sorted(filtered_sessions, key=lambda x: x["messages"], reverse=False)
elif sort_option == get_text("most_messages"):
    filtered_sessions = sorted(filtered_sessions, key=lambda x: x["messages"], reverse=True)
elif sort_option == get_text("least_messages"):
    filtered_sessions = sorted(filtered_sessions, key=lambda x: x["messages"], reverse=False)

if len(filtered_sessions) == 0:
    st.warning(get_text("no_sessions_found"))
else:
    for idx, session in enumerate(filtered_sessions):
        with st.expander(f"Session {idx+1} | {session['start_time'].strftime('%Y-%m-%d %H:%M')} | 📝 {session['messages']} | 🤖 {session['model']}"):
            
            # Session info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(get_text("messages"), session['messages'])
            with col2:
                st.metric(get_text("model"), session['model'])
            with col3:
                duration = (session['end_time'] - session['start_time']).total_seconds() / 60
                st.metric(get_text("duration"), f"{max(int(duration), 1)}m")
            
            st.divider()
            
            # Q&A pairs
            st.markdown(f"**{get_text('question_answer_pairs')}**")
            
            session_df = session['data']
            
            for pair_idx, (_, row) in enumerate(session_df.iterrows(), 1):
                col_q, col_a = st.columns(2)
                
                with col_q:
                    st.markdown(f"**Q{pair_idx}** [{row['timestamp'].strftime('%H:%M')}]")
                    st.write(row["query"])
                
                with col_a:
                    st.markdown(f"**A{pair_idx}**")
                    st.write(row["response"])
                
                st.markdown("---")

