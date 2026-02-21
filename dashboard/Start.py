import streamlit as st
from translations import TRANSLATIONS

def get_text(key):
    """Get translated text for current language"""
    lang = st.session_state.get("language", "de")
    return TRANSLATIONS.get(lang, {}).get(key, key)

st.set_page_config(page_title="Welcome", layout="wide")

# Minimalist CSS
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

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

# Clean Header
st.markdown(f"""
# {get_text('welcome_title')}

{get_text('welcome_subtitle')}
""")

st.markdown("---")

# Main Content - Two Column Layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown(f"""
## {get_text('welcome_greeting')}

{get_text('welcome_description')}
    """)

with col2:
    st.info(f"💡 {get_text('tip')}")

st.markdown("---")


