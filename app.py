import streamlit as st

# Настройка страницы
st.set_page_config(page_title="Страхование Workers Comp", layout="wide")

# Страницы
analysis_page = st.Page("analysis_and_model.py", title="Анализ и модель", icon="📊")
presentation_page = st.Page("presentation.py", title="Презентация", icon="🎯")

# Навигация
pg = st.navigation([analysis_page, presentation_page])
pg.run()
