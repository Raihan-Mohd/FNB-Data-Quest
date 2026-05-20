"""
DataQuest 2026 - Interpretable Credit Models
Main entry point. Sidebar navigation routes to view modules.
"""
import streamlit as st

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="DataQuest 2026 | Credit Modelling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit's modern navigation API (requires Streamlit >= 1.36)
home = st.Page("views/home.py", title="Home", icon="🏠", default=True)
research = st.Page("views/research.py", title="Research", icon="📚")
data_quality = st.Page("views/data_quality.py", title="Data Quality", icon="🔍")
univariate = st.Page("views/univariate.py", title="Univariate Explorer", icon="📈")
bivariate = st.Page("views/bivariate.py", title="Bivariate Explorer", icon="🔗")
modelling = st.Page("views/modelling.py", title="Modelling", icon="🧮")
business = st.Page("views/business.py", title="Business Decisions", icon="💼")
ai_log = st.Page("views/ai_log.py", title="AI Usage Log", icon="🤖")

pg = st.navigation(
    {
        "Overview": [home, research],
        "Exploration": [data_quality, univariate, bivariate],
        "Modelling & Strategy": [modelling, business],
        "Process": [ai_log],
    }
)

# Run the selected page
pg.run()
