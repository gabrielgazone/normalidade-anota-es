import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime
import time
import sys

st.set_page_config(
    page_title="⚡ SPORTS SCIENCE PRO | DARK EDITION",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("✅ Configuração da página OK")

# CSS básico para teste (versão simplificada)
st.markdown("""
<style>
    .stApp {
        background: #0a0f1f;
    }
    h1 {
        color: #00ffff;
    }
</style>
""", unsafe_allow_html=True)

st.write("✅ CSS carregado")
st.write("### App carregou corretamente!")