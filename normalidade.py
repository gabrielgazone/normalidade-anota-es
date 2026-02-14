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

# CSS básico para teste
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

# ============================================================================
# INICIALIZAÇÃO DO SESSION STATE - SIMPLIFICADA PARA TESTE
# ============================================================================

if 'df_completo' not in st.session_state:
    st.session_state.df_completo = None
if 'variaveis_quantitativas' not in st.session_state:
    st.session_state.variaveis_quantitativas = []
if 'variavel_selecionada' not in st.session_state:
    st.session_state.variavel_selecionada = None
if 'atletas_selecionados' not in st.session_state:
    st.session_state.atletas_selecionados = []
if 'posicoes_selecionadas' not in st.session_state:
    st.session_state.posicoes_selecionadas = []
if 'todos_posicoes' not in st.session_state:
    st.session_state.todos_posicoes = []
if 'periodos_selecionados' not in st.session_state:
    st.session_state.periodos_selecionados = []
if 'todos_periodos' not in st.session_state:
    st.session_state.todos_periodos = []
if 'processar_click' not in st.session_state:
    st.session_state.processar_click = False
if 'idioma' not in st.session_state:
    st.session_state.idioma = 'pt'

st.write("✅ Session state inicializado")
st.write("### App carregou corretamente!")
st.write("Session state:", dict(st.session_state))