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

# CSS COMPLETO
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a0f1f 0%, #030614 100%);
        position: relative;
    }
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1425 0%, #070b1a 100%) !important;
        border-right: 2px solid #00ffff;
        box-shadow: 5px 0 30px rgba(0, 255, 255, 0.2);
    }
    
    .sidebar-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #00ffff;
        text-align: center;
        padding: 15px;
        margin: 15px 0;
        border: 2px solid #00ffff;
        border-radius: 10px;
        background: linear-gradient(135deg, #0f1425, #1a1f35);
        text-shadow: 0 0 10px #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        animation: borderGlow 2s infinite;
    }
    
    @keyframes borderGlow {
        0% { border-color: #00ffff; box-shadow: 0 0 20px rgba(0, 255, 255, 0.3); }
        50% { border-color: #ff00ff; box-shadow: 0 0 30px rgba(255, 0, 255, 0.5); }
        100% { border-color: #00ffff; box-shadow: 0 0 20px rgba(0, 255, 255, 0.3); }
    }
    
    .sci-card {
        background: linear-gradient(135deg, #1a1f35, #0f1425);
        border-radius: 15px;
        padding: 25px;
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, #00ffff, #ff00ff) 1;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 30px rgba(0, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    .sci-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0, 255, 255, 0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .sci-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.6), 0 0 50px rgba(0, 255, 255, 0.4);
    }
    
    .sci-card .label {
        color: #8a8f9c;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
    }
    
    .sci-card .value {
        font-family: 'Source Code Pro', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ffff;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        margin: 5px 0;
    }
    
    .sci-card .delta {
        font-size: 0.9rem;
        color: #ff00ff;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #330000, #1a0000);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #ff4444;
        text-align: center;
        animation: alertPulse 1.5s infinite;
        box-shadow: 0 0 30px rgba(255, 68, 68, 0.3);
        margin-bottom: 15px;
    }
    
    @keyframes alertPulse {
        0% { border-color: #ff4444; box-shadow: 0 0 30px rgba(255, 68, 68, 0.3); }
        50% { border-color: #ff8888; box-shadow: 0 0 50px rgba(255, 68, 68, 0.6); }
        100% { border-color: #ff4444; box-shadow: 0 0 30px rgba(255, 68, 68, 0.3); }
    }
    
    .warning-card .label {
        color: #ff8888;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    .warning-card .value {
        font-family: 'Source Code Pro', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ff4444;
        text-shadow: 0 0 20px rgba(255, 68, 68, 0.5);
    }
    
    .zone-card {
        background: linear-gradient(135deg, #1a1f35, #0f1425);
        border-radius: 12px;
        padding: 18px;
        margin: 8px 0;
        border-left: 6px solid;
        border-right: 1px solid rgba(0, 255, 255, 0.2);
        border-top: 1px solid rgba(0, 255, 255, 0.2);
        border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .zone-card:hover {
        transform: translateX(10px);
        box-shadow: -10px 10px 30px rgba(0,0,0,0.5);
        border-right-color: #00ffff;
    }
    
    .zone-card .zone-name {
        font-size: 1rem;
        color: #8a8f9c;
        text-transform: uppercase;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    .zone-card .zone-value {
        font-family: 'Source Code Pro', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: white;
    }
    
    .zone-card .zone-count {
        font-size: 1.1rem;
        color: #00ffff;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(26, 31, 53, 0.8);
        backdrop-filter: blur(10px);
        padding: 8px;
        border-radius: 15px;
        border: 2px solid #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 25px;
        font-weight: 600;
        color: #8a8f9c !important;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ffff, #ff00ff) !important;
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ffff, #ff00ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1rem;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 0 40px rgba(255, 0, 255, 0.5);
        border-color: white;
    }
    
    .stat-container {
        background: linear-gradient(135deg, #1a1f35, #0f1425);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #00ffff;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    
    .stat-container h4 {
        color: #00ffff;
        font-size: 1.1rem;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-container p {
        color: #e2e8f0;
        margin: 8px 0;
    }
    
    .stat-container strong {
        color: #ff00ff;
    }
    
    .note-card {
        background: linear-gradient(135deg, #1a1f35, #0f1425);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #00ffff;
        border-right: 1px solid rgba(0, 255, 255, 0.2);
        border-top: 1px solid rgba(0, 255, 255, 0.2);
        border-bottom: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    .note-card .note-date {
        color: #8a8f9c;
        font-size: 0.85rem;
    }
    
    .note-card .note-text {
        color: white;
        margin: 5px 0;
    }
    
    h1 {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 0 30px #00ffff, 0 0 60px #ff00ff;
        animation: titleGlow 3s infinite;
    }
    
    @keyframes titleGlow {
        0% { text-shadow: 0 0 30px #00ffff, 0 0 60px #ff00ff; }
        50% { text-shadow: 0 0 50px #ff00ff, 0 0 80px #00ffff; }
        100% { text-shadow: 0 0 30px #00ffff, 0 0 60px #ff00ff; }
    }
    
    h2 {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #00ffff, #ff00ff) 1;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }
    
    h3 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #00ffff;
        margin-bottom: 20px;
    }
    
    .dataframe {
        background: #1a1f35 !important;
        border-radius: 15px !important;
        border: 2px solid #00ffff !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #00ffff, #ff00ff) !important;
        color: white !important;
        font-weight: 600;
        padding: 15px !important;
        font-size: 0.95rem;
    }
    
    .dataframe td {
        background: #0f1425 !important;
        color: #e2e8f0 !important;
        border-color: #00ffff !important;
        padding: 12px !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff) !important;
        border-radius: 10px;
        border: 1px solid white;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f1425;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00ffff, #ff00ff);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff00ff, #00ffff);
    }
</style>
""", unsafe_allow_html=True)

st.write("✅ CSS carregado")

# ============================================================================
# INICIALIZAÇÃO DO SESSION STATE
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

# ============================================================================
# TRADUÇÕES
# ============================================================================

translations = {
    'pt': {
        'upload': 'Upload de Dados',
        'variable': 'Variável',
        'position': 'Posição',
        'period': 'Período',
        'athlete': 'Atleta',
        'process': '⚡ PROCESSAR ANÁLISE',
        'tip_text': 'Múltiplos arquivos CSV'
    }
}

st.write("✅ Traduções carregadas")

# ============================================================================
# FUNÇÕES
# ============================================================================

def parse_identificacao(series):
    nomes = []
    periodos = []
    minutos = []
    for valor in series:
        try:
            texto = str(valor).strip()
            partes = texto.split('-')
            if len(partes) >= 3:
                nome = partes[0].strip()
                periodo = '-'.join(partes[1:-1]).strip()
                minuto = partes[-1].strip()
            else:
                nome = texto
                periodo = ''
                minuto = ''
        except:
            nome = ''
            periodo = ''
            minuto = ''
        nomes.append(nome)
        periodos.append(periodo)
        minutos.append(minuto)
    return nomes, periodos, minutos

def verificar_estruturas_arquivos(dataframes):
    if not dataframes:
        return False, []
    primeira_estrutura = dataframes[0].columns.tolist()
    for df in dataframes[1:]:
        if df.columns.tolist() != primeira_estrutura:
            return False, primeira_estrutura
    return True, primeira_estrutura

def processar_upload(files):
    if not files:
        return None, [], [], [], []
    dataframes = []
    nomes_arquivos = []
    for file in files:
        try:
            df = pd.read_csv(file)
            if df.shape[1] >= 3:
                dataframes.append(df)
                nomes_arquivos.append(file.name)
        except:
            continue
    if not dataframes:
        return None, [], [], [], []
    estruturas_ok, _ = verificar_estruturas_arquivos(dataframes)
    if not estruturas_ok:
        return None, [], [], [], []
    df_concatenado = pd.concat(dataframes, ignore_index=True)
    primeira_coluna = df_concatenado.iloc[:, 0].astype(str)
    nomes, periodos, minutos = parse_identificacao(primeira_coluna)
    posicoes = df_concatenado.iloc[:, 1].astype(str)
    variaveis_quant = []
    dados_quant = {}
    for col_idx in range(2, df_concatenado.shape[1]):
        nome_var = df_concatenado.columns[col_idx]
        valores = pd.to_numeric(df_concatenado.iloc[:, col_idx], errors='coerce')
        if not valores.dropna().empty:
            variaveis_quant.append(nome_var)
            dados_quant[nome_var] = valores.values
    if not variaveis_quant:
        return None, [], [], [], []
    df_estruturado = pd.DataFrame({
        'Nome': nomes,
        'Posição': posicoes,
        'Período': periodos,
        'Minuto': minutos
    })
    for var_nome, var_valores in dados_quant.items():
        df_estruturado[var_nome] = var_valores
    df_estruturado = df_estruturado[
        (df_estruturado['Nome'].str.len() > 0) & 
        (df_estruturado['Posição'].str.len() > 0)
    ].reset_index(drop=True)
    if df_estruturado.empty:
        return None, [], [], [], []
    periodos_unicos = sorted([p for p in df_estruturado['Período'].unique() if p and p.strip()])
    posicoes_unicas = sorted([p for p in df_estruturado['Posição'].unique() if p and p.strip()])
    return df_estruturado, variaveis_quant, periodos_unicos, posicoes_unicas, nomes_arquivos

st.write("✅ Funções carregadas")

# ============================================================================
# SIDEBAR - VERSÃO SIMPLIFICADA PARA TESTE
# ============================================================================

with st.sidebar:
    st.markdown("### 🌐 IDIOMA")
    
    idiomas = ['pt', 'en', 'es']
    idx_idioma = idiomas.index(st.session_state.idioma) if st.session_state.idioma in idiomas else 0
    idioma = st.selectbox("Selecione o idioma", idiomas, index=idx_idioma)
    
    if idioma != st.session_state.idioma:
        st.session_state.idioma = idioma
        st.rerun()
    
    t = translations['pt']
    
    st.markdown("---")
    st.markdown(f"### 📂 {t['upload']}")
    
    uploaded_files = st.file_uploader(
        "Upload de arquivos CSV",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Arquivos selecionados: {len(uploaded_files)}")

st.write("✅ Sidebar carregada")
st.write("### App carregou corretamente!")