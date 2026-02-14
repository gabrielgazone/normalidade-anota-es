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

# TESTE CSS - BLOCO 1
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