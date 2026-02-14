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
# FUNÇÕES DE PROCESSAMENTO
# ============================================================================

def parse_identificacao(series):
    """Parseia a coluna de identificação"""
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
    """Verifica se todos os dataframes têm a mesma estrutura"""
    if not dataframes:
        return False, []
    
    primeira_estrutura = dataframes[0].columns.tolist()
    
    for i, df in enumerate(dataframes[1:], 1):
        if df.columns.tolist() != primeira_estrutura:
            return False, primeira_estrutura
    
    return True, primeira_estrutura

def processar_upload(files):
    """Processa múltiplos arquivos CSV"""
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
        except Exception as e:
            st.error(f"Erro ao ler {file.name}")
            continue
    
    if not dataframes:
        return None, [], [], [], []
    
    estruturas_ok, estrutura_base = verificar_estruturas_arquivos(dataframes)
    
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

st.write("✅ Funções de processamento carregadas")
st.write("### App carregou corretamente!")