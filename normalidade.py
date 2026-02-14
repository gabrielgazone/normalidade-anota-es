import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÃO DA PÁGINA - PRIMEIRA INSTRUÇÃO
st.set_page_config(
    page_title="⚡ SPORTS SCIENCE PRO | DARK EDITION",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INICIALIZAÇÃO DO SESSION STATE - COMPLETA
# ============================================================================

def init_session_state():
    """Inicializa todas as variáveis de sessão"""
    defaults = {
        'df_completo': None,
        'variaveis_quantitativas': [],
        'variavel_selecionada': None,
        'atletas_selecionados': [],
        'posicoes_selecionadas': [],
        'todos_posicoes': [],
        'periodos_selecionados': [],
        'todos_periodos': [],
        'ordem_personalizada': [],
        'upload_files_names': [],
        'idioma': 'pt',
        'processar_click': False,
        'dados_processados': False,
        'metodo_zona': 'percentis',
        'grupo1': None,
        'grupo2': None,
        'zona_key': 0,
        'anotacoes': [],
        'n_classes': 5,
        'upload_concluido': False,
        'atleta1_comp': None,
        'atleta2_comp': None,
        'vars_comp': [],
        'window_size': 3,
        'show_critical_markers': True,
        'show_moving_average': True,
        'confidence_level': 0.95,
        'gamification_level': 1,
        'xp_points': 0,
        'achievements': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# CSS DARK INOVADOR COM TOQUE DE GAMIFICAÇÃO
# ============================================================================

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
    
    /* Background escuro com efeito de grade */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a0f1f 0%, #030614 100%);
        position: relative;
    }
    
    /* Efeito de grid futurista */
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
    
    /* Sidebar dark futurista */
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
    
    /* Cards científicos com efeito neon */
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
    
    /* Warning card com efeito de alerta */
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
    
    /* Zone cards neon */
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
    
    /* Tabs neon */
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
    
    /* Botões neon */
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
    
    /* Containers de estatísticas */
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
    
    /* Note cards */
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
    
    /* Headers */
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
    
    /* Dataframe */
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
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff) !important;
        border-radius: 10px;
        border: 1px solid white;
    }
    
    /* Scrollbar */
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

# ============================================================================
# HEADER CIENTÍFICO
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>⚡ SPORTS SCIENCE ANALYTICS PRO</h1>
        <p style="color: #00ffff; font-size: 1.2rem; letter-spacing: 3px; font-family: 'Orbitron', sans-serif;">
            QUANTUM NEURAL PERFORMANCE ANALYZER
        </p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px;">
            <span style="background: linear-gradient(135deg, #00ffff20, #ff00ff20); color: #00ffff; padding: 8px 20px; border-radius: 20px; border: 1px solid #00ffff;">⚡ NEURAL CORE</span>
            <span style="background: linear-gradient(135deg, #ff00ff20, #00ffff20); color: #ff00ff; padding: 8px 20px; border-radius: 20px; border: 1px solid #ff00ff;">🔬 QUANTUM LAB</span>
            <span style="background: linear-gradient(135deg, #00ffff20, #ff00ff20); color: #00ffff; padding: 8px 20px; border-radius: 20px; border: 1px solid #00ffff;">📊 DEEP ANALYTICS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sistema de gamificação sutil
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="sci-card" style="padding: 10px;">
        <p class="label">🎮 LEVEL</p>
        <p class="value" style="font-size: 1.5rem;">{st.session_state.gamification_level}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    xp_percent = st.session_state.xp_points % 100
    st.markdown(f"""
    <div class="sci-card" style="padding: 10px;">
        <p class="label">⚡ XP</p>
        <div style="background: #0f1425; border-radius: 10px; height: 20px; margin: 10px 0; border: 1px solid #00ffff;">
            <div style="width: {xp_percent}%; background: linear-gradient(90deg, #00ffff, #ff00ff); height: 100%; border-radius: 10px;"></div>
        </div>
        <p class="value" style="font-size: 1rem;">{st.session_state.xp_points} / 100</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="sci-card" style="padding: 10px;">
        <p class="label">🏆 ACHIEVEMENTS</p>
        <p class="value" style="font-size: 1.5rem;">{len(st.session_state.achievements)}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# INTERNACIONALIZAÇÃO COMPLETA
# ============================================================================

translations = {
    'pt': {
        'title': 'Sports Science Analytics Pro',
        'upload': 'Upload de Dados',
        'variable': 'Variável',
        'position': 'Posição',
        'period': 'Período',
        'athlete': 'Atleta',
        'config': 'Configurações',
        'tab_distribution': '📊 Distribuição',
        'tab_temporal': '📈 Análise Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlações',
        'tab_comparison': '⚖️ Comparações',
        'tab_executive': '📋 Executivo',
        'mean': 'Média',
        'median': 'Mediana',
        'mode': 'Moda',
        'std': 'Desvio Padrão',
        'variance': 'Variância',
        'cv': 'CV',
        'min': 'Mínimo',
        'max': 'Máximo',
        'amplitude': 'Amplitude',
        'q1': 'Q1 (25%)',
        'q3': 'Q3 (75%)',
        'iqr': 'IQR',
        'skewness': 'Assimetria',
        'kurtosis': 'Curtose',
        'max_value': 'MÁXIMO',
        'min_value': 'MÍNIMO',
        'minute_of_max': 'Minuto do Máx',
        'minute_of_min': 'Minuto do Mín',
        'threshold_80': 'LIMIAR 80%',
        'critical_events': 'EVENTOS CRÍTICOS',
        'above_threshold': 'acima do limiar',
        'intensity_zones': 'Zonas de Intensidade',
        'zone_method': 'Método',
        'percentiles': 'Percentis',
        'based_on_max': 'Baseado no Máximo',
        'very_low': 'Muito Baixa',
        'low': 'Baixa',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muito Alta',
        'process': '⚡ PROCESSAR ANÁLISE',
        'descriptive_stats': '📊 Estatísticas Descritivas',
        'confidence_interval': '🎯 Intervalo de Confiança',
        'normality_test': '🧪 Teste de Normalidade',
        'summary_by_group': '🏃 Resumo por Grupo',
        'symmetric': 'Simétrica',
        'moderate_skew': 'Assimetria Moderada',
        'high_skew': 'Assimetria Forte',
        'leptokurtic': 'Leptocúrtica',
        'platykurtic': 'Platicúrtica',
        'mesokurtic': 'Mesocúrtica',
        'strong_positive': 'Correlação Forte Positiva',
        'moderate_positive': 'Correlação Moderada Positiva',
        'weak_positive': 'Correlação Fraca Positiva',
        'very_weak_positive': 'Correlação Muito Fraca Positiva',
        'very_weak_negative': 'Correlação Muito Fraca Negativa',
        'weak_negative': 'Correlação Fraca Negativa',
        'moderate_negative': 'Correlação Moderada Negativa',
        'strong_negative': 'Correlação Forte Negativa',
        'iqr_title': '📌 IQR',
        'iqr_explanation': 'Intervalo Interquartil (Q3 - Q1)',
        'step1': '👈 **Passo 1:** Faça upload dos dados CSV',
        'step2': '👈 **Passo 2:** Selecione os filtros e processe',
        'file_format': '### 📋 Formato do Arquivo',
        'components': '📌 Componentes',
        'name_ex': 'Nome: João, Maria...',
        'period_ex': 'Período: 1º Tempo...',
        'minute_ex': 'Minuto: 00:00-01:00...',
        'position_ex': 'Posição: Atacante...',
        'tip': '💡 Dica',
        'tip_text': 'Múltiplos arquivos CSV',
        'multi_file_ex': '📁 Múltiplos Arquivos',
        'moving_average': 'Média Móvel',
        'window_size': 'Janela',
        'critical_markers': 'Marcadores Críticos',
        'trend_analysis': 'Análise de Tendência',
        'observations': 'Observações',
        'r_squared': 'R²',
        'gain_xp': 'Ganhou +5 XP!',
        'achievement_unlocked': '🏆 Conquista Desbloqueada!',
        'first_analysis': 'Primeira Análise',
        'critical_finder': 'Caçador de Críticos',
        'master_analyst': 'Mestre Analista'
    },
    'en': {
        'title': 'Sports Science Analytics Pro',
        'upload': 'Data Upload',
        'variable': 'Variable',
        'position': 'Position',
        'period': 'Period',
        'athlete': 'Athlete',
        'config': 'Settings',
        'tab_distribution': '📊 Distribution',
        'tab_temporal': '📈 Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlations',
        'tab_comparison': '⚖️ Comparisons',
        'tab_executive': '📋 Executive',
        'mean': 'Mean',
        'median': 'Median',
        'mode': 'Mode',
        'std': 'Std Dev',
        'variance': 'Variance',
        'cv': 'CV',
        'min': 'Minimum',
        'max': 'Maximum',
        'amplitude': 'Range',
        'q1': 'Q1 (25%)',
        'q3': 'Q3 (75%)',
        'iqr': 'IQR',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis',
        'max_value': 'MAXIMUM',
        'min_value': 'MINIMUM',
        'minute_of_max': 'Max Minute',
        'minute_of_min': 'Min Minute',
        'threshold_80': '80% THRESHOLD',
        'critical_events': 'CRITICAL EVENTS',
        'above_threshold': 'above threshold',
        'intensity_zones': 'Intensity Zones',
        'zone_method': 'Method',
        'percentiles': 'Percentiles',
        'based_on_max': 'Based on Max',
        'very_low': 'Very Low',
        'low': 'Low',
        'moderate': 'Moderate',
        'high': 'High',
        'very_high': 'Very High',
        'process': '⚡ PROCESS ANALYSIS',
        'descriptive_stats': '📊 Descriptive Statistics',
        'confidence_interval': '🎯 Confidence Interval',
        'normality_test': '🧪 Normality Test',
        'summary_by_group': '🏃 Summary by Group',
        'symmetric': 'Symmetric',
        'moderate_skew': 'Moderate Skew',
        'high_skew': 'High Skew',
        'leptokurtic': 'Leptokurtic',
        'platykurtic': 'Platykurtic',
        'mesokurtic': 'Mesokurtic',
        'strong_positive': 'Strong Positive',
        'moderate_positive': 'Moderate Positive',
        'weak_positive': 'Weak Positive',
        'very_weak_positive': 'Very Weak Positive',
        'very_weak_negative': 'Very Weak Negative',
        'weak_negative': 'Weak Negative',
        'moderate_negative': 'Moderate Negative',
        'strong_negative': 'Strong Negative',
        'iqr_title': '📌 IQR',
        'iqr_explanation': 'Interquartile Range (Q3 - Q1)',
        'step1': '👈 **Step 1:** Upload CSV data',
        'step2': '👈 **Step 2:** Select filters and process',
        'file_format': '### 📋 File Format',
        'components': '📌 Components',
        'name_ex': 'Name: Joao, Maria...',
        'period_ex': 'Period: 1st Half...',
        'minute_ex': 'Minute: 00:00-01:00...',
        'position_ex': 'Position: Forward...',
        'tip': '💡 Tip',
        'tip_text': 'Multiple CSV files',
        'multi_file_ex': '📁 Multiple Files',
        'moving_average': 'Moving Average',
        'window_size': 'Window',
        'critical_markers': 'Critical Markers',
        'trend_analysis': 'Trend Analysis',
        'observations': 'Observations',
        'r_squared': 'R²',
        'gain_xp': 'Earned +5 XP!',
        'achievement_unlocked': '🏆 Achievement Unlocked!',
        'first_analysis': 'First Analysis',
        'critical_finder': 'Critical Hunter',
        'master_analyst': 'Master Analyst'
    },
    'es': {
        'title': 'Sports Science Analytics Pro',
        'upload': 'Carga de Datos',
        'variable': 'Variable',
        'position': 'Posición',
        'period': 'Período',
        'athlete': 'Atleta',
        'config': 'Configuración',
        'tab_distribution': '📊 Distribución',
        'tab_temporal': '📈 Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlaciones',
        'tab_comparison': '⚖️ Comparaciones',
        'tab_executive': '📋 Ejecutivo',
        'mean': 'Media',
        'median': 'Mediana',
        'mode': 'Moda',
        'std': 'Desviación Estándar',
        'variance': 'Varianza',
        'cv': 'CV',
        'min': 'Mínimo',
        'max': 'Máximo',
        'amplitude': 'Amplitud',
        'q1': 'Q1 (25%)',
        'q3': 'Q3 (75%)',
        'iqr': 'IQR',
        'skewness': 'Asimetría',
        'kurtosis': 'Curtosis',
        'max_value': 'MÁXIMO',
        'min_value': 'MÍNIMO',
        'minute_of_max': 'Minuto del Máx',
        'minute_of_min': 'Minuto del Mín',
        'threshold_80': 'UMBRAL 80%',
        'critical_events': 'EVENTOS CRÍTICOS',
        'above_threshold': 'sobre el umbral',
        'intensity_zones': 'Zonas de Intensidad',
        'zone_method': 'Método',
        'percentiles': 'Percentiles',
        'based_on_max': 'Basado en Máximo',
        'very_low': 'Muy Baja',
        'low': 'Baja',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muy Alta',
        'process': '⚡ PROCESAR ANÁLISIS',
        'descriptive_stats': '📊 Estadísticas Descriptivas',
        'confidence_interval': '🎯 Intervalo de Confianza',
        'normality_test': '🧪 Prueba de Normalidad',
        'summary_by_group': '🏃 Resumen por Grupo',
        'symmetric': 'Simétrica',
        'moderate_skew': 'Asimetría Moderada',
        'high_skew': 'Asimetría Fuerte',
        'leptokurtic': 'Leptocúrtica',
        'platykurtic': 'Platicúrtica',
        'mesokurtic': 'Mesocúrtica',
        'strong_positive': 'Correlación Fuerte Positiva',
        'moderate_positive': 'Correlación Moderada Positiva',
        'weak_positive': 'Correlación Débil Positiva',
        'very_weak_positive': 'Correlación Muy Débil Positiva',
        'very_weak_negative': 'Correlación Muy Débil Negativa',
        'weak_negative': 'Correlación Débil Negativa',
        'moderate_negative': 'Correlación Moderada Negativa',
        'strong_negative': 'Correlación Fuerte Negativa',
        'iqr_title': '📌 IQR',
        'iqr_explanation': 'Rango Intercuartil (Q3 - Q1)',
        'step1': '👈 **Paso 1:** Cargue datos CSV',
        'step2': '👈 **Paso 2:** Seleccione filtros y procese',
        'file_format': '### 📋 Formato del Archivo',
        'components': '📌 Componentes',
        'name_ex': 'Nombre: Juan, María...',
        'period_ex': 'Período: 1er Tiempo...',
        'minute_ex': 'Minuto: 00:00-01:00...',
        'position_ex': 'Posición: Delantero...',
        'tip': '💡 Consejo',
        'tip_text': 'Múltiples archivos CSV',
        'multi_file_ex': '📁 Múltiples Archivos',
        'moving_average': 'Media Móvil',
        'window_size': 'Ventana',
        'critical_markers': 'Marcadores Críticos',
        'trend_analysis': 'Análisis de Tendencia',
        'observations': 'Observaciones',
        'r_squared': 'R²',
        'gain_xp': '¡Ganaste +5 XP!',
        'achievement_unlocked': '🏆 ¡Logro Desbloqueado!',
        'first_analysis': 'Primer Análisis',
        'critical_finder': 'Cazador de Críticos',
        'master_analyst': 'Analista Maestro'
    },
    'es-mx': {
        'title': 'Sports Science Analytics Pro',
        'upload': 'Carga de Datos',
        'variable': 'Variable',
        'position': 'Posición',
        'period': 'Periodo',
        'athlete': 'Atleta',
        'config': 'Configuración',
        'tab_distribution': '📊 Distribución',
        'tab_temporal': '📈 Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlaciones',
        'tab_comparison': '⚖️ Comparaciones',
        'tab_executive': '📋 Ejecutivo',
        'mean': 'Media',
        'median': 'Mediana',
        'mode': 'Moda',
        'std': 'Desviación Estándar',
        'variance': 'Varianza',
        'cv': 'CV',
        'min': 'Mínimo',
        'max': 'Máximo',
        'amplitude': 'Amplitud',
        'q1': 'Q1 (25%)',
        'q3': 'Q3 (75%)',
        'iqr': 'IQR',
        'skewness': 'Asimetría',
        'kurtosis': 'Curtosis',
        'max_value': 'MÁXIMO',
        'min_value': 'MÍNIMO',
        'minute_of_max': 'Minuto del Máx',
        'minute_of_min': 'Minuto del Mín',
        'threshold_80': 'UMBRAL 80%',
        'critical_events': 'EVENTOS CRÍTICOS',
        'above_threshold': 'arriba del umbral',
        'intensity_zones': 'Zonas de Intensidad',
        'zone_method': 'Método',
        'percentiles': 'Percentiles',
        'based_on_max': 'Basado en Máximo',
        'very_low': 'Muy Baja',
        'low': 'Baja',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muy Alta',
        'process': '⚡ PROCESAR ANÁLISIS',
        'descriptive_stats': '📊 Estadísticas Descriptivas',
        'confidence_interval': '🎯 Intervalo de Confianza',
        'normality_test': '🧪 Prueba de Normalidad',
        'summary_by_group': '🏃 Resumen por Grupo',
        'symmetric': 'Simétrica',
        'moderate_skew': 'Asimetría Moderada',
        'high_skew': 'Asimetría Fuerte',
        'leptokurtic': 'Leptocúrtica',
        'platykurtic': 'Platicúrtica',
        'mesokurtic': 'Mesocúrtica',
        'strong_positive': 'Correlación Fuerte Positiva',
        'moderate_positive': 'Correlación Moderada Positiva',
        'weak_positive': 'Correlación Débil Positiva',
        'very_weak_positive': 'Correlación Muy Débil Positiva',
        'very_weak_negative': 'Correlación Muy Débil Negativa',
        'weak_negative': 'Correlación Débil Negativa',
        'moderate_negative': 'Correlación Moderada Negativa',
        'strong_negative': 'Correlación Fuerte Negativa',
        'iqr_title': '📌 IQR',
        'iqr_explanation': 'Rango Intercuartil (Q3 - Q1)',
        'step1': '👈 **Paso 1:** Cargue archivos CSV',
        'step2': '👈 **Paso 2:** Seleccione filtros y procese',
        'file_format': '### 📋 Formato del Archivo',
        'components': '📌 Componentes',
        'name_ex': 'Nombre: Juan, María...',
        'period_ex': 'Periodo: 1er Tiempo...',
        'minute_ex': 'Minuto: 00:00-01:00...',
        'position_ex': 'Posición: Delantero...',
        'tip': '💡 Consejo',
        'tip_text': 'Múltiples archivos CSV',
        'multi_file_ex': '📁 Múltiples Archivos',
        'moving_average': 'Media Móvil',
        'window_size': 'Ventana',
        'critical_markers': 'Marcadores Críticos',
        'trend_analysis': 'Análisis de Tendencia',
        'observations': 'Observaciones',
        'r_squared': 'R²',
        'gain_xp': '¡Ganaste +5 XP!',
        'achievement_unlocked': '🏆 ¡Logro Desbloqueado!',
        'first_analysis': 'Primer Análisis',
        'critical_finder': 'Cazador de Críticos',
        'master_analyst': 'Analista Maestro'
    }
}

# ============================================================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# ============================================================================

def parse_identificacao(series):
    """Parseia a coluna de identificação de forma robusta"""
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
            st.error(f"Erro ao ler {file.name}: {str(e)}")
            continue
    
    if not dataframes:
        st.error("❌ Nenhum arquivo válido encontrado!")
        return None, [], [], [], []
    
    estruturas_ok, estrutura_base = verificar_estruturas_arquivos(dataframes)
    
    if not estruturas_ok:
        st.error(f"❌ Arquivos com estruturas diferentes!")
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
        st.error("❌ Nenhuma variável numérica encontrada!")
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
        st.error("❌ Dataframe vazio após processamento!")
        return None, [], [], [], []
    
    periodos_unicos = sorted([p for p in df_estruturado['Período'].unique() if p and p.strip()])
    posicoes_unicas = sorted([p for p in df_estruturado['Posição'].unique() if p and p.strip()])
    
    return df_estruturado, variaveis_quant, periodos_unicos, posicoes_unicas, nomes_arquivos

def media_movel(series, window):
    """Calcula média móvel"""
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()

def extrair_minuto_extremo(df, coluna_valor, coluna_minuto, extremo='max'):
    """Extrai o minuto onde ocorre o valor extremo"""
    try:
        if df.empty:
            return "N/A"
        
        if extremo == 'max':
            idx = df[coluna_valor].idxmax()
        else:
            idx = df[coluna_valor].idxmin()
        
        if pd.notna(idx):
            return str(df.loc[idx, coluna_minuto])
        
        return "N/A"
    except:
        return "N/A"

def calcular_cv(media, desvio):
    """Calcula coeficiente de variação"""
    if media != 0 and not np.isnan(media) and not np.isnan(desvio):
        return (desvio / abs(media)) * 100
    return 0

def criar_zonas_intensidade(df, variavel, metodo='percentis'):
    """Cria zonas de intensidade"""
    if metodo == 'percentis':
        return {
            'Muito Baixa': df[variavel].quantile(0.2),
            'Baixa': df[variavel].quantile(0.4),
            'Moderada': df[variavel].quantile(0.6),
            'Alta': df[variavel].quantile(0.8),
            'Muito Alta': df[variavel].quantile(1.0)
        }
    else:
        max_val = df[variavel].max()
        return {
            'Muito Baixa': max_val * 0.2,
            'Baixa': max_val * 0.4,
            'Moderada': max_val * 0.6,
            'Alta': max_val * 0.8,
            'Muito Alta': max_val
        }

def comparar_grupos(df, variavel, grupo1, grupo2):
    """Compara dois grupos usando teste estatístico"""
    try:
        dados1 = df[df['Posição'] == grupo1][variavel].dropna()
        dados2 = df[df['Posição'] == grupo2][variavel].dropna()
        
        if len(dados1) < 3 or len(dados2) < 3:
            return None
        
        if len(dados1) <= 5000 and len(dados2) <= 5000:
            _, p1 = stats.shapiro(dados1)
            _, p2 = stats.shapiro(dados2)
            
            if p1 > 0.05 and p2 > 0.05:
                stat, p_valor = stats.ttest_ind(dados1, dados2)
                teste = "Teste t"
            else:
                stat, p_valor = stats.mannwhitneyu(dados1, dados2)
                teste = "Mann-Whitney"
        else:
            stat, p_valor = stats.mannwhitneyu(dados1, dados2)
            teste = "Mann-Whitney"
        
        return {
            'teste': teste,
            'p_valor': p_valor,
            'significativo': p_valor < 0.05,
            'media_g1': dados1.mean(),
            'media_g2': dados2.mean(),
            'std_g1': dados1.std(),
            'std_g2': dados2.std(),
            'n_g1': len(dados1),
            'n_g2': len(dados2)
        }
    except Exception as e:
        return None

def interpretar_teste(p_valor, nome_teste, t):
    """Interpreta teste de normalidade"""
    if p_valor < 0.0001:
        p_text = f"{p_valor:.2e}"
    else:
        p_text = f"{p_valor:.5f}"
    
    if p_valor > 0.05:
        status = f"✅ {t['normality_test']}"
        cor = "#10b981"
    else:
        status = f"⚠️ {t['normality_test']}"
        cor = "#ef4444"
    
    st.markdown(f"""
    <div class="stat-container" style="border-left: 5px solid {cor};">
        <h4 style="color: {cor};">{status}</h4>
        <p style="color: #e2e8f0;"><strong>Teste:</strong> {nome_teste}</p>
        <p style="color: #e2e8f0;"><strong>p-valor:</strong> <span style="color: {cor};">{p_text}</span></p>
    </div>
    """, unsafe_allow_html=True)

def add_xp(amount=5):
    """Adiciona XP e verifica level up"""
    st.session_state.xp_points += amount
    if st.session_state.xp_points >= st.session_state.gamification_level * 100:
        st.session_state.gamification_level += 1
        st.balloons()
        st.success(f"🎮 LEVEL UP! Agora você é nível {st.session_state.gamification_level}!")

def executive_card(titulo, valor, delta, icone, cor_status="#00ffff"):
    """Card executivo com delta"""
    delta_icon = "▲" if delta > 0 else "▼"
    delta_color = "#10b981" if delta > 0 else "#ef4444"
    
    st.markdown(f"""
    <div class="sci-card" style="border-left: 4px solid {cor_status};">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <p class="label">{titulo}</p>
                <p class="value">{valor}</p>
                <p style="color: {delta_color};">{delta_icon} {abs(delta):.1f}%</p>
            </div>
            <div style="font-size: 2.5rem;">{icone}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def time_metric_card(label, valor, sub_label="", cor="#00ffff"):
    """Card para métricas temporais"""
    st.markdown(f"""
    <div class="sci-card" style="border-left: 4px solid {cor};">
        <p class="label">{label}</p>
        <p class="value" style="font-size: 1.8rem;">{valor}</p>
        <p style="color: #8a8f9c;">{sub_label}</p>
    </div>
    """, unsafe_allow_html=True)

def warning_card(titulo, valor, subtitulo, icone="⚠️"):
    """Card de aviso"""
    st.markdown(f"""
    <div class="warning-card">
        <p class="label">{icone} {titulo}</p>
        <p class="value">{valor}</p>
        <p style="color: #ff8888;">{subtitulo}</p>
    </div>
    """, unsafe_allow_html=True)

def comparar_atletas(df, atleta1, atleta2, variaveis, t):
    """Comparação lado a lado de dois atletas"""
    try:
        dados1 = df[df['Nome'] == atleta1][variaveis].mean()
        dados2 = df[df['Nome'] == atleta2][variaveis].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {atleta1}")
            for var in variaveis:
                delta = ((dados1[var] - dados2[var]) / dados2[var]) * 100 if dados2[var] != 0 else 0
                cor = "#10b981" if delta > 0 else "#ef4444"
                st.markdown(f"""
                <div class="stat-container" style="border-left: 4px solid {cor};">
                    <span style="color: #8a8f9c;">{var}:</span>
                    <span style="color: white; float: right; font-weight: bold;">{dados1[var]:.2f}</span>
                    <br>
                    <span style="color: {cor};">{delta:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### {atleta2}")
            for var in variaveis:
                delta = ((dados2[var] - dados1[var]) / dados1[var]) * 100 if dados1[var] != 0 else 0
                cor = "#10b981" if delta > 0 else "#ef4444"
                st.markdown(f"""
                <div class="stat-container" style="border-left: 4px solid {cor};">
                    <span style="color: #8a8f9c;">{var}:</span>
                    <span style="color: white; float: right; font-weight: bold;">{dados2[var]:.2f}</span>
                    <br>
                    <span style="color: {cor};">{delta:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erro na comparação: {str(e)}")

def sistema_anotacoes(t):
    """Sistema de anotações"""
    with st.expander("📝 Anotações"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            nova_anotacao = st.text_area("Nova anotação:", height=100, key="nova_anotacao")
        
        with col2:
            if st.button("➕ Adicionar", use_container_width=True):
                if nova_anotacao:
                    st.session_state.anotacoes.append({
                        'data': datetime.now().strftime("%d/%m/%Y %H:%M"),
                        'texto': nova_anotacao
                    })
                    add_xp(2)
                    st.rerun()
        
        for anotacao in reversed(st.session_state.anotacoes):
            st.markdown(f"""
            <div class="note-card">
                <p class="note-date">{anotacao['data']}</p>
                <p class="note-text">{anotacao['texto']}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# FUNÇÃO PRINCIPAL DE TIMELINE
# ============================================================================

def criar_timeline_quantum(df_completo, atletas_selecionados, periodos_selecionados,
                          variavel, window_size, t):
    """Cria timeline com todas as funcionalidades"""
    try:
        if not atletas_selecionados or not periodos_selecionados:
            return None, [], []
        
        df_filtrado = df_completo[
            df_completo['Nome'].isin(atletas_selecionados) &
            df_completo['Período'].isin(periodos_selecionados)
        ].copy()
        
        if df_filtrado.empty:
            return None, [], []
        
        df_filtrado = df_filtrado.sort_values('Minuto')
        
        combinacoes = df_filtrado.groupby(['Nome', 'Período']).size().reset_index()[['Nome', 'Período']]
        combinacoes_list = list(zip(combinacoes['Nome'], combinacoes['Período']))
        
        fig = go.Figure()
        cores = px.colors.qualitative.Set2 + px.colors.qualitative.Set1
        
        valor_maximo = df_filtrado[variavel].max()
        limiar_80 = valor_maximo * 0.8 if valor_maximo > 0 else 0
        
        fig.add_hline(
            y=limiar_80,
            line_dash="solid",
            line_color="#ef4444",
            line_width=3,
            annotation_text=f"🔴 {t['threshold_80']}: {limiar_80:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="white", size=12)
        )
        
        if st.session_state.show_moving_average:
            df_filtrado['Media_Movel'] = media_movel(df_filtrado[variavel], window_size)
            fig.add_trace(go.Scatter(
                x=df_filtrado['Minuto'],
                y=df_filtrado['Media_Movel'],
                mode='lines',
                name=f"📈 {t['moving_average']} ({window_size})",
                line=dict(color='#ff00ff', width=3, dash='dash'),
                opacity=0.9
            ))
        
        minutos_criticos = []
        
        for i, (atleta, periodo) in enumerate(combinacoes_list):
            df_combo = df_filtrado[
                (df_filtrado['Nome'] == atleta) &
                (df_filtrado['Período'] == periodo)
            ].copy().sort_values('Minuto')
            
            if df_combo.empty:
                continue
            
            cor = cores[i % len(cores)]
            mask_critico = df_combo[variavel] > limiar_80
            df_normal = df_combo[~mask_critico]
            df_critico = df_combo[mask_critico]
            
            minutos_criticos.extend(df_critico['Minuto'].tolist())
            
            fig.add_trace(go.Scatter(
                x=df_combo['Minuto'],
                y=df_combo[variavel],
                mode='lines',
                name=f"{atleta[:15]} - {periodo[:10]}",
                line=dict(color=cor, width=2)
            ))
            
            if not df_normal.empty:
                fig.add_trace(go.Scatter(
                    x=df_normal['Minuto'],
                    y=df_normal[variavel],
                    mode='markers',
                    marker=dict(size=6, color=cor, opacity=0.6, line=dict(color='white', width=1)),
                    showlegend=False
                ))
            
            if not df_critico.empty:
                fig.add_trace(go.Scatter(
                    x=df_critico['Minuto'],
                    y=df_critico[variavel],
                    mode='markers',
                    marker=dict(size=10, color='#ef4444', symbol='circle', line=dict(color='white', width=1)),
                    showlegend=False
                ))
        
        if minutos_criticos and st.session_state.show_critical_markers:
            y_min = df_filtrado[variavel].min()
            y_range = df_filtrado[variavel].max() - y_min
            y_pos = y_min - (y_range * 0.1)
            
            fig.add_trace(go.Scatter(
                x=minutos_criticos,
                y=[y_pos] * len(minutos_criticos),
                mode='markers',
                name=t['critical_markers'],
                marker=dict(size=8, color='#ef4444', symbol='triangle-down', line=dict(color='white', width=1))
            ))
            
            for m in set(minutos_criticos):
                fig.add_vline(x=m, line_width=1, line_dash="dot", line_color="#ef4444", opacity=0.3)
        
        media_global = df_filtrado[variavel].mean()
        desvio_global = df_filtrado[variavel].std()
        
        fig.add_hline(
            y=media_global,
            line_dash="dash",
            line_color="#00ffff",
            annotation_text=f"📊 {t['mean']}: {media_global:.2f}",
            annotation_position="bottom left"
        )
        
        fig.add_hrect(
            y0=media_global - desvio_global,
            y1=media_global + desvio_global,
            fillcolor="#00ffff",
            opacity=0.1,
            line_width=0
        )
        
        fig.update_layout(
            title=dict(
                text=f"⚡ {t['trend_analysis']} - {variavel}",
                font=dict(size=20, color='#00ffff'),
                x=0.5
            ),
            xaxis_title="⏱️ Minuto",
            yaxis_title=f"📊 {variavel}",
            hovermode='x unified',
            plot_bgcolor='#0f1425',
            paper_bgcolor='#0f1425',
            font=dict(color='#e2e8f0'),
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='#1a1f35',
                bordercolor='#00ffff',
                borderwidth=1
            )
        )
        
        fig.update_xaxes(gridcolor='#2a2f45', tickangle=-45, tickfont=dict(color='#8a8f9c'))
        fig.update_yaxes(gridcolor='#2a2f45', tickfont=dict(color='#8a8f9c'))
        
        return fig, combinacoes_list, minutos_criticos
        
    except Exception as e:
        st.error(f"Erro na timeline: {str(e)}")
        return None, [], []

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<div class='sidebar-title'>🌐 IDIOMA</div>", unsafe_allow_html=True)
    
    idiomas = ['pt', 'en', 'es', 'es-mx']
    idx_idioma = idiomas.index(st.session_state.idioma) if st.session_state.idioma in idiomas else 0
    idioma = st.selectbox("Selecione o idioma", idiomas, index=idx_idioma, label_visibility="collapsed", key="idioma_selector")
    
    if idioma != st.session_state.idioma:
        st.session_state.idioma = idioma
        st.rerun()
    
    t = translations[st.session_state.idioma]
    
    st.markdown("---")
    st.markdown(f"<div class='sidebar-title'>📂 {t['upload']}</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
    "Upload de arquivos CSV",
    type=['csv'],
    accept_multiple_files=True,
    help=t['tip_text'],
    key="file_uploader"
)
    
    if uploaded_files and len(uploaded_files) > 0:
        with st.spinner('⚡ Processando...'):
            df, vars_quant, periodos, posicoes, nomes = processar_upload(uploaded_files)
            
            if df is not None:
                st.session_state.df_completo = df
                st.session_state.variaveis_quantitativas = vars_quant
                st.session_state.todos_periodos = periodos
                st.session_state.periodos_selecionados = periodos.copy()
                st.session_state.todos_posicoes = posicoes
                st.session_state.posicoes_selecionadas = posicoes.copy()
                st.session_state.atletas_selecionados = sorted(df['Nome'].unique().tolist())
                st.session_state.upload_files_names = nomes
                
                if vars_quant and st.session_state.variavel_selecionada is None:
                    st.session_state.variavel_selecionada = vars_quant[0]
                
                st.success(f"✅ {len(nomes)} {t['upload']}")
                add_xp(10)
                st.rerun()
    
    if st.session_state.df_completo is not None:
        st.markdown("---")
        
        if st.session_state.variaveis_quantitativas:
            st.markdown(f"<div class='sidebar-title'>📈 {t['variable']}</div>", unsafe_allow_html=True)
            
            idx_var = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                idx_var = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            var_sel = st.selectbox(
    "Selecione a variável",
    options=st.session_state.variaveis_quantitativas,
    index=idx_var,
    label_visibility="collapsed",
    key="var_selector"
)
            
            if var_sel != st.session_state.variavel_selecionada:
                st.session_state.variavel_selecionada = var_sel
                st.session_state.dados_processados = False
                st.rerun()
        
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown(f"<div class='sidebar-title'>📍 {t['position']}</div>", unsafe_allow_html=True)
            
            select_all_pos = st.checkbox(
                "Todos" if st.session_state.idioma in ['pt', 'es', 'es-mx'] else "All",
                value=len(st.session_state.posicoes_selecionadas) == len(st.session_state.todos_posicoes),
                key="todos_posicoes_check"
            )
            
            if select_all_pos:
                if st.session_state.posicoes_selecionadas != st.session_state.todos_posicoes:
                    st.session_state.posicoes_selecionadas = st.session_state.todos_posicoes.copy()
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                pos_sel = st.multiselect(
    "Selecione as posições",
    options=st.session_state.todos_posicoes,
    default=st.session_state.posicoes_selecionadas,
    label_visibility="collapsed",
    key="posicoes_selector"
)
                if pos_sel != st.session_state.posicoes_selecionadas:
                    st.session_state.posicoes_selecionadas = pos_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown(f"<div class='sidebar-title'>📅 {t['period']}</div>", unsafe_allow_html=True)
            
            select_all_per = st.checkbox(
                "Todos" if st.session_state.idioma in ['pt', 'es', 'es-mx'] else "All",
                value=len(st.session_state.periodos_selecionados) == len(st.session_state.todos_periodos),
                key="todos_periodos_check"
            )
            
            if select_all_per:
                if st.session_state.periodos_selecionados != st.session_state.todos_periodos:
                    st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                    st.session_state.ordem_personalizada = st.session_state.todos_periodos.copy()
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                per_sel = st.multiselect(
    "Selecione os períodos",
    options=st.session_state.todos_periodos,
    default=st.session_state.periodos_selecionados,
    label_visibility="collapsed",
    key="periodos_selector"
)
                if per_sel != st.session_state.periodos_selecionados:
                    st.session_state.periodos_selecionados = per_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown(f"<div class='sidebar-title'>👤 {t['athlete']}</div>", unsafe_allow_html=True)
        
        df_temp = st.session_state.df_completo.copy()
        if st.session_state.posicoes_selecionadas:
            df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
        if st.session_state.periodos_selecionados:
            df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
        
        atletas_disp = sorted(df_temp['Nome'].unique().tolist())
        
        if not st.session_state.atletas_selecionados or len(st.session_state.atletas_selecionados) == 0:
            if atletas_disp:
                st.session_state.atletas_selecionados = [atletas_disp[0]]
                st.rerun()
        
        select_all_atl = st.checkbox(
            "Todos" if st.session_state.idioma in ['pt', 'es', 'es-mx'] else "All",
            value=len(st.session_state.atletas_selecionados) == len(atletas_disp) and len(atletas_disp) > 0,
            key="todos_atletas_check"
        )
        
        if select_all_atl:
            if st.session_state.atletas_selecionados != atletas_disp:
                st.session_state.atletas_selecionados = atletas_disp.copy()
                st.session_state.dados_processados = False
                st.rerun()
        else:
            atl_sel = st.multiselect(
    "Selecione os atletas",
    options=atletas_disp,
    default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else (atletas_disp[:1] if atletas_disp else []),
    label_visibility="collapsed",
    key="atletas_selector"
)
            if atl_sel != st.session_state.atletas_selecionados:
                if len(atl_sel) > 0:
                    st.session_state.atletas_selecionados = atl_sel
                else:
                    st.session_state.atletas_selecionados = [atletas_disp[0]]
                st.session_state.dados_processados = False
                st.rerun()
        
        st.markdown("---")
        st.markdown(f"<div class='sidebar-title'>⚙️ {t['config']}</div>", unsafe_allow_html=True)
        
        st.session_state.n_classes = st.slider(
            "Classes:" if st.session_state.idioma == 'en' else "Clases:",
            3, 20, st.session_state.n_classes, key="classes_slider"
        )
        
        st.session_state.window_size = st.slider(
            f"{t['window_size']}:",
            2, 10, st.session_state.window_size, key="window_slider"
        )
        
        st.session_state.show_moving_average = st.checkbox(
            t['moving_average'],
            st.session_state.show_moving_average,
            key="show_ma_check"
        )
        
        st.session_state.show_critical_markers = st.checkbox(
            t['critical_markers'],
            st.session_state.show_critical_markers,
            key="show_cm_check"
        )
        
        st.markdown("---")
        
        with st.expander("🏆 Conquistas"):
            if len(st.session_state.achievements) > 0:
                for ach in st.session_state.achievements:
                    nome_conquista = ""
                    if ach == 'primeira_analise':
                        nome_conquista = t['first_analysis']
                    elif ach == 'critical_master':
                        nome_conquista = t['critical_finder']
                    elif ach == 'multi_atleta':
                        nome_conquista = "Multi-Atleta"
                    elif ach == 'forte_correlacao':
                        nome_conquista = "Correlação Forte"
                    else:
                        nome_conquista = ach
                    
                    st.markdown(f"""
                    <div class="zone-card" style="border-left-color: #00ffff;">
                        <p class="zone-name">{nome_conquista}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Processe dados para ganhar conquistas!")
        
        st.markdown("---")
        
        pode_processar = (st.session_state.variavel_selecionada is not None and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados and 
                         len(st.session_state.atletas_selecionados) > 0)
        
        if st.button(t['process'], use_container_width=True, disabled=not pode_processar, key="process_button"):
            st.session_state.processar_click = True
            st.rerun()

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.processar_click and st.session_state.df_completo is not None:
    
    with st.spinner('⚡ Gerando análise quântica...'):
        time.sleep(0.5)
        
        df = st.session_state.df_completo
        atletas = st.session_state.atletas_selecionados
        posicoes = st.session_state.posicoes_selecionadas
        periodos = st.session_state.periodos_selecionados
        variavel = st.session_state.variavel_selecionada
        n_classes = st.session_state.n_classes
        window = st.session_state.window_size
        
        df_filtrado = df[
            df['Nome'].isin(atletas) &
            df['Posição'].isin(posicoes) &
            df['Período'].isin(periodos)
        ].copy()
        
        df_filtrado = df_filtrado.dropna(subset=[variavel])
        
        if df_filtrado.empty:
            st.warning("⚠️ Nenhum dado encontrado!")
        else:
            st.session_state.dados_processados = True
            t = translations[st.session_state.idioma]
            
            add_xp(5)
            
            if 'primeira_analise' not in st.session_state.achievements:
                st.session_state.achievements.append('primeira_analise')
                st.balloons()
                st.success(f"🏆 {t['achievement_unlocked']}: {t['first_analysis']}")
            
            st.markdown(f"<h2>📊 {t['tab_executive']}</h2>", unsafe_allow_html=True)
            
            media_global = df_filtrado[variavel].mean()
            valor_max = df_filtrado[variavel].max()
            valor_min = df_filtrado[variavel].min()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                executive_card(t['mean'], f"{media_global:.2f}", 0, "📊")
            with col2:
                executive_card(t['max_value'], f"{valor_max:.2f}", 0, "📈")
            with col3:
                executive_card(t['min_value'], f"{valor_min:.2f}", 0, "📉")
            with col4:
                executive_card(t['observations'], str(len(df_filtrado)), 0, "👥")
            
            st.markdown("---")
            
            tabs = st.tabs([
                t['tab_distribution'],
                t['tab_temporal'],
                t['tab_boxplots'],
                t['tab_correlation'],
                t['tab_comparison'],
                t['tab_executive']
            ])
            
            # ABA 1: DISTRIBUIÇÃO
            with tabs[0]:
                st.markdown(f"<h3>{t['tab_distribution']}</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=df_filtrado[variavel],
                        nbinsx=n_classes,
                        name='Frequência',
                        marker_color='#00ffff',
                        opacity=0.7,
                        hovertemplate='<b>Valor:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>'
                    ))
                    
                    fig_hist.add_vline(
                        x=media_global,
                        line_dash="dash",
                        line_color="#ff00ff",
                        line_width=2,
                        annotation_text=f"{t['mean']}: {media_global:.2f}",
                        annotation_position="top",
                        annotation_font=dict(color="#ff00ff")
                    )
                    
                    mediana_val = df_filtrado[variavel].median()
                    fig_hist.add_vline(
                        x=mediana_val,
                        line_dash="dot",
                        line_color="#ffff00",
                        line_width=2,
                        annotation_text=f"{t['median']}: {mediana_val:.2f}",
                        annotation_position="bottom",
                        annotation_font=dict(color="#ffff00")
                    )
                    
                    fig_hist.update_layout(
                        title=f"Distribuição - {variavel}",
                        plot_bgcolor='#0f1425',
                        paper_bgcolor='#0f1425',
                        font=dict(color='#e2e8f0'),
                        xaxis_title=variavel,
                        yaxis_title="Frequência",
                        bargap=0.1,
                        showlegend=False
                    )
                    fig_hist.update_xaxes(gridcolor='#2a2f45', tickfont=dict(color='#8a8f9c'))
                    fig_hist.update_yaxes(gridcolor='#2a2f45', tickfont=dict(color='#8a8f9c'))
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    dados = df_filtrado[variavel].dropna()
                    quantis_teoricos = stats.norm.ppf(np.linspace(0.01, 0.99, len(dados)))
                    quantis_observados = np.sort(dados)
                    
                    z = np.polyfit(quantis_teoricos, quantis_observados, 1)
                    p = np.poly1d(z)
                    residuos = quantis_observados - p(quantis_teoricos)
                    ss_res = np.sum(residuos**2)
                    ss_tot = np.sum((quantis_observados - np.mean(quantis_observados))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    fig_qq = go.Figure()
                    
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=quantis_observados,
                        mode='markers',
                        name='Dados',
                        marker=dict(color='#00ffff', size=8, opacity=0.7),
                        hovertemplate='<b>Quantil Teórico:</b> %{x:.3f}<br><b>Quantil Observado:</b> %{y:.3f}<extra></extra>'
                    ))
                    
                    fig_qq.add_trace(go.Scatter(
                        x=[quantis_teoricos.min(), quantis_teoricos.max()],
                        y=[quantis_teoricos.min(), quantis_teoricos.max()],
                        mode='lines',
                        name=f'Referência (R² = 1.0)',
                        line=dict(color='#ff00ff', width=2, dash='dash'),
                        hovertemplate='<b>Linha de Referência</b><extra></extra>'
                    ))
                    
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=p(quantis_teoricos),
                        mode='lines',
                        name=f'{t["r_squared"]} = {r2:.3f}',
                        line=dict(color='#ffff00', width=2),
                        hovertemplate=f'<b>{t["r_squared"]}:</b> {r2:.3f}<extra></extra>'
                    ))
                    
                    fig_qq.update_layout(
                        title=f"QQ Plot - {variavel}",
                        plot_bgcolor='#0f1425',
                        paper_bgcolor='#0f1425',
                        font=dict(color='#e2e8f0'),
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis Observados"
                    )
                    fig_qq.update_xaxes(gridcolor='#2a2f45', tickfont=dict(color='#8a8f9c'))
                    fig_qq.update_yaxes(gridcolor='#2a2f45', tickfont=dict(color='#8a8f9c'))
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
            
            # ABA 2: TEMPORAL
            with tabs[1]:
                st.markdown(f"<h3>{t['tab_temporal']}</h3>", unsafe_allow_html=True)
                
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                minuto_max = extrair_minuto_extremo(df_tempo, variavel, 'Minuto', 'max')
                minuto_min = extrair_minuto_extremo(df_tempo, variavel, 'Minuto', 'min')
                limiar_80 = valor_max * 0.8
                
                eventos_criticos = (df_tempo[variavel] > limiar_80).sum()
                percentual_critico = (eventos_criticos / len(df_tempo)) * 100 if len(df_tempo) > 0 else 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    time_metric_card(t['max_value'], f"{valor_max:.2f}", f"{t['minute_of_max']}: {minuto_max}", "#ef4444")
                with col2:
                    time_metric_card(t['min_value'], f"{valor_min:.2f}", f"{t['minute_of_min']}: {minuto_min}", "#10b981")
                with col3:
                    time_metric_card(t['mean'], f"{media_global:.2f}", "", "#00ffff")
                with col4:
                    time_metric_card(t['threshold_80'], f"{limiar_80:.2f}", "", "#ff00ff")
                with col5:
                    warning_card(t['critical_events'], str(eventos_criticos), f"{percentual_critico:.1f}% {t['above_threshold']}")
                
                st.markdown("---")
                
                st.markdown(f"<h4>{t['intensity_zones']}</h4>", unsafe_allow_html=True)
                
                metodo = st.radio(
                    t['zone_method'],
                    [t['percentiles'], t['based_on_max']],
                    horizontal=True,
                    key="metodo_zona_radio"
                )
                
                st.session_state.metodo_zona = 'percentis' if metodo == t['percentiles'] else 'based_on_max'
                
                zonas = criar_zonas_intensidade(df_filtrado, variavel, st.session_state.metodo_zona)
                cores_zonas = ['#00ffff', '#ff00ff', '#ffff00', '#ef4444', '#10b981']
                
                cols_zone = st.columns(5)
                for i, (zona, limite) in enumerate(zonas.items()):
                    with cols_zone[i]:
                        if i == 0:
                            count = (df_filtrado[variavel] <= limite).sum()
                        else:
                            limite_anterior = list(zonas.values())[i-1]
                            count = ((df_filtrado[variavel] > limite_anterior) & (df_filtrado[variavel] <= limite)).sum()
                        pct = count / len(df_filtrado) * 100
                        st.markdown(f"""
                        <div class="zone-card" style="border-left-color: {cores_zonas[i]};">
                            <div class="zone-name">{zona}</div>
                            <div class="zone-value">{limite:.1f}</div>
                            <div class="zone-count">{count} ({pct:.0f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown(f"<h4>⏱️ {t['trend_analysis']}</h4>", unsafe_allow_html=True)
                
                resultado = criar_timeline_quantum(df, atletas, periodos, variavel, window, t)
                
                if resultado and resultado[0]:
                    fig_timeline, combinacoes, minutos_criticos = resultado
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"✅ {len(combinacoes)} combinações")
                        if len(combinacoes) > 5 and 'multi_atleta' not in st.session_state.achievements:
                            st.session_state.achievements.append('multi_atleta')
                            st.balloons()
                            st.success(f"🏆 {t['achievement_unlocked']}: Multi-Atleta!")
                    with col2:
                        st.warning(f"⚠️ {len(set(minutos_criticos))} minutos críticos")
                        if len(set(minutos_criticos)) > 3 and 'critical_master' not in st.session_state.achievements:
                            st.session_state.achievements.append('critical_master')
                            st.balloons()
                            st.success(f"🏆 {t['achievement_unlocked']}: {t['critical_finder']}")
                    with col3:
                        st.info(f"📊 {len(df_filtrado)} observações")
                
                st.markdown("---")
                
                st.markdown(f"<h4>{t['descriptive_stats']}</h4>", unsafe_allow_html=True)
                
                media = df_filtrado[variavel].mean()
                desvio = df_filtrado[variavel].std()
                mediana = df_filtrado[variavel].median()
                q1 = df_filtrado[variavel].quantile(0.25)
                q3 = df_filtrado[variavel].quantile(0.75)
                iqr = q3 - q1
                assimetria = df_filtrado[variavel].skew()
                curtose = df_filtrado[variavel].kurtosis()
                
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    st.markdown(f"""
                    <div class="stat-container">
                        <h4>{t['mean']}</h4>
                        <p><strong>{t['mean']}:</strong> {media:.3f}</p>
                        <p><strong>{t['median']}:</strong> {mediana:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_e2:
                    st.markdown(f"""
                    <div class="stat-container">
                        <h4>{t['std']}</h4>
                        <p><strong>{t['std']}:</strong> {desvio:.3f}</p>
                        <p><strong>{t['cv']}:</strong> {calcular_cv(media, desvio):.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_e3:
                    st.markdown(f"""
                    <div class="stat-container">
                        <h4>{t['iqr']}</h4>
                        <p><strong>{t['q1']}:</strong> {q1:.3f}</p>
                        <p><strong>{t['q3']}:</strong> {q3:.3f}</p>
                        <p><strong>{t['iqr']}:</strong> {iqr:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                col_a1, col_a2 = st.columns(2)
                
                with col_a1:
                    if abs(assimetria) < 0.5:
                        interp_ass = t['symmetric']
                    elif abs(assimetria) < 1:
                        interp_ass = t['moderate_skew']
                    else:
                        interp_ass = t['high_skew']
                    
                    st.markdown(f"""
                    <div class="stat-container">
                        <h4>{t['skewness']}</h4>
                        <p><strong>Valor:</strong> {assimetria:.3f}</p>
                        <p><strong>Interpretação:</strong> {interp_ass}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_a2:
                    if curtose > 0:
                        interp_curt = t['leptokurtic']
                    elif curtose < 0:
                        interp_curt = t['platykurtic']
                    else:
                        interp_curt = t['mesokurtic']
                    
                    st.markdown(f"""
                    <div class="stat-container">
                        <h4>{t['kurtosis']}</h4>
                        <p><strong>Valor:</strong> {curtose:.3f}</p>
                        <p><strong>Interpretação:</strong> {interp_curt}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown(f"<h4>{t['confidence_interval']}</h4>", unsafe_allow_html=True)
                
                col_ic1, col_ic2 = st.columns([1, 2])
                
                with col_ic1:
                    n = len(df_filtrado)
                    erro_padrao = desvio / np.sqrt(n)
                    
                    if n > 30:
                        z = stats.norm.ppf(0.975)
                        ic_inf = media - z * erro_padrao
                        ic_sup = media + z * erro_padrao
                        dist = "Normal"
                    else:
                        t_val = stats.t.ppf(0.975, n-1)
                        ic_inf = media - t_val * erro_padrao
                        ic_sup = media + t_val * erro_padrao
                        dist = "t-Student"
                    
                    st.markdown(f"""
                    <div class="stat-container">
                        <p><strong>{t['mean']}:</strong> {media:.3f}</p>
                        <p><strong>Erro Padrão:</strong> {erro_padrao:.3f}</p>
                        <p><strong>IC Inferior:</strong> {ic_inf:.3f}</p>
                        <p><strong>IC Superior:</strong> {ic_sup:.3f}</p>
                        <p><small>Distribuição: {dist}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_ic2:
                    fig_ic = go.Figure()
                    
                    fig_ic.add_trace(go.Scatter(
                        x=['IC 95%'],
                        y=[media],
                        mode='markers',
                        marker=dict(color='#00ffff', size=20),
                        error_y=dict(type='constant', value=(ic_sup - media), color='#ff00ff', thickness=3, width=15)
                    ))
                    
                    fig_ic.update_layout(
                        title=t['confidence_interval'],
                        plot_bgcolor='#0f1425',
                        paper_bgcolor='#0f1425',
                        font=dict(color='#e2e8f0'),
                        showlegend=False
                    )
                    fig_ic.update_xaxes(gridcolor='#2a2f45')
                    fig_ic.update_yaxes(gridcolor='#2a2f45')
                    
                    st.plotly_chart(fig_ic, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown(f"<h4>{t['normality_test']}</h4>", unsafe_allow_html=True)
                
                dados_teste = df_filtrado[variavel].dropna()
                n_teste = len(dados_teste)
                
                if n_teste < 3:
                    st.error("❌ Amostra muito pequena (n < 3)")
                elif n_teste > 5000:
                    try:
                        k2, p = stats.normaltest(dados_teste)
                        interpretar_teste(p, "D'Agostino-Pearson", t)
                    except:
                        st.warning("⚠️ Teste não disponível")
                else:
                    try:
                        shapiro = stats.shapiro(dados_teste)
                        interpretar_teste(shapiro.pvalue, "Shapiro-Wilk", t)
                    except:
                        st.error("❌ Erro no teste")
                
                st.markdown("---")
                
                st.markdown(f"<h4>{t['summary_by_group']}</h4>", unsafe_allow_html=True)
                
                resumo = []
                for nome in atletas:
                    for pos in posicoes:
                        for per in periodos:
                            dados = df_filtrado[
                                (df_filtrado['Nome'] == nome) & 
                                (df_filtrado['Posição'] == pos) &
                                (df_filtrado['Período'] == per)
                            ]
                            if len(dados) > 0:
                                media_grupo = dados[variavel].mean()
                                desvio_grupo = dados[variavel].std()
                                cv_grupo = calcular_cv(media_grupo, desvio_grupo)
                                max_grupo = dados[variavel].max()
                                min_grupo = dados[variavel].min()
                                minuto_max = extrair_minuto_extremo(dados, variavel, 'Minuto', 'max')
                                minuto_min = extrair_minuto_extremo(dados, variavel, 'Minuto', 'min')
                                
                                resumo.append({
                                    'Atleta': nome,
                                    'Posição': pos,
                                    'Período': per,
                                    f'Máx {variavel}': max_grupo,
                                    'Minuto Máx': minuto_max,
                                    f'Mín {variavel}': min_grupo,
                                    'Minuto Mín': minuto_min,
                                    'Amplitude': max_grupo - min_grupo,
                                    'Média': media_grupo,
                                    'CV (%)': cv_grupo,
                                    'N': len(dados)
                                })
                
                if resumo:
                    df_resumo = pd.DataFrame(resumo)
                    st.dataframe(df_resumo, use_container_width=True, hide_index=True)
            
            # ABA 3: BOXPLOTS
            with tabs[2]:
                st.markdown(f"<h3>{t['tab_boxplots']}</h3>", unsafe_allow_html=True)
                
                st.markdown(f"<h4>📍 {t['position']}</h4>", unsafe_allow_html=True)
                
                fig_box_pos = go.Figure()
                for pos in posicoes:
                    dados = df_filtrado[df_filtrado['Posição'] == pos][variavel]
                    if len(dados) > 0:
                        fig_box_pos.add_trace(go.Box(
                            y=dados,
                            name=pos,
                            boxmean='sd',
                            marker_color='#00ffff',
                            line_color='#ff00ff'
                        ))
                
                fig_box_pos.update_layout(
                    plot_bgcolor='#0f1425',
                    paper_bgcolor='#0f1425',
                    font=dict(color='#e2e8f0')
                )
                fig_box_pos.update_xaxes(gridcolor='#2a2f45')
                fig_box_pos.update_yaxes(gridcolor='#2a2f45')
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                st.markdown(f"<h4>👤 {t['athlete']}</h4>", unsafe_allow_html=True)
                
                fig_box_atl = go.Figure()
                for atl in atletas[:10]:
                    dados = df_filtrado[df_filtrado['Nome'] == atl][variavel]
                    if len(dados) > 0:
                        fig_box_atl.add_trace(go.Box(
                            y=dados,
                            name=atl[:15] + "..." if len(atl) > 15 else atl,
                            boxmean='sd',
                            marker_color='#ff00ff',
                            line_color='#00ffff'
                        ))
                
                fig_box_atl.update_layout(
                    plot_bgcolor='#0f1425',
                    paper_bgcolor='#0f1425',
                    font=dict(color='#e2e8f0'),
                    height=max(400, len(atletas[:10]) * 30)
                )
                fig_box_atl.update_xaxes(gridcolor='#2a2f45', tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#2a2f45')
                st.plotly_chart(fig_box_atl, use_container_width=True)
            
            # ABA 4: CORRELAÇÕES
            with tabs[3]:
                st.markdown(f"<h3>{t['tab_correlation']}</h3>", unsafe_allow_html=True)
                
                if len(st.session_state.variaveis_quantitativas) > 1:
                    vars_corr = st.multiselect(
                        "Variáveis:",
                        st.session_state.variaveis_quantitativas,
                        default=st.session_state.variaveis_quantitativas[:min(5, len(st.session_state.variaveis_quantitativas))]
                    )
                    
                    if len(vars_corr) >= 2:
                        df_corr = df_filtrado[vars_corr].corr()
                        
                        fig_corr = px.imshow(
                            df_corr,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale=[[0, '#000080'], [0.5, '#ffffff'], [1, '#8b0000']],
                            zmin=-1, zmax=1
                        )
                        fig_corr.update_layout(
                            plot_bgcolor='#0f1425',
                            paper_bgcolor='#0f1425',
                            font=dict(color='#e2e8f0'),
                            height=500
                        )
                        fig_corr.update_xaxes(tickfont=dict(color='#e2e8f0'))
                        fig_corr.update_yaxes(tickfont=dict(color='#e2e8f0'))
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        if len(vars_corr) == 2:
                            corr_valor = df_corr.iloc[0, 1]
                            if abs(corr_valor) > 0.7:
                                intensidade = "forte"
                            elif abs(corr_valor) > 0.5:
                                intensidade = "moderada"
                            else:
                                intensidade = "fraca"
                            
                            direcao = "positiva" if corr_valor > 0 else "negativa"
                            
                            st.info(f"Correlação {intensidade} {direcao} ({corr_valor:.3f})")
                            
                            if abs(corr_valor) > 0.8 and 'forte_correlacao' not in st.session_state.achievements:
                                st.session_state.achievements.append('forte_correlacao')
                                st.balloons()
                                st.success(f"🏆 {t['achievement_unlocked']}: Correlação Forte!")
            
            # ABA 5: COMPARAÇÕES
            with tabs[4]:
                st.markdown(f"<h3>{t['tab_comparison']}</h3>", unsafe_allow_html=True)
                
                if len(posicoes) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        grupo1 = st.selectbox("Grupo 1:", posicoes, index=0, key="g1")
                    with col2:
                        grupo2 = st.selectbox("Grupo 2:", posicoes, index=min(1, len(posicoes)-1), key="g2")
                    
                    if grupo1 != grupo2:
                        resultado = comparar_grupos(df_filtrado, variavel, grupo1, grupo2)
                        
                        if resultado:
                            st.markdown(f"""
                            <div class="stat-container">
                                <h4>📊 Resultado</h4>
                                <p><strong>{grupo1}:</strong> {resultado['media_g1']:.2f} ± {resultado['std_g1']:.2f} (n={resultado['n_g1']})</p>
                                <p><strong>{grupo2}:</strong> {resultado['media_g2']:.2f} ± {resultado['std_g2']:.2f} (n={resultado['n_g2']})</p>
                                <p><strong>p-valor:</strong> {resultado['p_valor']:.4f}</p>
                                <p><strong>{'✅ Significativo' if resultado['significativo'] else '❌ Não significativo'}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # ABA 6: EXECUTIVO
            with tabs[5]:
                st.markdown(f"<h3>{t['tab_executive']}</h3>", unsafe_allow_html=True)
                
                if len(atletas) >= 2:
                    st.markdown("### 🆚 Comparação de Atletas")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        atl1 = st.selectbox("Atleta 1:", atletas, index=0, key="atl1")
                    with col2:
                        atl2 = st.selectbox("Atleta 2:", atletas, index=min(1, len(atletas)-1), key="atl2")
                    
                    if atl1 != atl2:
                        vars_comp = st.multiselect(
                            "Variáveis:",
                            st.session_state.variaveis_quantitativas,
                            default=st.session_state.variaveis_quantitativas[:3]
                        )
                        
                        if vars_comp:
                            comparar_atletas(df_filtrado, atl1, atl2, vars_comp, t)
                
                st.markdown("---")
                
                sistema_anotacoes(t)
            
            with st.expander("📋 Dados Filtrados"):
                st.dataframe(df_filtrado, use_container_width=True)
    
    st.session_state.processar_click = False

elif st.session_state.df_completo is None:
    t = translations[st.session_state.idioma]
    st.info("👈 **Faça upload dos arquivos CSV para começar**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(t['file_format'])
        
        exemplo = pd.DataFrame({
            'Nome-Período-Minuto': [
                'João-1º Tempo-00:00-01:00',
                'Maria-2º Tempo-05:00-06:00',
                'Pedro-1º Tempo-44:00-45:00'
            ],
            'Posição': ['Atacante', 'Meio-campo', 'Zagueiro'],
            'Distância': [250, 180, 200],
            'Velocidade': [23, 29, 33]
        })
        st.dataframe(exemplo, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown(f"""
        <div class="sci-card">
            <h4>{t['components']}</h4>
            <p>{t['name_ex']}</p>
            <p>{t['period_ex']}</p>
            <p>{t['minute_ex']}</p>
            <p>{t['position_ex']}</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.dados_processados:
    t = translations[st.session_state.idioma]
    st.info("👈 **Selecione os filtros e clique em Processar**")
    
    with st.expander("📋 Preview dos dados"):
        if st.session_state.upload_files_names:
            st.caption(f"Arquivos: {', '.join(st.session_state.upload_files_names)}")
        st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
        st.caption(f"Total: {len(st.session_state.df_completo)} observações")
        st.caption(f"Variáveis: {', '.join(st.session_state.variaveis_quantitativas)}")