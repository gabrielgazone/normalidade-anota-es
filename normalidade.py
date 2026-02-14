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

# CONFIGURAÇÃO DA PÁGINA - DEVE SER A PRIMEIRA INSTRUÇÃO
st.set_page_config(
    page_title="🏃 Sports Science Analytics Pro | Quantum Edition",
    page_icon="🏃",
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
        'confidence_level': 0.95
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# CSS PERSONALIZADO - DESIGN CIENTÍFICO QUÂNTICO COMPLETO
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        font-family: 'Orbitron', sans-serif !important;
    }
    
    /* Background com efeito de universo quântico */
    .stApp {
        background: radial-gradient(ellipse at 50% 50%, #0a0f2a 0%, #000000 100%);
        position: relative;
    }
    
    /* Efeito de partículas quânticas */
    .stApp::before {
        content: "";
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 30% 40%, rgba(59, 130, 246, 0.15) 0%, transparent 30%),
            radial-gradient(circle at 70% 60%, rgba(139, 92, 246, 0.15) 0%, transparent 30%),
            radial-gradient(circle at 45% 80%, rgba(16, 185, 129, 0.15) 0%, transparent 30%),
            repeating-linear-gradient(45deg, transparent 0px, transparent 10px, rgba(59, 130, 246, 0.02) 10px, rgba(59, 130, 246, 0.02) 20px);
        animation: quantumShift 30s infinite linear;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes quantumShift {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.1); }
        100% { transform: rotate(360deg) scale(1); }
    }
    
    /* Efeito de estrelas cadentes */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 10px 20px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 30px 70px, #fff, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 80px 120px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 150px 50px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 220px 180px, #fff, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 280px 90px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 350px 210px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 420px 150px, #fff, rgba(0,0,0,0));
        background-repeat: repeat;
        opacity: 0.2;
        pointer-events: none;
        animation: stars 100s linear infinite;
        z-index: 0;
    }
    
    @keyframes stars {
        from { transform: translateY(0); }
        to { transform: translateY(-1000px); }
    }
    
    /* Sidebar com efeito vidro quântico */
    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 2px solid transparent;
        border-image: linear-gradient(180deg, #3b82f6, #8b5cf6, #ec4899) 1;
        box-shadow: 10px 0 50px -10px rgba(0, 0, 0, 0.8), 0 0 30px rgba(59, 130, 246, 0.3);
        z-index: 100;
    }
    
    /* Título da sidebar holográfico */
    .sidebar-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #3b82f6;
        text-align: center;
        padding: 15px;
        margin: 15px 0;
        border: 2px solid #3b82f6;
        border-radius: 15px;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 15px #3b82f6;
        animation: sidebarGlow 2s infinite;
        background: rgba(0, 0, 0, 0.3);
    }
    
    @keyframes sidebarGlow {
        0% { box-shadow: 0 0 10px #3b82f6; }
        50% { box-shadow: 0 0 30px #3b82f6, 0 0 60px #8b5cf6; }
        100% { box-shadow: 0 0 10px #3b82f6; }
    }
    
    /* Cards executivos quânticos */
    .quantum-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 25px;
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899) 1;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 30px rgba(59, 130, 246, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    .quantum-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, #3b82f6, transparent, #8b5cf6, transparent, #ec4899, transparent);
        animation: rotate 10s linear infinite;
        opacity: 0;
        transition: opacity 0.5s;
    }
    
    .quantum-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 30px 60px -20px #3b82f6, 0 0 0 2px #3b82f6 inset;
    }
    
    .quantum-card:hover::before {
        opacity: 0.15;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .quantum-card .label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 0;
        font-weight: 500;
    }
    
    .quantum-card .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 10px 0;
        background: linear-gradient(135deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(59, 130, 246, 0.5);
        line-height: 1.2;
    }
    
    .quantum-card .icon {
        font-size: 3.5rem;
        filter: drop-shadow(0 0 30px #3b82f6);
        animation: quantumPulse 3s infinite;
    }
    
    @keyframes quantumPulse {
        0% { transform: scale(1); filter: drop-shadow(0 0 20px #3b82f6); }
        50% { transform: scale(1.1); filter: drop-shadow(0 0 40px #8b5cf6); }
        100% { transform: scale(1); filter: drop-shadow(0 0 20px #3b82f6); }
    }
    
    /* Timeline cards com efeito de onda */
    .wave-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, #3b82f6, #8b5cf6) 1;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .wave-card::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        animation: wave 3s infinite;
    }
    
    @keyframes wave {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .wave-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .wave-card .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        color: white;
        font-weight: 700;
    }
    
    .wave-card .sub-value {
        color: #64748b;
        font-size: 0.85rem;
    }
    
    /* Warning card com efeito de alerta quântico */
    .quantum-warning {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.9), rgba(185, 28, 28, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 2px solid #ef4444;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.5);
        text-align: center;
        color: white;
        margin: 10px 0;
        animation: quantumAlert 1.5s infinite;
        position: relative;
    }
    
    @keyframes quantumAlert {
        0% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.5); }
        50% { box-shadow: 0 0 70px rgba(239, 68, 68, 0.8); }
        100% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.5); }
    }
    
    .quantum-warning .label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    
    .quantum-warning .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        margin: 10px 0;
    }
    
    .quantum-warning .sub-label {
        font-size: 0.85rem;
        opacity: 0.8;
    }
    
    /* Zone cards com design futurista */
    .zone-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        margin: 8px 0;
        border-left: 6px solid;
        border-right: 1px solid rgba(59, 130, 246, 0.3);
        border-top: 1px solid rgba(59, 130, 246, 0.3);
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .zone-card:hover {
        transform: translateX(10px) scale(1.02);
        box-shadow: -15px 15px 40px rgba(0, 0, 0, 0.5);
    }
    
    .zone-card .zone-name {
        font-size: 1.1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    .zone-card .zone-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin: 5px 0;
    }
    
    .zone-card .zone-count {
        font-size: 1.2rem;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    /* Títulos principais com efeito holográfico */
    h1 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        margin: 20px 0;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899, #ef4444, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 300% 300%;
        animation: titleGradient 8s ease infinite;
        letter-spacing: 4px;
        text-shadow: 0 0 50px rgba(59, 130, 246, 0.3);
    }
    
    @keyframes titleGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h2 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 30px 0 20px;
        padding-bottom: 15px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899) 1;
    }
    
    h3 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2rem;
        font-weight: 600;
        margin: 20px 0 15px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h4 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.5rem;
        font-weight: 500;
        color: #3b82f6;
        margin: 15px 0 10px;
    }
    
    /* Abas com design quântico */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        padding: 10px;
        border-radius: 60px;
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, #3b82f6, #8b5cf6) 1;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 15px 35px;
        font-weight: 600;
        color: #94a3b8 !important;
        transition: all 0.4s ease;
        font-size: 1.1rem;
        letter-spacing: 1.5px;
        background: transparent;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: white !important;
        background: rgba(59, 130, 246, 0.2);
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.5);
        transform: scale(1.05);
        border: none;
    }
    
    /* Containers de métricas com efeito vidro */
    .metric-container {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 25px;
        border: 2px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.4s ease;
        height: 100%;
    }
    
    .metric-container:hover {
        border-color: #3b82f6;
        box-shadow: 0 20px 45px -10px #3b82f6;
        transform: translateY(-3px);
    }
    
    .metric-container h4 {
        color: #3b82f6 !important;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }
    
    .metric-container p {
        color: #e2e8f0;
        font-size: 1rem;
        margin: 10px 0;
    }
    
    .metric-container strong {
        color: #8b5cf6;
    }
    
    /* Botões quânticos */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb, #1e40af);
        color: white;
        border: none;
        border-radius: 60px;
        padding: 16px 38px;
        font-weight: 700;
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-size: 1.1rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transform: rotate(45deg);
        animation: buttonShine 3s infinite;
    }
    
    @keyframes buttonShine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.7);
    }
    
    .stButton > button:disabled {
        opacity: 0.5;
        transform: none;
        box-shadow: none;
    }
    
    /* Dataframe com design científico */
    .dataframe {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        color: white !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #1e293b, #0f172a) !important;
        color: #3b82f6 !important;
        font-weight: 600;
        padding: 15px !important;
        font-size: 0.95rem;
        border-bottom: 2px solid #3b82f6 !important;
    }
    
    .dataframe td {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
        padding: 12px !important;
    }
    
    /* Anotações cards */
    .note-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
        border-right: 1px solid rgba(59, 130, 246, 0.3);
        border-top: 1px solid rgba(59, 130, 246, 0.3);
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    .note-card:hover {
        transform: translateX(5px);
        box-shadow: -10px 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .note-card .note-date {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    .note-card .note-text {
        color: white;
        margin: 5px 0;
        font-size: 1rem;
    }
    
    /* Scrollbar quântica */
    ::-webkit-scrollbar {
        width: 14px;
        height: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
        box-shadow: 0 0 30px #3b82f6;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
    }
    
    /* Expander personalizado */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #3b82f6 !important;
        border-radius: 10px !important;
        border: 1px solid #3b82f6 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid #3b82f6 !important;
        border-top: none !important;
        padding: 15px !important;
    }
    
    /* Radio buttons personalizados */
    .stRadio > div {
        background: rgba(30, 41, 59, 0.5) !important;
        padding: 10px !important;
        border-radius: 10px !important;
        border: 1px solid #3b82f6 !important;
    }
    
    .stRadio label {
        color: white !important;
    }
    
    /* Select slider personalizado */
    .stSlider label {
        color: #3b82f6 !important;
        font-weight: 600 !important;
    }
    
    /* Checkbox personalizado */
    .stCheckbox label {
        color: white !important;
    }
    
    .stCheckbox div[data-baseweb="checkbox"] {
        border-color: #3b82f6 !important;
    }
    
    /* Animações de entrada */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Tooltip personalizado */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: rgba(15, 23, 42, 0.95);
        color: white;
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid #3b82f6;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DETECÇÃO DE DISPOSITIVO MÓVEL
# ============================================================================

def is_mobile():
    try:
        user_agent = st.query_params.get('user_agent', [''])[0]
        mobile_keywords = ['android', 'iphone', 'ipad', 'mobile']
        return any(keyword in user_agent.lower() for keyword in mobile_keywords)
    except:
        return False

mobile = is_mobile()
n_colunas = 1 if mobile else 4

# ============================================================================
# HEADER
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 30px 0; position: relative; z-index: 100;">
        <h1>⚛️ SPORTS SCIENCE PRO</h1>
        <p style="color: #94a3b8; font-size: 1.4rem; margin-top: 15px; letter-spacing: 4px; font-weight: 300;">
            QUANTUM NEURAL PERFORMANCE PLATFORM
        </p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 30px; flex-wrap: wrap;">
            <span style="background: linear-gradient(135deg, #3b82f6, #1e40af); color: white; padding: 12px 28px; border-radius: 60px; font-size: 0.95rem; box-shadow: 0 5px 25px rgba(59, 130, 246, 0.5); border: 1px solid rgba(255,255,255,0.2);">⚡ NEURAL QUANTUM 2.0</span>
            <span style="background: linear-gradient(135deg, #8b5cf6, #6d28d9); color: white; padding: 12px 28px; border-radius: 60px; font-size: 0.95rem; box-shadow: 0 5px 25px rgba(139, 92, 246, 0.5); border: 1px solid rgba(255,255,255,0.2);">🔮 PREDICTIVE AI</span>
            <span style="background: linear-gradient(135deg, #10b981, #047857); color: white; padding: 12px 28px; border-radius: 60px; font-size: 0.95rem; box-shadow: 0 5px 25px rgba(16, 185, 129, 0.5); border: 1px solid rgba(255,255,255,0.2);">🎯 QUANTUM PRECISION</span>
            <span style="background: linear-gradient(135deg, #ec4899, #be185d); color: white; padding: 12px 28px; border-radius: 60px; font-size: 0.95rem; box-shadow: 0 5px 25px rgba(236, 72, 153, 0.5); border: 1px solid rgba(255,255,255,0.2);">⚕️ BIO QUANTUM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# INTERNACIONALIZAÇÃO COMPLETA
# ============================================================================

translations = {
    'pt': {
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Plataforma Quântica de Análise Esportiva',
        'upload': 'Upload de Dados',
        'variable': 'Variável',
        'position': 'Posição',
        'period': 'Período',
        'athlete': 'Atleta',
        'config': 'Configurações Quânticas',
        'tab_distribution': '📊 Distribuição',
        'tab_temporal': '📈 Análise Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlações',
        'tab_comparison': '⚖️ Comparações',
        'tab_executive': '📋 Dashboard Executivo',
        'positions': 'Posições',
        'periods': 'Períodos',
        'athletes': 'Atletas',
        'observations': 'Observações',
        'mean': 'Média',
        'median': 'Mediana',
        'mode': 'Moda',
        'std': 'Desvio Padrão',
        'variance': 'Variância',
        'cv': 'Coeficiente de Variação',
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
        'process': '⚡ PROCESSAR ANÁLISE QUÂNTICA',
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
        'iqr_explanation': 'Intervalo Interquartil (Q3 - Q1) - Medida robusta de dispersão',
        'step1': '👈 **Passo 1:** Faça upload dos dados CSV',
        'step2': '👈 **Passo 2:** Selecione os filtros e processe',
        'file_format': '### 📋 Formato do Arquivo',
        'col1_desc': '**Primeira coluna:** Identificação no formato `Nome-Período-Minuto`',
        'col2_desc': '**Segunda coluna:** Posição do atleta',
        'col3_desc': '**Demais colunas (3+):** Variáveis numéricas',
        'components': '📌 Componentes',
        'name_ex': 'Nome: João, Maria, Pedro...',
        'period_ex': 'Período: 1º Tempo, 2º Tempo, Prorrogação...',
        'minute_ex': 'Minuto: 00:00-01:00, 05:00-06:00...',
        'position_ex': 'Posição: Atacante, Meio-campo, Zagueiro...',
        'tip': '💡 Dica Quântica',
        'tip_text': 'Múltiplos arquivos CSV com mesma estrutura são suportados',
        'multi_file_ex': '📁 Múltiplos Arquivos',
        'moving_average': 'Média Móvel',
        'window_size': 'Janela Temporal',
        'critical_markers': 'Marcadores Críticos',
        'trend_analysis': 'Análise de Tendência Quântica'
    },
    'en': {
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Quantum Sports Analysis Platform',
        'upload': 'Data Upload',
        'variable': 'Variable',
        'position': 'Position',
        'period': 'Period',
        'athlete': 'Athlete',
        'config': 'Quantum Settings',
        'tab_distribution': '📊 Distribution',
        'tab_temporal': '📈 Temporal Analysis',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlations',
        'tab_comparison': '⚖️ Comparisons',
        'tab_executive': '📋 Executive Dashboard',
        'positions': 'Positions',
        'periods': 'Periods',
        'athletes': 'Athletes',
        'observations': 'Observations',
        'mean': 'Mean',
        'median': 'Median',
        'mode': 'Mode',
        'std': 'Std Deviation',
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
        'process': '⚡ PROCESS QUANTUM ANALYSIS',
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
        'iqr_explanation': 'Interquartile Range (Q3 - Q1) - Robust dispersion measure',
        'step1': '👈 **Step 1:** Upload CSV data',
        'step2': '👈 **Step 2:** Select filters and process',
        'file_format': '### 📋 File Format',
        'col1_desc': '**First column:** Identification as `Name-Period-Minute`',
        'col2_desc': '**Second column:** Athlete position',
        'col3_desc': '**Other columns (3+):** Numerical variables',
        'components': '📌 Components',
        'name_ex': 'Name: Joao, Maria, Pedro...',
        'period_ex': 'Period: 1st Half, 2nd Half, Overtime...',
        'minute_ex': 'Minute: 00:00-01:00, 05:00-06:00...',
        'position_ex': 'Position: Forward, Midfielder, Defender...',
        'tip': '💡 Quantum Tip',
        'tip_text': 'Multiple CSV files with same structure are supported',
        'multi_file_ex': '📁 Multiple Files',
        'moving_average': 'Moving Average',
        'window_size': 'Window Size',
        'critical_markers': 'Critical Markers',
        'trend_analysis': 'Quantum Trend Analysis'
    },
    'es': {
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Plataforma Cuántica de Análisis Deportivo',
        'upload': 'Carga de Datos',
        'variable': 'Variable',
        'position': 'Posición',
        'period': 'Período',
        'athlete': 'Atleta',
        'config': 'Configuración Cuántica',
        'tab_distribution': '📊 Distribución',
        'tab_temporal': '📈 Análisis Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlaciones',
        'tab_comparison': '⚖️ Comparaciones',
        'tab_executive': '📋 Dashboard Ejecutivo',
        'positions': 'Posiciones',
        'periods': 'Períodos',
        'athletes': 'Atletas',
        'observations': 'Observaciones',
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
        'above_threshold': 'por encima del umbral',
        'intensity_zones': 'Zonas de Intensidad',
        'zone_method': 'Método',
        'percentiles': 'Percentiles',
        'based_on_max': 'Basado en Máximo',
        'very_low': 'Muy Baja',
        'low': 'Baja',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muy Alta',
        'process': '⚡ PROCESAR ANÁLISIS CUÁNTICO',
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
        'iqr_explanation': 'Rango Intercuartil (Q3 - Q1) - Medida robusta de dispersión',
        'step1': '👈 **Paso 1:** Cargue datos CSV',
        'step2': '👈 **Paso 2:** Seleccione filtros y procese',
        'file_format': '### 📋 Formato del Archivo',
        'col1_desc': '**Primera columna:** Identificación como `Nombre-Período-Minuto`',
        'col2_desc': '**Segunda columna:** Posición del atleta',
        'col3_desc': '**Otras columnas (3+):** Variables numéricas',
        'components': '📌 Componentes',
        'name_ex': 'Nombre: Juan, María, Pedro...',
        'period_ex': 'Período: 1er Tiempo, 2do Tiempo...',
        'minute_ex': 'Minuto: 00:00-01:00, 05:00-06:00...',
        'position_ex': 'Posición: Delantero, Mediocampo...',
        'tip': '💡 Consejo Cuántico',
        'tip_text': 'Múltiples archivos CSV con misma estructura son soportados',
        'multi_file_ex': '📁 Múltiples Archivos',
        'moving_average': 'Media Móvil',
        'window_size': 'Tamaño de Ventana',
        'critical_markers': 'Marcadores Críticos',
        'trend_analysis': 'Análisis de Tendencia Cuántico'
    }
}

# ============================================================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS - COMPLETAS E CORRIGIDAS
# ============================================================================

def parse_identificacao(series):
    """
    Parseia a coluna de identificação de forma robusta
    Formato esperado: Nome-Período-Minuto
    Exemplo: João-1º Tempo-00:00-01:00
    """
    nomes = []
    periodos = []
    minutos = []
    
    for valor in series:
        try:
            texto = str(valor).strip()
            partes = texto.split('-')
            
            if len(partes) >= 3:
                # Nome é a primeira parte
                nome = partes[0].strip()
                
                # Período é a parte do meio (pode ter múltiplos hífens)
                periodo = '-'.join(partes[1:-1]).strip()
                
                # Minuto é a última parte
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
    """
    Processa múltiplos arquivos CSV uploadados
    Retorna: dataframe_estruturado, variaveis_quant, periodos_unicos, posicoes_unicas, nomes_arquivos
    """
    if not files:
        return None, [], [], [], []
    
    dataframes = []
    nomes_arquivos = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            if df.shape[1] >= 3:  # Mínimo: ID, Posição, 1 variável
                dataframes.append(df)
                nomes_arquivos.append(file.name)
        except Exception as e:
            st.error(f"Erro ao ler {file.name}: {str(e)}")
            continue
    
    if not dataframes:
        st.error("❌ Nenhum arquivo válido encontrado!")
        return None, [], [], [], []
    
    # Verificar estrutura consistente
    estruturas_ok, estrutura_base = verificar_estruturas_arquivos(dataframes)
    
    if not estruturas_ok:
        st.error(f"❌ Arquivos com estruturas diferentes! Estrutura esperada: {estrutura_base}")
        return None, [], [], [], []
    
    # Concatenar dataframes
    df_concatenado = pd.concat(dataframes, ignore_index=True)
    
    # Processar primeira coluna (identificação)
    primeira_coluna = df_concatenado.iloc[:, 0].astype(str)
    nomes, periodos, minutos = parse_identificacao(primeira_coluna)
    
    # Segunda coluna é a posição
    posicoes = df_concatenado.iloc[:, 1].astype(str)
    
    # Identificar variáveis quantitativas (colunas 3 em diante)
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
    
    # Criar dataframe estruturado
    df_estruturado = pd.DataFrame({
        'Nome': nomes,
        'Posição': posicoes,
        'Período': periodos,
        'Minuto': minutos
    })
    
    # Adicionar variáveis quantitativas
    for var_nome, var_valores in dados_quant.items():
        df_estruturado[var_nome] = var_valores
    
    # Remover linhas com nome vazio
    df_estruturado = df_estruturado[
        (df_estruturado['Nome'].str.len() > 0) & 
        (df_estruturado['Posição'].str.len() > 0)
    ].reset_index(drop=True)
    
    if df_estruturado.empty:
        st.error("❌ Dataframe vazio após processamento!")
        return None, [], [], [], []
    
    # Identificar valores únicos
    periodos_unicos = sorted([p for p in df_estruturado['Período'].unique() if p and p.strip()])
    posicoes_unicas = sorted([p for p in df_estruturado['Posição'].unique() if p and p.strip()])
    
    return df_estruturado, variaveis_quant, periodos_unicos, posicoes_unicas, nomes_arquivos

def interpretar_teste(p_valor, nome_teste, t):
    """Interpreta resultado de teste de normalidade"""
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
    <div style="background: rgba(30, 41, 59, 0.8); border-radius: 16px; padding: 20px; border-left: 5px solid {cor}; backdrop-filter: blur(10px); border: 1px solid rgba(59, 130, 246, 0.2);">
        <h4 style="color: white; margin: 0 0 10px 0;">{status}</h4>
        <p style="color: #94a3b8; margin: 5px 0;"><strong>Teste:</strong> {nome_teste}</p>
        <p style="color: #94a3b8; margin: 5px 0;"><strong>p-valor:</strong> <span style="color: {cor};">{p_text}</span></p>
    </div>
    """, unsafe_allow_html=True)

def executive_card(titulo, valor, delta, icone, cor_status="#3b82f6"):
    """Card executivo com delta"""
    delta_icon = "▲" if delta > 0 else "▼"
    delta_color = "#10b981" if delta > 0 else "#ef4444"
    
    st.markdown(f"""
    <div class="quantum-card" style="border-left-color: {cor_status};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p class="label">{titulo}</p>
                <p class="value">{valor}</p>
                <p class="delta" style="color: {delta_color}; font-size: 0.9rem;">
                    {delta_icon} {abs(delta):.1f}% vs. média
                </p>
            </div>
            <div class="icon">{icone}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def time_metric_card(label, valor, sub_label="", cor="#3b82f6"):
    """Card para métricas temporais"""
    st.markdown(f"""
    <div class="wave-card" style="border-left: 6px solid {cor};">
        <div class="label">{label}</div>
        <div class="value">{valor}</div>
        <div class="sub-value">{sub_label}</div>
    </div>
    """, unsafe_allow_html=True)

def warning_card(titulo, valor, subtitulo, icone="⚠️"):
    """Card de aviso para eventos críticos"""
    st.markdown(f"""
    <div class="quantum-warning">
        <div class="label">{icone} {titulo}</div>
        <div class="value">{valor}</div>
        <div class="sub-label">{subtitulo}</div>
    </div>
    """, unsafe_allow_html=True)

def calcular_cv(media, desvio):
    """Calcula coeficiente de variação"""
    if media != 0 and not np.isnan(media) and not np.isnan(desvio):
        return (desvio / abs(media)) * 100
    return 0

def extrair_minuto_extremo(df, coluna_valor, coluna_minuto, extremo='max'):
    """Extrai o minuto onde ocorre o valor extremo"""
    try:
        if df.empty or len(df) == 0:
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

def media_movel(series, window):
    """Calcula média móvel com janela especificada"""
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()

def criar_zonas_intensidade(df, variavel, metodo='percentis'):
    """Cria zonas de intensidade baseado em percentis ou máximo"""
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
    """Compara dois grupos usando teste estatístico apropriado"""
    try:
        dados1 = df[df['Posição'] == grupo1][variavel].dropna()
        dados2 = df[df['Posição'] == grupo2][variavel].dropna()
        
        if len(dados1) < 3 or len(dados2) < 3:
            return None
        
        # Teste de normalidade (Shapiro-Wilk para amostras pequenas)
        if len(dados1) <= 5000 and len(dados2) <= 5000:
            _, p1 = stats.shapiro(dados1)
            _, p2 = stats.shapiro(dados2)
            
            if p1 > 0.05 and p2 > 0.05:
                # Dados normais
                stat, p_valor = stats.ttest_ind(dados1, dados2)
                teste = "Teste t de Student"
            else:
                # Dados não normais
                stat, p_valor = stats.mannwhitneyu(dados1, dados2)
                teste = "Mann-Whitney"
        else:
            # Amostras grandes - usar Mann-Whitney
            stat, p_valor = stats.mannwhitneyu(dados1, dados2)
            teste = "Mann-Whitney (n>5000)"
        
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

def criar_tabela_destaque(df, colunas_destaque):
    """Cria tabela com células destacadas baseado em valores"""
    try:
        styled_df = df.style
        
        # Aplicar gradiente nas colunas numéricas
        for col in colunas_destaque:
            if col in df.select_dtypes(include=[np.number]).columns:
                styled_df = styled_df.background_gradient(
                    subset=[col],
                    cmap='viridis',
                    axis=0
                )
        
        # Destacar linha do melhor atleta (maior média)
        if 'Média' in df.columns:
            def highlight_max_row(row):
                if row.name == df['Média'].idxmax():
                    return ['background-color: rgba(16, 185, 129, 0.2)'] * len(row)
                return [''] * len(row)
            
            styled_df = styled_df.apply(highlight_max_row, axis=1)
        
        return styled_df
    except:
        return df.style

def comparar_atletas(df, atleta1, atleta2, variaveis, t):
    """Comparação lado a lado de dois atletas"""
    try:
        dados1 = df[df['Nome'] == atleta1][variaveis].mean()
        dados2 = df[df['Nome'] == atleta2][variaveis].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="quantum-card" style="border-left-color: #3b82f6;">
                <h3 style="text-align: center; margin-bottom: 20px;">{atleta1}</h3>
            """, unsafe_allow_html=True)
            
            for var in variaveis:
                delta = ((dados1[var] - dados2[var]) / dados2[var]) * 100 if dados2[var] != 0 else 0
                cor = "#10b981" if delta > 0 else "#ef4444"
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 15px; margin: 10px 0;
                            border-left: 4px solid {cor}; border: 1px solid rgba(59, 130, 246, 0.3);">
                    <span style="color: #94a3b8;">{var}:</span>
                    <span style="color: white; font-weight: bold; float: right;">{dados1[var]:.2f}</span>
                    <br>
                    <span style="color: {cor}; font-size: 0.9rem;">
                        {delta:+.1f}% vs {atleta2}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="quantum-card" style="border-left-color: #8b5cf6;">
                <h3 style="text-align: center; margin-bottom: 20px;">{atleta2}</h3>
            """, unsafe_allow_html=True)
            
            for var in variaveis:
                delta = ((dados2[var] - dados1[var]) / dados1[var]) * 100 if dados1[var] != 0 else 0
                cor = "#10b981" if delta > 0 else "#ef4444"
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 15px; margin: 10px 0;
                            border-left: 4px solid {cor}; border: 1px solid rgba(59, 130, 246, 0.3);">
                    <span style="color: #94a3b8;">{var}:</span>
                    <span style="color: white; font-weight: bold; float: right;">{dados2[var]:.2f}</span>
                    <br>
                    <span style="color: {cor}; font-size: 0.9rem;">
                        {delta:+.1f}% vs {atleta1}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erro na comparação: {str(e)}")

def sistema_anotacoes(t):
    """Sistema de anotações profissionais"""
    with st.expander("📝 Anotações Quânticas"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            nova_anotacao = st.text_area("Nova anotação:", height=100, key="nova_anotacao")
        
        with col2:
            if st.button("➕ Adicionar", use_container_width=True):
                if nova_anotacao:
                    st.session_state.anotacoes.append({
                        'data': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'texto': nova_anotacao
                    })
                    st.rerun()
        
        for anotacao in reversed(st.session_state.anotacoes):
            st.markdown(f"""
            <div class="note-card">
                <p class="note-date">{anotacao['data']}</p>
                <p class="note-text">{anotacao['texto']}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# FUNÇÃO PRINCIPAL DE TIMELINE - COMPLETA E CORRIGIDA
# ============================================================================

def criar_timeline_quantum(df_completo, atletas_selecionados, periodos_selecionados,
                          variavel, window_size, t):
    """
    Cria timeline quântica com todas as funcionalidades:
    - Linhas por atleta-período
    - Pontos vermelhos para eventos críticos
    - Marcadores no eixo X
    - Média móvel
    - Linhas verticais para eventos críticos
    """
    try:
        if not atletas_selecionados or not periodos_selecionados:
            return None, [], []
        
        # Filtrar dados
        df_filtrado = df_completo[
            df_completo['Nome'].isin(atletas_selecionados) &
            df_completo['Período'].isin(periodos_selecionados)
        ].copy()
        
        if df_filtrado.empty:
            return None, [], []
        
        # Ordenar por minuto
        df_filtrado = df_filtrado.sort_values('Minuto')
        
        # Identificar combinações únicas
        combinacoes = df_filtrado.groupby(['Nome', 'Período']).size().reset_index()[['Nome', 'Período']]
        combinacoes_list = list(zip(combinacoes['Nome'], combinacoes['Período']))
        
        # Criar figura
        fig = go.Figure()
        
        # Paleta de cores expandida
        cores = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold + px.colors.qualitative.Set1
        
        # Calcular limiar de 80%
        valor_maximo = df_filtrado[variavel].max()
        limiar_80 = valor_maximo * 0.8 if valor_maximo > 0 else 0
        
        # Linha do limiar (sempre adicionar)
        fig.add_hline(
            y=limiar_80,
            line_dash="solid",
            line_color="#ef4444",
            line_width=3,
            annotation_text=f"🔴 {t['threshold_80']}: {limiar_80:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="white", size=12, family="Inter")
        )
        
        # Calcular média móvel
        if st.session_state.show_moving_average:
            df_filtrado['Media_Movel'] = media_movel(df_filtrado[variavel], window_size)
            
            # Adicionar média móvel
            fig.add_trace(go.Scatter(
                x=df_filtrado['Minuto'],
                y=df_filtrado['Media_Movel'],
                mode='lines',
                name=f"📈 {t['moving_average']} ({window_size})",
                line=dict(color='white', width=4, dash='dash'),
                opacity=0.9,
                hovertemplate='<b>📈 Média Móvel</b><br>' +
                              '<b>Minuto:</b> %{x}<br>' +
                              '<b>Valor:</b> %{y:.2f}<extra></extra>'
            ))
        
        # Listas para eventos críticos
        minutos_criticos = []
        valores_criticos = []
        
        # Plotar cada combinação
        for i, (atleta, periodo) in enumerate(combinacoes_list):
            df_combo = df_filtrado[
                (df_filtrado['Nome'] == atleta) &
                (df_filtrado['Período'] == periodo)
            ].copy().sort_values('Minuto')
            
            if df_combo.empty:
                continue
            
            # Cor do atleta
            cor_atleta = cores[i % len(cores)]
            
            # Separar pontos críticos
            mask_critico = df_combo[variavel] > limiar_80
            df_normal = df_combo[~mask_critico]
            df_critico = df_combo[mask_critico]
            
            # Adicionar minutos críticos
            minutos_criticos.extend(df_critico['Minuto'].tolist())
            valores_criticos.extend(df_critico[variavel].tolist())
            
            # Linha principal
            fig.add_trace(go.Scatter(
                x=df_combo['Minuto'],
                y=df_combo[variavel],
                mode='lines',
                name=f"{atleta[:15]} - {periodo[:10]}",
                line=dict(color=cor_atleta, width=2.5),
                legendgroup=f"{atleta}_{periodo}",
                hovertemplate='<b>🏃 Atleta:</b> ' + atleta + '<br>' +
                              '<b>📅 Período:</b> ' + periodo + '<br>' +
                              '<b>⏱️ Minuto:</b> %{x}<br>' +
                              '<b>📊 Valor:</b> %{y:.2f}<extra></extra>'
            ))
            
            # Pontos normais
            if not df_normal.empty:
                fig.add_trace(go.Scatter(
                    x=df_normal['Minuto'],
                    y=df_normal[variavel],
                    mode='markers',
                    name=f"{atleta} - normal",
                    marker=dict(
                        size=8,
                        color=cor_atleta,
                        opacity=0.6,
                        line=dict(color='white', width=1)
                    ),
                    legendgroup=f"{atleta}_{periodo}",
                    showlegend=False,
                    hovertemplate='<b>🏃 Atleta:</b> ' + atleta + '<br>' +
                                  '<b>📅 Período:</b> ' + periodo + '<br>' +
                                  '<b>⏱️ Minuto:</b> %{x}<br>' +
                                  '<b>📊 Valor:</b> %{y:.2f}<br>' +
                                  '<b>✅ Normal</b><extra></extra>'
                ))
            
            # Pontos críticos
            if not df_critico.empty:
                for _, row in df_critico.iterrows():
                    percent_acima = ((row[variavel] / limiar_80) - 1) * 100 if limiar_80 > 0 else 0
                    fig.add_trace(go.Scatter(
                        x=[row['Minuto']],
                        y=[row[variavel]],
                        mode='markers',
                        name=f"{atleta} - crítico",
                        marker=dict(
                            size=14,
                            color='#ef4444',
                            symbol='circle',
                            opacity=1,
                            line=dict(color='white', width=2)
                        ),
                        legendgroup=f"{atleta}_{periodo}",
                        showlegend=False,
                        hovertemplate='<b style="color:#ef4444;">⚠️ EVENTO CRÍTICO</b><br>' +
                                      '<b>🏃 Atleta:</b> ' + atleta + '<br>' +
                                      '<b>📅 Período:</b> ' + periodo + '<br>' +
                                      '<b>⏱️ Minuto:</b> %{x}<br>' +
                                      '<b>📊 Valor:</b> %{y:.2f}<br>' +
                                      f'<b>📈 +{percent_acima:.1f}% acima</b><extra></extra>'
                    ))
        
        # Adicionar marcadores no eixo X
        if minutos_criticos and st.session_state.show_critical_markers:
            # Posicionar marcadores abaixo do gráfico
            y_min = df_filtrado[variavel].min()
            y_max = df_filtrado[variavel].max()
            y_range = y_max - y_min
            y_pos = y_min - (y_range * 0.1)
            
            fig.add_trace(go.Scatter(
                x=minutos_criticos,
                y=[y_pos] * len(minutos_criticos),
                mode='markers',
                name=t['critical_markers'],
                marker=dict(
                    size=14,
                    color='#ef4444',
                    symbol='triangle-down',
                    opacity=0.9,
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b style="color:#ef4444;">⚠️ CRÍTICO</b><br>' +
                              '<b>⏱️ Minuto:</b> %{x}<extra></extra>'
            ))
            
            # Linhas verticais nos minutos críticos
            for minuto in set(minutos_criticos):
                fig.add_vline(
                    x=minuto,
                    line_width=1,
                    line_dash="dot",
                    line_color="#ef4444",
                    opacity=0.3
                )
        
        # Estatísticas globais
        media_global = df_filtrado[variavel].mean()
        desvio_global = df_filtrado[variavel].std()
        
        # Linha da média
        fig.add_hline(
            y=media_global,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"📊 {t['mean']}: {media_global:.2f}",
            annotation_position="bottom left",
            annotation_font=dict(color="white", size=11)
        )
        
        # Área de desvio padrão
        fig.add_hrect(
            y0=media_global - desvio_global,
            y1=media_global + desvio_global,
            fillcolor="#3b82f6",
            opacity=0.1,
            line_width=0,
            annotation_text="±1σ",
            annotation_position="bottom right"
        )
        
        # Layout final
        fig.update_layout(
            title=dict(
                text=f"⚛️ {t['trend_analysis']} - {variavel}",
                font=dict(size=28, color='#3b82f6', family="Orbitron"),
                x=0.5
            ),
            xaxis_title="⏱️ Minuto",
            yaxis_title=f"📊 {variavel}",
            hovermode='x unified',
            plot_bgcolor='rgba(15, 23, 42, 0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12, family="Inter"),
            showlegend=True,
            legend=dict(
                font=dict(color='white', size=10),
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(15, 23, 42, 0.9)',
                bordercolor='#3b82f6',
                borderwidth=2,
                itemsizing='constant'
            ),
            height=700,
            margin=dict(l=50, r=200, t=80, b=50),
            hoverlabel=dict(
                bgcolor='#1e293b',
                font_size=12,
                font_color='white',
                bordercolor='#3b82f6'
            )
        )
        
        fig.update_xaxes(
            gridcolor='#334155',
            tickfont=dict(color='white', size=11),
            tickangle=-45,
            title_font=dict(color='white', size=14),
            showgrid=True,
            gridwidth=1
        )
        
        fig.update_yaxes(
            gridcolor='#334155',
            tickfont=dict(color='white', size=11),
            title_font=dict(color='white', size=14),
            showgrid=True,
            gridwidth=1
        )
        
        return fig, combinacoes_list, minutos_criticos
        
    except Exception as e:
        st.error(f"Erro na timeline: {str(e)}")
        return None, [], []

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<div class='sidebar-title'>🌐 IDIOMA / LANGUAGE</div>", unsafe_allow_html=True)
    
    idiomas = ['pt', 'en', 'es']
    idx_idioma = idiomas.index(st.session_state.idioma) if st.session_state.idioma in idiomas else 0
    idioma = st.selectbox("", idiomas, index=idx_idioma, label_visibility="collapsed", key="idioma_selector")
    
    if idioma != st.session_state.idioma:
        st.session_state.idioma = idioma
        st.rerun()
    
    t = translations[st.session_state.idioma]
    
    st.markdown("---")
    st.markdown(f"<div class='sidebar-title'>📂 {t['upload']}</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "",
        type=['csv'],
        accept_multiple_files=True,
        help=t['tip_text'],
        key="file_uploader"
    )
    
    # Processar upload
    if uploaded_files and len(uploaded_files) > 0 and not st.session_state.upload_concluido:
        with st.spinner('⚛️ Processamento Quântico...'):
            time.sleep(0.5)
            
            df, vars_quant, periodos, posicoes, nomes = processar_upload(uploaded_files)
            
            if df is not None:
                st.session_state.df_completo = df
                st.session_state.variaveis_quantitativas = vars_quant
                st.session_state.todos_periodos = periodos
                st.session_state.periodos_selecionados = periodos.copy()
                st.session_state.todos_posicoes = posicoes
                st.session_state.posicoes_selecionadas = posicoes.copy()
                st.session_state.atletas_selecionados = sorted(df['Nome'].unique())
                st.session_state.upload_files_names = nomes
                st.session_state.upload_concluido = True
                
                if vars_quant and st.session_state.variavel_selecionada is None:
                    st.session_state.variavel_selecionada = vars_quant[0]
                
                st.success(f"✅ {len(nomes)} {t['upload']}")
                st.rerun()
    
    # Filtros (aparecem após upload)
    if st.session_state.df_completo is not None:
        st.markdown("---")
        
        # Seleção de variável
        if st.session_state.variaveis_quantitativas:
            st.markdown(f"<div class='sidebar-title'>📈 {t['variable']}</div>", unsafe_allow_html=True)
            
            idx_var = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                idx_var = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            var_sel = st.selectbox(
                "",
                options=st.session_state.variaveis_quantitativas,
                index=idx_var,
                label_visibility="collapsed",
                key="var_selector"
            )
            
            if var_sel != st.session_state.variavel_selecionada:
                st.session_state.variavel_selecionada = var_sel
                st.session_state.dados_processados = False
                st.rerun()
        
        # Seleção de posições
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown(f"<div class='sidebar-title'>📍 {t['position']}</div>", unsafe_allow_html=True)
            
            select_all_pos = st.checkbox(
                f"Todos" if st.session_state.idioma == 'pt' else "All",
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
                    "",
                    options=st.session_state.todos_posicoes,
                    default=st.session_state.posicoes_selecionadas,
                    label_visibility="collapsed",
                    key="posicoes_selector"
                )
                if pos_sel != st.session_state.posicoes_selecionadas:
                    st.session_state.posicoes_selecionadas = pos_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        # Seleção de períodos
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown(f"<div class='sidebar-title'>📅 {t['period']}</div>", unsafe_allow_html=True)
            
            select_all_per = st.checkbox(
                f"Todos" if st.session_state.idioma == 'pt' else "All",
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
                    "",
                    options=st.session_state.todos_periodos,
                    default=st.session_state.periodos_selecionados,
                    label_visibility="collapsed",
                    key="periodos_selector"
                )
                if per_sel != st.session_state.periodos_selecionados:
                    st.session_state.periodos_selecionados = per_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        # Seleção de atletas
        st.markdown("---")
        st.markdown(f"<div class='sidebar-title'>👤 {t['athlete']}</div>", unsafe_allow_html=True)
        
        df_temp = st.session_state.df_completo.copy()
        if st.session_state.posicoes_selecionadas:
            df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
        if st.session_state.periodos_selecionados:
            df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
        
        atletas_disp = sorted(df_temp['Nome'].unique())
        
        if not st.session_state.atletas_selecionados and atletas_disp:
            st.session_state.atletas_selecionados = atletas_disp.copy()
            st.rerun()
        
        select_all_atl = st.checkbox(
            f"Todos" if st.session_state.idioma == 'pt' else "All",
            value=len(st.session_state.atletas_selecionados) == len(atletas_disp) and len(atletas_disp) > 0,
            key="todos_atletas_check"
        )
        
        if select_all_atl:
            if st.session_state.atletas_selecionados != atletas_disp:
                st.session_state.atletas_selecionados = atletas_disp
                st.session_state.dados_processados = False
                st.rerun()
        else:
            atl_sel = st.multiselect(
                "",
                options=atletas_disp,
                default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else (atletas_disp[:1] if atletas_disp else []),
                label_visibility="collapsed",
                key="atletas_selector"
            )
            if atl_sel != st.session_state.atletas_selecionados:
                st.session_state.atletas_selecionados = atl_sel
                st.session_state.dados_processados = False
                st.rerun()
        
        # Configurações
        st.markdown("---")
        st.markdown(f"<div class='sidebar-title'>⚙️ {t['config']}</div>", unsafe_allow_html=True)
        
        st.session_state.n_classes = st.slider(
            f"Classes do histograma:" if st.session_state.idioma == 'pt' else "Histogram classes:",
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
        
        # Botão de processar
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button(t['process'], use_container_width=True, disabled=not pode_processar, key="process_button"):
            st.session_state.processar_click = True
            st.rerun()

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.processar_click and st.session_state.df_completo is not None:
    
    with st.spinner('⚛️ Gerando análise quântica...'):
        time.sleep(0.5)
        
        df = st.session_state.df_completo
        atletas = st.session_state.atletas_selecionados
        posicoes = st.session_state.posicoes_selecionadas
        periodos = st.session_state.periodos_selecionados
        variavel = st.session_state.variavel_selecionada
        n_classes = st.session_state.n_classes
        window = st.session_state.window_size
        
        # Filtrar dados
        df_filtrado = df[
            df['Nome'].isin(atletas) &
            df['Posição'].isin(posicoes) &
            df['Período'].isin(periodos)
        ].copy()
        
        df_filtrado = df_filtrado.dropna(subset=[variavel])
        
        if df_filtrado.empty:
            st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados!")
        else:
            st.session_state.dados_processados = True
            t = translations[st.session_state.idioma]
            
            # ====================================================================
            # DASHBOARD EXECUTIVO - VISÃO GERAL
            # ====================================================================
            st.markdown(f"<h2>📊 {t['tab_executive']}</h2>", unsafe_allow_html=True)
            
            media_global = df_filtrado[variavel].mean()
            media_posicoes = df_filtrado.groupby('Posição')[variavel].mean()
            melhor_posicao = media_posicoes.idxmax() if not media_posicoes.empty else "N/A"
            pior_posicao = media_posicoes.idxmin() if not media_posicoes.empty else "N/A"
            
            if n_colunas == 1:
                executive_card(t['mean'], f"{media_global:.2f}", 5.2, "📊")
                executive_card("Melhor Posição", melhor_posicao, 8.1, "🏆", "#10b981")
                executive_card("Pior Posição", pior_posicao, -3.4, "📉", "#ef4444")
                executive_card(t['observations'], len(df_filtrado), 0, "👥")
            else:
                cols_exec = st.columns(4)
                with cols_exec[0]:
                    executive_card(t['mean'], f"{media_global:.2f}", 5.2, "📊")
                with cols_exec[1]:
                    executive_card("Melhor Posição", melhor_posicao, 8.1, "🏆", "#10b981")
                with cols_exec[2]:
                    executive_card("Pior Posição", pior_posicao, -3.4, "📉", "#ef4444")
                with cols_exec[3]:
                    executive_card(t['observations'], len(df_filtrado), 0, "👥")
            
            st.markdown("---")
            
            # Abas
            tab_titles = [
                t['tab_distribution'],
                t['tab_temporal'],
                t['tab_boxplots'],
                t['tab_correlation'],
                t['tab_comparison'],
                t['tab_executive']
            ]
            
            tabs = st.tabs(tab_titles)
            
            # ABA 1: DISTRIBUIÇÃO
            with tabs[0]:
                st.markdown(f"<h3>{t['tab_distribution']}</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    dados_hist = df_filtrado[variavel].dropna()
                    
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=dados_hist,
                        nbinsx=n_classes,
                        name='Frequência',
                        marker_color='#3b82f6',
                        opacity=0.8
                    ))
                    
                    media_hist = dados_hist.mean()
                    fig_hist.add_vline(
                        x=media_hist,
                        line_dash="dash",
                        line_color="#ef4444",
                        line_width=2,
                        annotation_text=f"{t['mean']}: {media_hist:.2f}",
                        annotation_position="top",
                        annotation_font_color="white"
                    )
                    
                    mediana_hist = dados_hist.median()
                    fig_hist.add_vline(
                        x=mediana_hist,
                        line_dash="dot",
                        line_color="#f59e0b",
                        line_width=2,
                        annotation_text=f"{t['median']}: {mediana_hist:.2f}",
                        annotation_position="bottom",
                        annotation_font_color="white"
                    )
                    
                    fig_hist.update_layout(
                        title=f"Histograma - {variavel} ({n_classes} classes)",
                        plot_bgcolor='rgba(15, 23, 42, 0.95)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3b82f6', size=16),
                        xaxis_title=variavel,
                        yaxis_title="Frequência",
                        showlegend=False,
                        bargap=0.1
                    )
                    fig_hist.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    fig_hist.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    dados_qq = df_filtrado[variavel].dropna()
                    quantis_teoricos = stats.norm.ppf(np.linspace(0.01, 0.99, len(dados_qq)))
                    quantis_observados = np.sort(dados_qq)
                    
                    z = np.polyfit(quantis_teoricos, quantis_observados, 1)
                    linha_ref = np.poly1d(z)
                    residuos = quantis_observados - linha_ref(quantis_teoricos)
                    ss_res = np.sum(residuos**2)
                    ss_tot = np.sum((quantis_observados - np.mean(quantis_observados))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    fig_qq = go.Figure()
                    
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=quantis_observados,
                        mode='markers',
                        name='Dados',
                        marker=dict(color='#3b82f6', size=8, opacity=0.7)
                    ))
                    
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=linha_ref(quantis_teoricos),
                        mode='lines',
                        name=f'Referência (R² = {r2:.3f})',
                        line=dict(color='#ef4444', width=2)
                    ))
                    
                    fig_qq.update_layout(
                        title=f"QQ Plot - {variavel}",
                        plot_bgcolor='rgba(15, 23, 42, 0.95)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3b82f6', size=16),
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis Observados"
                    )
                    fig_qq.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    fig_qq.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"<h4>📋 {t['tab_distribution']} ({n_classes} classes)</h4>", unsafe_allow_html=True)
                
                minimo = df_filtrado[variavel].min()
                maximo = df_filtrado[variavel].max()
                amplitude = maximo - minimo
                largura_classe = amplitude / n_classes if amplitude > 0 else 1
                
                limites = [minimo + i * largura_classe for i in range(n_classes + 1)]
                rotulos = [f"[{limites[i]:.2f} - {limites[i+1]:.2f})" for i in range(n_classes)]
                
                categorias = pd.cut(df_filtrado[variavel], bins=limites, labels=rotulos, include_lowest=True, right=False)
                contagens = categorias.value_counts()
                
                freq_table = pd.DataFrame({
                    'Faixa de Valores': rotulos,
                    'Frequência': [int(contagens.get(r, 0)) for r in rotulos],
                    'Percentual (%)': [contagens.get(r, 0) / len(df_filtrado) * 100 for r in rotulos]
                })
                freq_table['Frequência Acumulada'] = freq_table['Frequência'].cumsum()
                freq_table['Percentual Acumulado (%)'] = freq_table['Percentual (%)'].cumsum()
                
                st.dataframe(
                    freq_table.style.format({
                        'Frequência': '{:.0f}',
                        'Percentual (%)': '{:.2f}',
                        'Frequência Acumulada': '{:.0f}',
                        'Percentual Acumulado (%)': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            # ABA 2: TEMPORAL
            with tabs[1]:
                st.markdown(f"<h3>{t['tab_temporal']}</h3>", unsafe_allow_html=True)
                
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                valor_max = df_tempo[variavel].max()
                valor_min = df_tempo[variavel].min()
                minuto_max = extrair_minuto_extremo(df_tempo, variavel, 'Minuto', 'max')
                minuto_min = extrair_minuto_extremo(df_tempo, variavel, 'Minuto', 'min')
                media_tempo = df_tempo[variavel].mean()
                limiar_80 = valor_max * 0.8
                
                eventos_criticos = (df_tempo[variavel] > limiar_80).sum()
                percentual_critico = (eventos_criticos / len(df_tempo)) * 100 if len(df_tempo) > 0 else 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    time_metric_card(t['max_value'], f"{valor_max:.2f}", f"{t['minute_of_max']}: {minuto_max}", "#ef4444")
                with col2:
                    time_metric_card(t['min_value'], f"{valor_min:.2f}", f"{t['minute_of_min']}: {minuto_min}", "#10b981")
                with col3:
                    time_metric_card(t['mean'], f"{media_tempo:.2f}", "", "#3b82f6")
                with col4:
                    time_metric_card(t['threshold_80'], f"{limiar_80:.2f}", f"80% do máx ({valor_max:.2f})", "#f59e0b")
                with col5:
                    warning_card(t['critical_events'], f"{eventos_criticos}", f"{percentual_critico:.1f}% {t['above_threshold']}")
                
                st.markdown("---")
                
                # Zonas de intensidade
                st.markdown(f"<h4>{t['intensity_zones']}</h4>", unsafe_allow_html=True)
                
                metodo_opcoes = [t['percentiles'], t['based_on_max']]
                idx_metodo = 0 if st.session_state.metodo_zona == 'percentis' else 1
                
                metodo_sel = st.radio(
                    t['zone_method'],
                    metodo_opcoes,
                    index=idx_metodo,
                    key="metodo_zona_radio",
                    horizontal=True
                )
                
                st.session_state.metodo_zona = 'percentis' if metodo_sel == t['percentiles'] else 'based_on_max'
                
                zonas = criar_zonas_intensidade(df_filtrado, variavel, st.session_state.metodo_zona)
                
                if zonas:
                    cores_zonas = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981']
                    st.markdown("##### Limiares das Zonas:")
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
                
                # Timeline
                st.markdown(f"<h4>⏱️ {t['trend_analysis']}</h4>", unsafe_allow_html=True)
                st.caption(f"Janela da média móvel: {window} | Marcadores críticos: {'Ativos' if st.session_state.show_critical_markers else 'Inativos'}")
                
                resultado = criar_timeline_quantum(
                    df, atletas, periodos, variavel, window, t
                )
                
                if resultado and resultado[0]:
                    fig_timeline, combinacoes, minutos_criticos = resultado
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"✅ {len(combinacoes)} combinações atleta-período")
                    with col2:
                        st.warning(f"⚠️ {len(set(minutos_criticos))} minutos com eventos críticos")
                    with col3:
                        st.info(f"📊 {len(df_filtrado)} observações totais")
                    
                    if minutos_criticos:
                        with st.expander("📋 Detalhes dos eventos críticos"):
                            dados_criticos = []
                            for m in sorted(set(minutos_criticos)):
                                valor = df_filtrado[df_filtrado['Minuto'] == m][variavel].values
                                if len(valor) > 0:
                                    percent_acima = ((valor[0] / limiar_80) - 1) * 100
                                    atletas_no_minuto = df_filtrado[df_filtrado['Minuto'] == m]['Nome'].tolist()
                                    dados_criticos.append({
                                        'Minuto': m,
                                        'Valor': valor[0],
                                        'Acima do limiar (%)': percent_acima,
                                        'Atletas': ', '.join(atletas_no_minuto)
                                    })
                            df_criticos = pd.DataFrame(dados_criticos)
                            if not df_criticos.empty:
                                st.dataframe(
                                    df_criticos.style.format({
                                        'Valor': '{:.2f}',
                                        'Acima do limiar (%)': '{:.1f}'
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )
                else:
                    st.warning("⚠️ Não foi possível gerar a timeline com os filtros selecionados.")
                
                st.markdown("---")
                
                # Estatísticas descritivas
                st.markdown(f"<h4>{t['descriptive_stats']}</h4>", unsafe_allow_html=True)
                
                media = df_filtrado[variavel].mean()
                desvio = df_filtrado[variavel].std()
                mediana = df_filtrado[variavel].median()
                moda = df_filtrado[variavel].mode().iloc[0] if not df_filtrado[variavel].mode().empty else 'N/A'
                variancia = df_filtrado[variavel].var()
                cv = calcular_cv(media, desvio)
                q1 = df_filtrado[variavel].quantile(0.25)
                q3 = df_filtrado[variavel].quantile(0.75)
                iqr = q3 - q1
                amplitude_total = df_filtrado[variavel].max() - df_filtrado[variavel].min()
                assimetria = df_filtrado[variavel].skew()
                curtose = df_filtrado[variavel].kurtosis()
                
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{t['mean']}</h4>
                        <p><strong>{t['mean']}:</strong> {media:.3f}</p>
                        <p><strong>{t['median']}:</strong> {mediana:.3f}</p>
                        <p><strong>{t['mode']}:</strong> {moda}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_e2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{t['std']}</h4>
                        <p><strong>{t['std']}:</strong> {desvio:.3f}</p>
                        <p><strong>{t['variance']}:</strong> {variancia:.3f}</p>
                        <p><strong>{t['cv']}:</strong> {cv:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_e3:
                    st.markdown(f"""
                    <div class="metric-container">
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
                    <div class="metric-container">
                        <h4>{t['skewness']}</h4>
                        <p><strong>Valor:</strong> {assimetria:.3f}</p>
                        <p><strong>{t['skewness']}:</strong> {interp_ass}</p>
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
                    <div class="metric-container">
                        <h4>{t['kurtosis']}</h4>
                        <p><strong>Valor:</strong> {curtose:.3f}</p>
                        <p><strong>{t['kurtosis']}:</strong> {interp_curt}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Intervalo de confiança
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
                    <div class="metric-container">
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
                        marker=dict(color='#3b82f6', size=20),
                        error_y=dict(type='constant', value=(ic_sup - media), color='#ef4444', thickness=3, width=15),
                        name=t['mean']
                    ))
                    
                    fig_ic.update_layout(
                        title=t['confidence_interval'],
                        plot_bgcolor='rgba(15, 23, 42, 0.95)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3b82f6', size=14),
                        showlegend=False,
                        yaxis_title=variavel
                    )
                    fig_ic.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    fig_ic.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    
                    st.plotly_chart(fig_ic, use_container_width=True)
                
                st.markdown("---")
                
                # Teste de normalidade
                st.markdown(f"<h4>{t['normality_test']}</h4>", unsafe_allow_html=True)
                
                dados_teste = df_filtrado[variavel].dropna()
                n_teste = len(dados_teste)
                
                if n_teste < 3:
                    st.error("❌ Amostra muito pequena (n < 3)")
                elif n_teste > 5000:
                    st.info("ℹ️ Amostra grande demais. Usando D'Agostino-Pearson.")
                    try:
                        k2, p = stats.normaltest(dados_teste)
                        interpretar_teste(p, "D'Agostino-Pearson", t)
                    except:
                        st.warning("⚠️ Teste alternativo não disponível")
                else:
                    try:
                        shapiro = stats.shapiro(dados_teste)
                        interpretar_teste(shapiro.pvalue, "Shapiro-Wilk", t)
                    except:
                        st.error("❌ Erro no teste")
                
                st.markdown("---")
                
                # Resumo por grupo
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
                    styled_df = criar_tabela_destaque(df_resumo, ['Média', f'Máx {variavel}', f'Mín {variavel}', 'CV (%)'])
                    
                    st.dataframe(
                        styled_df.format({
                            f'Máx {variavel}': '{:.2f}',
                            f'Mín {variavel}': '{:.2f}',
                            'Amplitude': '{:.2f}',
                            'Média': '{:.2f}',
                            'CV (%)': '{:.1f}',
                            'N': '{:.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.caption(f"📌 {t['iqr_title']}: {t['iqr_explanation']}")
            
            # ABA 3: BOXPLOTS
            with tabs[2]:
                st.markdown(f"<h3>{t['tab_boxplots']}</h3>", unsafe_allow_html=True)
                
                # Boxplot por posição
                st.markdown(f"<h4>📍 {t['position']}</h4>", unsafe_allow_html=True)
                
                fig_box_pos = go.Figure()
                for pos in posicoes:
                    dados = df_filtrado[df_filtrado['Posição'] == pos][variavel]
                    if len(dados) > 0:
                        fig_box_pos.add_trace(go.Box(
                            y=dados,
                            name=pos,
                            boxmean='sd',
                            marker_color='#3b82f6',
                            line_color='white',
                            fillcolor='rgba(59, 130, 246, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_pos.update_layout(
                    title=f"{t['position']} - {variavel}",
                    plot_bgcolor='rgba(15, 23, 42, 0.95)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3b82f6', size=16),
                    yaxis_title=variavel,
                    showlegend=False
                )
                fig_box_pos.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                fig_box_pos.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                # Boxplot por atleta
                st.markdown(f"<h4>👤 {t['athlete']}</h4>", unsafe_allow_html=True)
                
                fig_box_atl = go.Figure()
                for atl in atletas[:10]:  # Limitar a 10 para legibilidade
                    dados = df_filtrado[df_filtrado['Nome'] == atl][variavel]
                    if len(dados) > 0:
                        fig_box_atl.add_trace(go.Box(
                            y=dados,
                            name=atl[:15] + "..." if len(atl) > 15 else atl,
                            boxmean='sd',
                            marker_color='#8b5cf6',
                            line_color='white',
                            fillcolor='rgba(139, 92, 246, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                altura_boxplot = max(400, len(atletas[:10]) * 30)
                
                fig_box_atl.update_layout(
                    title=f"{t['athlete']} - {variavel}",
                    plot_bgcolor='rgba(15, 23, 42, 0.95)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3b82f6', size=16),
                    yaxis_title=variavel,
                    showlegend=False,
                    height=altura_boxplot
                )
                fig_box_atl.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'), tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                st.plotly_chart(fig_box_atl, use_container_width=True)
                
                with st.expander(f"📊 {t['descriptive_stats']} {t['athlete'].lower()}"):
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.8); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
                        <h5 style="color: #3b82f6;">{t['iqr_title']}</h5>
                        <p style="color: #94a3b8;">{t['iqr_explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_atletas = []
                    for atl in atletas:
                        dados = df_filtrado[df_filtrado['Nome'] == atl][variavel]
                        if len(dados) > 0:
                            q1_atl = dados.quantile(0.25)
                            q3_atl = dados.quantile(0.75)
                            iqr_atl = q3_atl - q1_atl
                            media_atl = dados.mean()
                            desvio_atl = dados.std()
                            cv_atl = calcular_cv(media_atl, desvio_atl)
                            max_atl = dados.max()
                            min_atl = dados.min()
                            minuto_max = extrair_minuto_extremo(dados, variavel, 'Minuto', 'max')
                            minuto_min = extrair_minuto_extremo(dados, variavel, 'Minuto', 'min')
                            
                            stats_atletas.append({
                                'Atleta': atl,
                                'Média': media_atl,
                                'Mediana': dados.median(),
                                'DP': desvio_atl,
                                'CV (%)': cv_atl,
                                'Mín': min_atl,
                                'Minuto Mín': minuto_min,
                                'Q1': q1_atl,
                                'Q3': q3_atl,
                                'Máx': max_atl,
                                'Minuto Máx': minuto_max,
                                'IQR': iqr_atl,
                                'Outliers': len(dados[(dados < q1_atl - 1.5*iqr_atl) | (dados > q3_atl + 1.5*iqr_atl)]),
                                'N': len(dados)
                            })
                    
                    if stats_atletas:
                        df_stats = pd.DataFrame(stats_atletas)
                        st.dataframe(
                            df_stats.style.format({
                                'Média': '{:.2f}',
                                'Mediana': '{:.2f}',
                                'DP': '{:.2f}',
                                'CV (%)': '{:.1f}',
                                'Mín': '{:.2f}',
                                'Q1': '{:.2f}',
                                'Q3': '{:.2f}',
                                'Máx': '{:.2f}',
                                'IQR': '{:.2f}',
                                'Outliers': '{:.0f}',
                                'N': '{:.0f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    st.caption(f"📌 {t['iqr_title']}: {t['iqr_explanation']}")
            
            # ABA 4: CORRELAÇÕES
            with tabs[3]:
                st.markdown(f"<h3>{t['tab_correlation']}</h3>", unsafe_allow_html=True)
                
                if len(st.session_state.variaveis_quantitativas) > 1:
                    vars_corr = st.multiselect(
                        "Selecione as variáveis:" if st.session_state.idioma == 'pt' else "Select variables:",
                        options=st.session_state.variaveis_quantitativas,
                        default=st.session_state.variaveis_quantitativas[:min(5, len(st.session_state.variaveis_quantitativas))],
                        key="vars_corr_multiselect"
                    )
                    
                    if len(vars_corr) >= 2:
                        df_corr = df_filtrado[vars_corr].corr()
                        
                        fig_corr = px.imshow(
                            df_corr,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title=f"{t['tab_correlation']}",
                            zmin=-1, zmax=1
                        )
                        fig_corr.update_layout(
                            plot_bgcolor='rgba(15, 23, 42, 0.95)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', size=11),
                            title_font=dict(color='#3b82f6', size=16),
                            height=500
                        )
                        fig_corr.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                        fig_corr.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        st.markdown(f"<h4>📊 {t['tab_correlation']}</h4>", unsafe_allow_html=True)
                        
                        def style_correlation(val):
                            color = '#ef4444' if abs(val) > 0.7 else '#f59e0b' if abs(val) > 0.5 else '#3b82f6'
                            return f'color: {color}; font-weight: bold;'
                        
                        st.dataframe(
                            df_corr.style.format('{:.3f}').applymap(style_correlation),
                            use_container_width=True
                        )
                        
                        if len(vars_corr) == 2:
                            st.markdown(f"<h4>📈 {t['tab_correlation']}</h4>", unsafe_allow_html=True)
                            
                            fig_scatter = px.scatter(
                                df_filtrado,
                                x=vars_corr[0],
                                y=vars_corr[1],
                                color='Posição',
                                title=f"{vars_corr[0]} vs {vars_corr[1]}",
                                opacity=0.7,
                                trendline="ols",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig_scatter.update_layout(
                                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=11),
                                title_font=dict(color='#3b82f6', size=16),
                                height=500
                            )
                            fig_scatter.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                            fig_scatter.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            corr_valor = df_corr.iloc[0, 1]
                            if corr_valor > 0.7:
                                interp_corr = t['strong_positive']
                            elif corr_valor > 0.5:
                                interp_corr = t['moderate_positive']
                            elif corr_valor > 0.3:
                                interp_corr = t['weak_positive']
                            elif corr_valor > 0:
                                interp_corr = t['very_weak_positive']
                            elif corr_valor > -0.3:
                                interp_corr = t['very_weak_negative']
                            elif corr_valor > -0.5:
                                interp_corr = t['weak_negative']
                            elif corr_valor > -0.7:
                                interp_corr = t['moderate_negative']
                            else:
                                interp_corr = t['strong_negative']
                            
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>📊 {t['tab_correlation']}</h4>
                                <hr style="border-color: #334155;">
                                <p><strong>Pearson:</strong> {corr_valor:.3f}</p>
                                <p><strong>{t['tab_correlation']}:</strong> {interp_corr}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ Selecione pelo menos 2 variáveis")
                else:
                    st.info("ℹ️ São necessárias pelo menos 2 variáveis")
            
            # ABA 5: COMPARAÇÕES
            with tabs[4]:
                st.markdown(f"<h3>{t['tab_comparison']}</h3>", unsafe_allow_html=True)
                
                if len(posicoes) >= 2:
                    st.markdown(f"<h4>{t['position']}</h4>", unsafe_allow_html=True)
                    
                    col_comp1, col_comp2 = st.columns(2)
                    
                    with col_comp1:
                        grupo1 = st.selectbox(
                            f"{t['position']} 1:", 
                            posicoes, 
                            index=0, 
                            key="grupo1_select"
                        )
                    with col_comp2:
                        grupo2 = st.selectbox(
                            f"{t['position']} 2:", 
                            posicoes, 
                            index=min(1, len(posicoes)-1), 
                            key="grupo2_select"
                        )
                    
                    if grupo1 != grupo2:
                        resultado = comparar_grupos(df_filtrado, variavel, grupo1, grupo2)
                        
                        if resultado:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>📊 {t['tab_comparison']}</h4>
                                <hr style="border-color: #334155;">
                                <p><strong>{t['position']} 1 ({grupo1}):</strong> {resultado['media_g1']:.2f} ± {resultado['std_g1']:.2f} (n={resultado['n_g1']})</p>
                                <p><strong>{t['position']} 2 ({grupo2}):</strong> {resultado['media_g2']:.2f} ± {resultado['std_g2']:.2f} (n={resultado['n_g2']})</p>
                                <p><strong>Teste:</strong> {resultado['teste']}</p>
                                <p><strong>p-valor:</strong> {resultado['p_valor']:.4f}</p>
                                <p><strong>Diferença:</strong> {resultado['media_g1'] - resultado['media_g2']:.2f}</p>
                                <p><strong>{'✅ Significativo' if resultado['significativo'] else '❌ Não significativo'}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig_comp = go.Figure()
                            
                            dados_comp1 = df_filtrado[df_filtrado['Posição'] == grupo1][variavel]
                            dados_comp2 = df_filtrado[df_filtrado['Posição'] == grupo2][variavel]
                            
                            fig_comp.add_trace(go.Box(
                                y=dados_comp1,
                                name=grupo1,
                                boxmean='sd',
                                marker_color='#3b82f6',
                                line_color='white'
                            ))
                            fig_comp.add_trace(go.Box(
                                y=dados_comp2,
                                name=grupo2,
                                boxmean='sd',
                                marker_color='#ef4444',
                                line_color='white'
                            ))
                            
                            fig_comp.update_layout(
                                title=f"{grupo1} vs {grupo2} - {variavel}",
                                plot_bgcolor='rgba(15, 23, 42, 0.95)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=11),
                                title_font=dict(color='#3b82f6', size=16),
                                yaxis_title=variavel
                            )
                            fig_comp.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                            fig_comp.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                            
                            st.plotly_chart(fig_comp, use_container_width=True)
                        else:
                            st.warning("⚠️ Dados insuficientes para comparação")
                    else:
                        st.info("ℹ️ Selecione grupos diferentes")
                else:
                    st.info("ℹ️ Selecione pelo menos 2 posições")
            
            # ABA 6: EXECUTIVO
            with tabs[5]:
                st.markdown(f"<h3>{t['tab_executive']}</h3>", unsafe_allow_html=True)
                
                st.markdown("### 🆚 Comparação de Atletas")
                if len(atletas) >= 2:
                    col_atl1, col_atl2 = st.columns(2)
                    with col_atl1:
                        atl1 = st.selectbox(
                            "Atleta 1" if st.session_state.idioma == 'pt' else "Athlete 1", 
                            atletas, 
                            index=0, 
                            key="atleta1_comp_select"
                        )
                    with col_atl2:
                        atl2 = st.selectbox(
                            "Atleta 2" if st.session_state.idioma == 'pt' else "Athlete 2", 
                            atletas, 
                            index=min(1, len(atletas)-1), 
                            key="atleta2_comp_select"
                        )
                    
                    if atl1 != atl2:
                        vars_comp = st.multiselect(
                            "Variáveis para comparar:" if st.session_state.idioma == 'pt' else "Variables to compare:",
                            st.session_state.variaveis_quantitativas,
                            default=st.session_state.vars_comp if st.session_state.vars_comp else st.session_state.variaveis_quantitativas[:3],
                            key="vars_comp_select"
                        )
                        
                        if len(vars_comp) >= 1:
                            comparar_atletas(df_filtrado, atl1, atl2, vars_comp, t)
                    else:
                        st.info("Selecione atletas diferentes para comparação" if st.session_state.idioma == 'pt' else "Select different athletes")
                else:
                    st.info("Selecione pelo menos 2 atletas para comparação" if st.session_state.idioma == 'pt' else "Select at least 2 athletes")
                
                st.markdown("---")
                
                sistema_anotacoes(t)
            
            # Dados brutos
            with st.expander("📋 Visualizar dados filtrados" if st.session_state.idioma == 'pt' else "📋 View filtered data"):
                st.dataframe(df_filtrado, use_container_width=True)
    
    st.session_state.processar_click = False

elif st.session_state.df_completo is None:
    # Tela inicial
    t = translations[st.session_state.idioma]
    st.info("👈 **Faça upload dos arquivos CSV para iniciar a análise quântica**" if st.session_state.idioma == 'pt' else "👈 **Upload CSV files to start quantum analysis**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(t['file_format'])
        
        exemplo_data = {
            'Nome-Período-Minuto': [
                'João-1º Tempo-00:00-01:00',
                'Maria-2º Tempo-05:00-06:00',
                'Pedro-1º Tempo-44:00-45:00',
                'Ana-Prorrogação-90:00-91:00',
                'Carlos-2º Tempo-22:00-23:00'
            ],
            'Posição': ['Atacante', 'Meio-campo', 'Zagueiro', 'Atacante', 'Goleiro'],
            'Distância (m)': [250, 180, 200, 120, 45],
            'Velocidade (km/h)': [23, 29, 33, 25, 15],
            'Aceleração (m/s²)': [3.6, 4.2, 4.9, 3.8, 2.8]
        }
        
        df_exemplo = pd.DataFrame(exemplo_data)
        st.dataframe(df_exemplo, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown(f"""
        <div class="quantum-card">
            <h4>{t['components']}</h4>
            <hr style="border-color: #334155;">
            <p>{t['name_ex']}</p>
            <p>{t['period_ex']}</p>
            <p>{t['minute_ex']}</p>
            <p>{t['position_ex']}</p>
        </div>
        
        <div class="quantum-card" style="margin-top: 20px;">
            <h4>{t['tip']}</h4>
            <hr style="border-color: #334155;">
            <p>{t['tip_text']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander(t['multi_file_ex']):
        st.markdown("""
        ### Instruções para múltiplos arquivos:
        1. Prepare seus arquivos CSV com a **mesma estrutura** de colunas
        2. Selecione todos os arquivos desejados
        3. O sistema verificará compatibilidade e concatenará automaticamente
        4. Todos os dados serão combinados para análise unificada
        """ if st.session_state.idioma == 'pt' else """
        ### Instructions for multiple files:
        1. Prepare your CSV files with the **same column structure**
        2. Select all desired files
        3. The system will check compatibility and concatenate automatically
        4. All data will be combined for unified analysis
        """)

elif st.session_state.dados_processados:
    t = translations[st.session_state.idioma]
    st.info("👈 **Selecione os filtros na sidebar e clique em Processar Análise**" if st.session_state.idioma == 'pt' else "👈 **Select filters in sidebar and click Process Analysis**")
    
    with st.expander("📋 Preview dos dados carregados" if st.session_state.idioma == 'pt' else "📋 Loaded data preview"):
        if st.session_state.upload_files_names:
            st.caption(f"**{t['upload']}:** {', '.join(st.session_state.upload_files_names)}")
            st.markdown("---")
        
        st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
        st.caption(f"**{t['observations']}:** {len(st.session_state.df_completo)}")
        st.caption(f"**{t['variable']}s:** {', '.join(st.session_state.variaveis_quantitativas)}")
        if st.session_state.todos_posicoes:
            st.caption(f"**{t['positions']}:** {', '.join(st.session_state.todos_posicoes)}")
        if st.session_state.todos_periodos:
            st.caption(f"**{t['periods']}:** {', '.join(st.session_state.todos_periodos)}")