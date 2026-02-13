import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import time
import warnings
warnings.filterwarnings('ignore')

# Configuração da página - DEVE ser a primeira chamada
st.set_page_config(
    page_title="Sports Science Analytics Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🏃"
)

# ============================================================================
# TRADUÇÕES
# ============================================================================

translations = {
    'pt': {
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Dashboard Profissional para Análise de Desempenho Esportivo',
        'upload': 'Upload dos Dados',
        'variable': 'Variável',
        'position': 'Posição',
        'period': 'Período',
        'athlete': 'Atleta',
        'config': 'Configurações',
        'tab_distribution': '📊 Distribuição',
        'tab_temporal': '📈 Estatísticas & Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlações',
        'tab_comparison': '⚖️ Comparações',
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
        'max_value': 'VALOR MÁXIMO',
        'min_value': 'VALOR MÍNIMO',
        'minute_of_max': 'Minuto do Máx',
        'minute_of_min': 'Minuto do Mín',
        'threshold_80': 'LIMIAR 80%',
        'critical_events': 'EVENTOS CRÍTICOS',
        'above_threshold': 'acima do limiar',
        'intensity_zones': '🎚️ Zonas de Intensidade',
        'zone_method': 'Método de definição',
        'percentiles': 'Percentis',
        'based_on_max': 'Baseado no Máximo',
        'very_low': 'Muito Baixa',
        'low': 'Baixa',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muito Alta',
        'process': '🚀 Processar Análise',
        'descriptive_stats': '📊 Estatísticas Descritivas',
        'confidence_interval': '🎯 Intervalo de Confiança (95%)',
        'normality_test': '🧪 Teste de Normalidade',
        'summary_by_group': '🏃 Resumo por Atleta, Posição e Período',
        'symmetric': 'Aproximadamente simétrica',
        'moderate_skew': 'Moderadamente assimétrica',
        'high_skew': 'Fortemente assimétrica',
        'leptokurtic': 'Leptocúrtica (caudas pesadas)',
        'platykurtic': 'Platicúrtica (caudas leves)',
        'mesokurtic': 'Mesocúrtica (normal)',
        'strong_positive': 'Correlação forte positiva',
        'moderate_positive': 'Correlação moderada positiva',
        'weak_positive': 'Correlação fraca positiva',
        'very_weak_positive': 'Correlação muito fraca positiva',
        'very_weak_negative': 'Correlação muito fraca negativa',
        'weak_negative': 'Correlação fraca negativa',
        'moderate_negative': 'Correlação moderada negativa',
        'strong_negative': 'Correlação forte negativa',
        'iqr_title': '📌 O que é IQR?',
        'iqr_explanation': 'IQR (Intervalo Interquartil) é a diferença entre Q3 e Q1, representando os 50% centrais dos dados.',
        'step1': '👈 **Passo 1:** Faça upload de um ou mais arquivos CSV para começar',
        'step2': '👈 **Passo 2:** Selecione os filtros e clique em Processar Análise',
        'file_format': '### 📋 Formato esperado do arquivo:',
        'col1_desc': '**Primeira coluna:** Identificação no formato `Nome-Período-Minuto`',
        'col2_desc': '**Segunda coluna:** Posição do atleta',
        'col3_desc': '**Demais colunas (3+):** Variáveis numéricas para análise',
        'components': '📌 Componentes',
        'name_ex': 'Nome: Mariano, Maria, Joao...',
        'period_ex': 'Período: 1 TEMPO, SEGUNDO TEMPO...',
        'minute_ex': 'Minuto: 00:00-01:00, 05:00-06:00...',
        'position_ex': 'Posição: Atacante, Meio-campo...',
        'tip': '💡 Dica',
        'tip_text': 'Você pode selecionar múltiplos arquivos CSV com a mesma estrutura.',
        'multi_file_ex': '📁 Exemplo com múltiplos arquivos',
        'multi_file_text': '''
            ### Carregando múltiplos arquivos:
            1. Prepare seus arquivos CSV com a **mesma estrutura** de colunas
            2. Selecione todos os arquivos desejados
            3. O sistema verificará compatibilidade e concatenará automaticamente
        '''
    },
    'en': {
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Professional Dashboard for Sports Performance Analysis',
        'upload': 'Data Upload',
        'variable': 'Variable',
        'position': 'Position',
        'period': 'Period',
        'athlete': 'Athlete',
        'config': 'Settings',
        'tab_distribution': '📊 Distribution',
        'tab_temporal': '📈 Statistics & Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlations',
        'tab_comparison': '⚖️ Comparisons',
        'positions': 'Positions',
        'periods': 'Periods',
        'athletes': 'Athletes',
        'observations': 'Observations',
        'mean': 'Mean',
        'median': 'Median',
        'mode': 'Mode',
        'std': 'Standard Deviation',
        'variance': 'Variance',
        'cv': 'Coefficient of Variation',
        'min': 'Minimum',
        'max': 'Maximum',
        'amplitude': 'Range',
        'q1': 'Q1 (25%)',
        'q3': 'Q3 (75%)',
        'iqr': 'IQR',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis',
        'max_value': 'MAXIMUM VALUE',
        'min_value': 'MINIMUM VALUE',
        'minute_of_max': 'Max Minute',
        'minute_of_min': 'Min Minute',
        'threshold_80': '80% THRESHOLD',
        'critical_events': 'CRITICAL EVENTS',
        'above_threshold': 'above threshold',
        'intensity_zones': '🎚️ Intensity Zones',
        'zone_method': 'Definition method',
        'percentiles': 'Percentiles',
        'based_on_max': 'Based on Maximum',
        'very_low': 'Very Low',
        'low': 'Low',
        'moderate': 'Moderate',
        'high': 'High',
        'very_high': 'Very High',
        'process': '🚀 Process Analysis',
        'descriptive_stats': '📊 Descriptive Statistics',
        'confidence_interval': '🎯 Confidence Interval (95%)',
        'normality_test': '🧪 Normality Test',
        'summary_by_group': '🏃 Summary by Athlete, Position and Period',
        'symmetric': 'Approximately symmetric',
        'moderate_skew': 'Moderately skewed',
        'high_skew': 'Highly skewed',
        'leptokurtic': 'Leptokurtic (heavy tails)',
        'platykurtic': 'Platykurtic (light tails)',
        'mesokurtic': 'Mesokurtic (normal)',
        'strong_positive': 'Strong positive correlation',
        'moderate_positive': 'Moderate positive correlation',
        'weak_positive': 'Weak positive correlation',
        'very_weak_positive': 'Very weak positive correlation',
        'very_weak_negative': 'Very weak negative correlation',
        'weak_negative': 'Weak negative correlation',
        'moderate_negative': 'Moderate negative correlation',
        'strong_negative': 'Strong negative correlation',
        'iqr_title': '📌 What is IQR?',
        'iqr_explanation': 'IQR (Interquartile Range) is the difference between Q3 and Q1, representing the middle 50% of data.',
        'step1': '👈 **Step 1:** Upload one or more CSV files to start',
        'step2': '👈 **Step 2:** Select filters and click Process Analysis',
        'file_format': '### 📋 Expected file format:',
        'col1_desc': '**First column:** Identification in `Name-Period-Minute` format',
        'col2_desc': '**Second column:** Athlete position',
        'col3_desc': '**Other columns (3+):** Numerical variables for analysis',
        'components': '📌 Components',
        'name_ex': 'Name: Mariano, Maria, Joao...',
        'period_ex': 'Period: 1 TEMPO, SEGUNDO TEMPO...',
        'minute_ex': 'Minute: 00:00-01:00, 05:00-06:00...',
        'position_ex': 'Position: Atacante, Meio-campo...',
        'tip': '💡 Tip',
        'tip_text': 'You can select multiple CSV files with the same structure.',
        'multi_file_ex': '📁 Example with multiple files',
        'multi_file_text': '''
            ### Loading multiple files:
            1. Prepare your CSV files with the **same column structure**
            2. Select all desired files
            3. The system will check compatibility and concatenate automatically
        '''
    }
}

# ============================================================================
# CSS PERSONALIZADO
# ============================================================================

st.markdown("""
<style>
    .stApp {
        background: #0f172a;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .css-1d391kg, .css-1wrcr25 {
        background: #020617 !important;
        border-right: 1px solid #334155;
    }
    
    .sidebar-title {
        color: #f8fafc !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #3b82f6;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        color: white !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.3);
        border-color: #3b82f6;
    }
    
    .metric-card .icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
        color: #3b82f6;
    }
    
    .metric-card h3 {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
        font-weight: 500;
    }
    
    .metric-card h2 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .time-metric-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 10px 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    .time-metric-card:hover {
        transform: translateX(5px);
        border-left-width: 6px;
    }
    
    .time-metric-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .time-metric-card .value {
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    .time-metric-card .sub-value {
        color: #64748b;
        font-size: 0.8rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.3);
        text-align: center;
        color: white;
        margin: 10px 0;
        animation: pulse 2s infinite;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .warning-card .label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .warning-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .warning-card .sub-label {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    .zone-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .zone-card:hover {
        transform: translateX(5px);
    }
    
    .zone-card .zone-name {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .zone-card .zone-value {
        font-size: 1.2rem;
        color: white;
        font-weight: 600;
    }
    
    .zone-card .zone-count {
        font-size: 0.9rem;
        color: #3b82f6;
    }
    
    h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    h2 {
        color: white !important;
        font-size: 1.8rem;
        font-weight: 600;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }
    
    h3 {
        color: #3b82f6 !important;
        font-size: 1.4rem;
        font-weight: 500;
    }
    
    h4 {
        color: #8b5cf6 !important;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        padding: 8px;
        border-radius: 50px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 10px 20px;
        font-weight: 500;
        color: #94a3b8 !important;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
    }
    
    .metric-container {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-container:hover {
        border-color: #3b82f6;
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.2);
    }
    
    .metric-container h4 {
        color: #3b82f6 !important;
        margin-bottom: 15px;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-container p {
        color: #e2e8f0 !important;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    
    .metric-container strong {
        color: #8b5cf6;
    }
    
    .dataframe {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        color: white !important;
    }
    
    .dataframe th {
        background: #1e293b !important;
        color: #3b82f6 !important;
        font-weight: 600;
        padding: 12px !important;
    }
    
    .dataframe td {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
        padding: 10px !important;
    }
    
    p, li, .caption, .stMarkdown {
        color: #cbd5e1 !important;
        line-height: 1.6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>🏃 Sports Science Analytics Pro</h1>
        <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 10px;">
            Professional Dashboard for Elite Performance Analysis
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
            <span style="background: #3b82f6; color: white; padding: 5px 15px; border-radius: 50px; font-size: 0.9rem;">⚡ Real-time</span>
            <span style="background: #8b5cf6; color: white; padding: 5px 15px; border-radius: 50px; font-size: 0.9rem;">📊 Statistical</span>
            <span style="background: #10b981; color: white; padding: 5px 15px; border-radius: 50px; font-size: 0.9rem;">🎯 Precision</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE - Inicialização completa
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
if 'upload_files_names' not in st.session_state:
    st.session_state.upload_files_names = []
if 'idioma' not in st.session_state:
    st.session_state.idioma = 'pt'
if 'processar_click' not in st.session_state:
    st.session_state.processar_click = False
if 'dados_processados' not in st.session_state:
    st.session_state.dados_processados = False
if 'metodo_zona' not in st.session_state:
    st.session_state.metodo_zona = 'percentis'
if 'grupo1' not in st.session_state:
    st.session_state.grupo1 = None
if 'grupo2' not in st.session_state:
    st.session_state.grupo2 = None
if 'n_classes' not in st.session_state:
    st.session_state.n_classes = 5

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def extrair_periodo(texto):
    """Extrai o período entre o nome e o minuto"""
    try:
        texto = str(texto)
        primeiro_hifen = texto.find('-')
        if primeiro_hifen == -1:
            return ""
        if len(texto) < 13:
            return ""
        periodo = texto[primeiro_hifen + 1:-13].strip()
        return periodo
    except:
        return ""

def verificar_estruturas_arquivos(dataframes):
    """Verifica se todos os dataframes têm a mesma estrutura"""
    if not dataframes:
        return False, []
    primeira_estrutura = dataframes[0].columns.tolist()
    for df in dataframes[1:]:
        if df.columns.tolist() != primeira_estrutura:
            return False, primeira_estrutura
    return True, primeira_estrutura

def metric_card(titulo, valor, icone, cor_gradiente):
    """Cria um card de métrica estilizado"""
    st.markdown(f"""
    <div class="metric-card fade-in">
        <div class="icon">{icone}</div>
        <h3>{titulo}</h3>
        <h2>{valor}</h2>
    </div>
    """, unsafe_allow_html=True)

def time_metric_card(label, valor, sub_label="", cor="#3b82f6"):
    """Cria um card para métricas temporais"""
    st.markdown(f"""
    <div class="time-metric-card" style="border-left-color: {cor};">
        <div class="label">{label}</div>
        <div class="value">{valor}</div>
        <div class="sub-value">{sub_label}</div>
    </div>
    """, unsafe_allow_html=True)

def warning_card(titulo, valor, subtitulo, icone="⚠️"):
    """Cria um card de alerta para eventos críticos"""
    st.markdown(f"""
    <div class="warning-card fade-in">
        <div class="label">{icone} {titulo}</div>
        <div class="value">{valor}</div>
        <div class="sub-label">{subtitulo}</div>
    </div>
    """, unsafe_allow_html=True)

def calcular_cv(media, desvio):
    """Calcula o coeficiente de variação"""
    if media != 0 and not np.isnan(media) and not np.isnan(desvio):
        return (desvio / media) * 100
    return 0

def extrair_minuto_do_extremo(df, coluna_valor, coluna_minuto, extremo='max'):
    """Extrai o minuto do valor máximo ou mínimo"""
    try:
        if df.empty or len(df) == 0:
            return "N/A"
        df_reset = df.reset_index(drop=True)
        if extremo == 'max':
            idx_extremo = df_reset[coluna_valor].idxmax()
        else:
            idx_extremo = df_reset[coluna_valor].idxmin()
        if pd.notna(idx_extremo) and idx_extremo < len(df_reset):
            return df_reset.loc[idx_extremo, coluna_minuto]
        return "N/A"
    except:
        return "N/A"

def criar_zonas_intensidade(df, variavel, metodo='percentis'):
    """Cria zonas de intensidade baseadas em percentis ou no máximo"""
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
    """Compara dois grupos com teste estatístico apropriado"""
    try:
        dados_grupo1 = df[df['Posição'] == grupo1][variavel].dropna()
        dados_grupo2 = df[df['Posição'] == grupo2][variavel].dropna()
        if len(dados_grupo1) < 3 or len(dados_grupo2) < 3:
            return None
        _, p1 = stats.shapiro(dados_grupo1)
        _, p2 = stats.shapiro(dados_grupo2)
        if p1 > 0.05 and p2 > 0.05:
            _, p_valor = stats.ttest_ind(dados_grupo1, dados_grupo2)
            teste = "Teste t de Student"
        else:
            _, p_valor = stats.mannwhitneyu(dados_grupo1, dados_grupo2)
            teste = "Teste de Mann-Whitney"
        return {
            'teste': teste,
            'p_valor': p_valor,
            'significativo': p_valor < 0.05,
            'media_g1': dados_grupo1.mean(),
            'media_g2': dados_grupo2.mean(),
            'std_g1': dados_grupo1.std(),
            'std_g2': dados_grupo2.std(),
            'n_g1': len(dados_grupo1),
            'n_g2': len(dados_grupo2)
        }
    except:
        return None

# ============================================================================
# CALLBACKS
# ============================================================================

def atualizar_metodo_zona():
    """Callback para atualizar método de zona"""
    if st.session_state.metodo_zona_radio == translations[st.session_state.idioma]['percentiles']:
        st.session_state.metodo_zona = 'percentis'
    else:
        st.session_state.metodo_zona = 'based_on_max'

def atualizar_grupo1():
    """Callback para atualizar grupo 1"""
    st.session_state.grupo1 = st.session_state.grupo1_select

def atualizar_grupo2():
    """Callback para atualizar grupo 2"""
    st.session_state.grupo2 = st.session_state.grupo2_select

def processar_click():
    """Callback para botão processar"""
    st.session_state.processar_click = True

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🌐 Idioma / Language</h2>", unsafe_allow_html=True)
    
    idioma = st.selectbox(
        "", 
        ['pt', 'en'], 
        index=0 if st.session_state.idioma == 'pt' else 1,
        label_visibility="collapsed",
        key="idioma_selector"
    )
    
    if idioma != st.session_state.idioma:
        st.session_state.idioma = idioma
        st.rerun()
    
    t = translations[st.session_state.idioma]
    
    st.markdown("---")
    st.markdown(f"<h2 class='sidebar-title'>📂 {t['upload']}</h2>", unsafe_allow_html=True)
    
    upload_files = st.file_uploader(
        "",
        type=['csv'],
        accept_multiple_files=True,
        help=t['tip_text'],
        key="file_uploader"
    )
    
    if upload_files and len(upload_files) > 0:
        with st.spinner('🔄 Processando...'):
            time.sleep(0.5)
            try:
                dataframes = []
                arquivos_validos = []
                
                for uploaded_file in upload_files:
                    try:
                        data = pd.read_csv(uploaded_file)
                        if data.shape[1] >= 3 and not data.empty:
                            dataframes.append(data)
                            arquivos_validos.append(uploaded_file.name)
                    except:
                        pass
                
                if dataframes:
                    estruturas_ok, _ = verificar_estruturas_arquivos(dataframes)
                    if not estruturas_ok:
                        st.error("❌ " + ("Arquivos com estruturas diferentes" if st.session_state.idioma == 'pt' else "Files with different structures"))
                        st.stop()
                    
                    data = pd.concat(dataframes, ignore_index=True)
                    
                    if data.shape[1] >= 3 and not data.empty:
                        primeira_coluna = data.iloc[:, 0].astype(str)
                        segunda_coluna = data.iloc[:, 1].astype(str)
                        
                        nomes = primeira_coluna.str.split('-').str[0].str.strip()
                        minutos = primeira_coluna.str[-13:].str.strip()
                        periodos = primeira_coluna.apply(extrair_periodo)
                        
                        periodos_unicos = sorted([p for p in periodos.unique() if p and p.strip() != ""])
                        posicoes_unicas = sorted([p for p in segunda_coluna.unique() if p and p.strip() != ""])
                        
                        variaveis_quant = []
                        dados_quantitativos = {}
                        
                        for col_idx in range(2, data.shape[1]):
                            nome_var = data.columns[col_idx]
                            valores = pd.to_numeric(data.iloc[:, col_idx], errors='coerce')
                            if not valores.dropna().empty:
                                variaveis_quant.append(nome_var)
                                dados_quantitativos[nome_var] = valores.reset_index(drop=True)
                        
                        if variaveis_quant:
                            df_completo = pd.DataFrame({
                                'Nome': nomes.reset_index(drop=True),
                                'Posição': segunda_coluna.reset_index(drop=True),
                                'Período': periodos.reset_index(drop=True),
                                'Minuto': minutos.reset_index(drop=True)
                            })
                            
                            for var_nome, var_valores in dados_quantitativos.items():
                                df_completo[var_nome] = var_valores
                            
                            df_completo = df_completo[df_completo['Nome'].str.len() > 0]
                            
                            if not df_completo.empty:
                                st.session_state.df_completo = df_completo
                                st.session_state.variaveis_quantitativas = variaveis_quant
                                st.session_state.atletas_selecionados = sorted(df_completo['Nome'].unique())
                                st.session_state.todos_posicoes = posicoes_unicas
                                st.session_state.posicoes_selecionadas = posicoes_unicas.copy()
                                st.session_state.todos_periodos = periodos_unicos
                                st.session_state.periodos_selecionados = periodos_unicos.copy()
                                st.session_state.upload_files_names = arquivos_validos
                                
                                if variaveis_quant and st.session_state.variavel_selecionada is None:
                                    st.session_state.variavel_selecionada = variaveis_quant[0]
                                
                                sucesso_msg = ("arquivo(s) carregado(s)" if st.session_state.idioma == 'pt' else "file(s) loaded")
                                st.success(f"✅ {len(arquivos_validos)} {sucesso_msg}")
                                st.rerun()
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
    
    if st.session_state.df_completo is not None:
        st.markdown("---")
        
        if st.session_state.variaveis_quantitativas:
            st.markdown(f"<h2 class='sidebar-title'>📈 {t['variable']}</h2>", unsafe_allow_html=True)
            
            current_index = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                current_index = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            variavel_selecionada = st.selectbox(
                "",
                options=st.session_state.variaveis_quantitativas,
                index=current_index,
                label_visibility="collapsed",
                key="variavel_selector"
            )
            
            if variavel_selecionada != st.session_state.variavel_selecionada:
                st.session_state.variavel_selecionada = variavel_selecionada
                st.session_state.dados_processados = False
                st.rerun()
            
            df_temp = st.session_state.df_completo[variavel_selecionada].dropna()
            if not df_temp.empty:
                st.caption(f"📊 {len(df_temp)} obs | {t['mean']}: {df_temp.mean():.2f}")
        
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown(f"<h2 class='sidebar-title'>📍 {t['position']}</h2>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox(
                f"Selecionar todas as {t['position'].lower()}s" if st.session_state.idioma == 'pt' else f"Select all {t['position'].lower()}s",
                value=True if len(st.session_state.posicoes_selecionadas) == len(st.session_state.todos_posicoes) else False,
                key="todos_posicoes_check"
            )
            
            if selecionar_todos:
                if st.session_state.posicoes_selecionadas != st.session_state.todos_posicoes:
                    st.session_state.posicoes_selecionadas = st.session_state.todos_posicoes.copy()
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                posicoes_sel = st.multiselect(
                    "",
                    options=st.session_state.todos_posicoes,
                    default=st.session_state.posicoes_selecionadas,
                    label_visibility="collapsed",
                    key="posicoes_selector"
                )
                if posicoes_sel != st.session_state.posicoes_selecionadas:
                    st.session_state.posicoes_selecionadas = posicoes_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown(f"<h2 class='sidebar-title'>📅 {t['period']}</h2>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox(
                f"Selecionar todos os {t['period'].lower()}s" if st.session_state.idioma == 'pt' else f"Select all {t['period'].lower()}s",
                value=True if len(st.session_state.periodos_selecionados) == len(st.session_state.todos_periodos) else False,
                key="todos_periodos_check"
            )
            
            if selecionar_todos:
                if st.session_state.periodos_selecionados != st.session_state.todos_periodos:
                    st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                periodos_sel = st.multiselect(
                    "",
                    options=st.session_state.todos_periodos,
                    default=st.session_state.periodos_selecionados,
                    label_visibility="collapsed",
                    key="periodos_selector"
                )
                if periodos_sel != st.session_state.periodos_selecionados:
                    st.session_state.periodos_selecionados = periodos_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        if st.session_state.atletas_selecionados:
            st.markdown("---")
            st.markdown(f"<h2 class='sidebar-title'>👤 {t['athlete']}</h2>", unsafe_allow_html=True)
            
            df_temp = st.session_state.df_completo.copy()
            if st.session_state.posicoes_selecionadas:
                df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
            if st.session_state.periodos_selecionados:
                df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
            
            atletas_disponiveis = sorted(df_temp['Nome'].unique())
            
            selecionar_todos = st.checkbox(
                f"Selecionar todos os {t['athlete'].lower()}s" if st.session_state.idioma == 'pt' else f"Select all {t['athlete'].lower()}s",
                value=True if len(st.session_state.atletas_selecionados) == len(atletas_disponiveis) and len(atletas_disponiveis) > 0 else False,
                key="todos_atletas_check"
            )
            
            if selecionar_todos:
                if st.session_state.atletas_selecionados != atletas_disponiveis:
                    st.session_state.atletas_selecionados = atletas_disponiveis
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                atletas_sel = st.multiselect(
                    "",
                    options=atletas_disponiveis,
                    default=[a for a in st.session_state.atletas_selecionados if a in atletas_disponiveis],
                    label_visibility="collapsed",
                    key="atletas_selector"
                )
                if atletas_sel != st.session_state.atletas_selecionados:
                    st.session_state.atletas_selecionados = atletas_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown(f"<h2 class='sidebar-title'>⚙️ {t['config']}</h2>", unsafe_allow_html=True)
        
        n_classes = st.slider(f"{t['config']}:", 3, 20, 5, key="classes_slider")
        st.session_state.n_classes = n_classes
        
        st.markdown("---")
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button(t['process'], use_container_width=True, disabled=not pode_processar, key="process_button", on_click=processar_click):
            pass

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.processar_click and st.session_state.df_completo is not None:
    
    with st.spinner('🔄 ' + ("Gerando análises..." if st.session_state.idioma == 'pt' else "Generating analysis...")):
        time.sleep(0.5)
        
        df_completo = st.session_state.df_completo
        atletas_selecionados = st.session_state.atletas_selecionados
        posicoes_selecionadas = st.session_state.posicoes_selecionadas
        periodos_selecionados = st.session_state.periodos_selecionados
        variavel_analise = st.session_state.variavel_selecionada
        n_classes = st.session_state.n_classes
        
        df_filtrado = df_completo[
            df_completo['Nome'].isin(atletas_selecionados) & 
            df_completo['Posição'].isin(posicoes_selecionadas) &
            df_completo['Período'].isin(periodos_selecionados)
        ].copy()
        
        df_filtrado = df_filtrado.dropna(subset=[variavel_analise])
        
        if df_filtrado.empty:
            st.warning("⚠️ " + ("Nenhum dado encontrado" if st.session_state.idioma == 'pt' else "No data found"))
        else:
            st.session_state.dados_processados = True
            t = translations[st.session_state.idioma]
            
            st.markdown(f"<h2>📊 {t['title'].split('Pro')[0] if 'Pro' in t['title'] else 'Visão Geral'}</h2>", unsafe_allow_html=True)
            
            cols = st.columns(4)
            with cols[0]:
                metric_card(t['positions'], len(posicoes_selecionadas), "📍", "linear-gradient(135deg, #3b82f6, #2563eb)")
            with cols[1]:
                metric_card(t['periods'], len(periodos_selecionados), "📅", "linear-gradient(135deg, #8b5cf6, #7c3aed)")
            with cols[2]:
                metric_card(t['athletes'], len(atletas_selecionados), "👥", "linear-gradient(135deg, #ef4444, #dc2626)")
            with cols[3]:
                metric_card(t['observations'], len(df_filtrado), "📊", "linear-gradient(135deg, #f59e0b, #d97706)")
            
            st.markdown("---")
            
            tab_titles = [
                t['tab_distribution'], 
                t['tab_temporal'], 
                t['tab_boxplots'], 
                t['tab_correlation'], 
                t['tab_comparison']
            ]
            
            tabs = st.tabs(tab_titles)
            
            # ABA 1: DISTRIBUIÇÃO
            with tabs[0]:
                st.markdown(f"<h3>{t['tab_distribution']}</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    dados_hist = df_filtrado[variavel_analise].dropna()
                    
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
                        title=f"Histograma - {variavel_analise} ({n_classes} classes)",
                        plot_bgcolor='rgba(30, 41, 59, 0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3b82f6', size=16),
                        xaxis_title=variavel_analise,
                        yaxis_title="Frequência",
                        showlegend=False,
                        bargap=0.1
                    )
                    fig_hist.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    fig_hist.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    dados_qq = df_filtrado[variavel_analise].dropna()
                    quantis_teoricos = stats.norm.ppf(np.linspace(0.01, 0.99, len(dados_qq)))
                    quantis_observados = np.sort(dados_qq)
                    
                    z = np.polyfit(quantis_teoricos, quantis_observados, 1)
                    linha_ref = np.poly1d(z)
                    
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
                        name='Referência',
                        line=dict(color='#ef4444', width=2)
                    ))
                    
                    fig_qq.update_layout(
                        title=f"QQ Plot - {variavel_analise}",
                        plot_bgcolor='rgba(30, 41, 59, 0.8)',
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
                st.markdown(f"<h4>📋 {t['tab_distribution'].replace('📊 ', '')} ({n_classes} classes)</h4>", unsafe_allow_html=True)
                
                minimo = df_filtrado[variavel_analise].min()
                maximo = df_filtrado[variavel_analise].max()
                amplitude = maximo - minimo
                largura_classe = amplitude / n_classes if amplitude > 0 else 1
                
                limites = [minimo + i * largura_classe for i in range(n_classes + 1)]
                rotulos = [f"[{limites[i]:.2f} - {limites[i+1]:.2f})" for i in range(n_classes)]
                
                categorias = pd.cut(df_filtrado[variavel_analise], bins=limites, labels=rotulos, include_lowest=True, right=False)
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
            
            # ABA 2: ESTATÍSTICAS & TEMPORAL
            with tabs[1]:
                st.markdown(f"<h3>{t['tab_temporal']}</h3>", unsafe_allow_html=True)
                
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                valor_maximo = df_tempo[variavel_analise].max()
                valor_minimo = df_tempo[variavel_analise].min()
                minuto_maximo = extrair_minuto_do_extremo(df_tempo, variavel_analise, 'Minuto', 'max')
                minuto_minimo = extrair_minuto_do_extremo(df_tempo, variavel_analise, 'Minuto', 'min')
                media_tempo = df_tempo[variavel_analise].mean()
                limiar_80 = valor_maximo * 0.8
                
                eventos_acima_80 = (df_tempo[variavel_analise] > limiar_80).sum()
                percentual_acima_80 = (eventos_acima_80 / len(df_tempo)) * 100 if len(df_tempo) > 0 else 0
                
                cols_t = st.columns(5)
                with cols_t[0]:
                    time_metric_card(t['max_value'], f"{valor_maximo:.2f}", f"{t['minute_of_max']}: {minuto_maximo}", "#ef4444")
                with cols_t[1]:
                    time_metric_card(t['min_value'], f"{valor_minimo:.2f}", f"{t['minute_of_min']}: {minuto_minimo}", "#10b981")
                with cols_t[2]:
                    time_metric_card(t['mean'], f"{media_tempo:.2f}", t['mean'], "#3b82f6")
                with cols_t[3]:
                    time_metric_card(t['threshold_80'], f"{limiar_80:.2f}", f"80% do máx ({valor_maximo:.2f})", "#f59e0b")
                with cols_t[4]:
                    warning_card(t['critical_events'], f"{eventos_acima_80}", f"{percentual_acima_80:.1f}% {t['above_threshold']}", "⚠️")
                
                st.markdown("---")
                st.markdown(f"<h4>{t['intensity_zones']}</h4>", unsafe_allow_html=True)
                
                # Radio button com callback
                metodo_opcoes = [t['percentiles'], t['based_on_max']]
                index_atual = 0 if st.session_state.metodo_zona == 'percentis' else 1
                
                st.radio(
                    t['zone_method'],
                    metodo_opcoes,
                    index=index_atual,
                    key="metodo_zona_radio",
                    on_change=atualizar_metodo_zona
                )
                
                # Usar o valor do session state
                metodo_interno = st.session_state.metodo_zona
                zonas = criar_zonas_intensidade(df_filtrado, variavel_analise, metodo_interno)
                
                if zonas:
                    cores_zonas = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981']
                    st.markdown("##### Limiares das Zonas:")
                    cols_zone = st.columns(5)
                    for i, (zona, limite) in enumerate(zonas.items()):
                        with cols_zone[i]:
                            if i == 0:
                                count = df_filtrado[variavel_analise] <= limite
                            else:
                                limite_anterior = list(zonas.values())[i-1]
                                count = (df_filtrado[variavel_analise] > limite_anterior) & (df_filtrado[variavel_analise] <= limite)
                            n_obs = count.sum()
                            st.markdown(f"""
                            <div class="zone-card" style="border-left-color: {cores_zonas[i]};">
                                <div class="zone-name">{zona}</div>
                                <div class="zone-value">{limite:.1f}</div>
                                <div class="zone-count">{n_obs} obs ({n_obs/len(df_filtrado)*100:.0f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"<h4>{t['tab_temporal']}</h4>", unsafe_allow_html=True)
                
                fig_tempo = go.Figure()
                
                cores_pontos = ['#ef4444' if v > limiar_80 else '#3b82f6' for v in df_tempo[variavel_analise]]
                
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'],
                    y=df_tempo[variavel_analise],
                    mode='lines',
                    line=dict(color='#94a3b8', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'],
                    y=df_tempo[variavel_analise],
                    mode='markers',
                    marker=dict(color=cores_pontos, size=10, line=dict(color='white', width=1)),
                    showlegend=False,
                    hovertemplate='Minuto: %{x}<br>Valor: %{y:.2f}<br>Status: %{text}<extra></extra>',
                    text=['⚠️ ACIMA' if v > limiar_80 else '✅ ABAIXO' for v in df_tempo[variavel_analise]]
                ))
                
                fig_tempo.add_hline(y=media_tempo, line_dash="dash", line_color="#3b82f6", line_width=2,
                                   annotation_text=f"{t['mean']}: {media_tempo:.2f}", annotation_position="left")
                fig_tempo.add_hline(y=limiar_80, line_dash="dot", line_color="#f59e0b", line_width=2,
                                   annotation_text=f"{t['threshold_80']}: {limiar_80:.2f}", annotation_position="right")
                fig_tempo.add_hline(y=valor_maximo, line_dash="solid", line_color="#ef4444", line_width=2,
                                   annotation_text=f"{t['max_value']}: {valor_maximo:.2f}", annotation_position="right")
                fig_tempo.add_hline(y=valor_minimo, line_dash="solid", line_color="#10b981", line_width=2,
                                   annotation_text=f"{t['min_value']}: {valor_minimo:.2f}", annotation_position="left")
                
                fig_tempo.update_layout(
                    title=f"{t['tab_temporal']} - {variavel_analise}",
                    plot_bgcolor='rgba(30, 41, 59, 0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3b82f6', size=16),
                    xaxis_title="Minuto",
                    yaxis_title=variavel_analise,
                    hovermode='closest'
                )
                fig_tempo.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'), tickangle=-45)
                fig_tempo.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                
                st.plotly_chart(fig_tempo, use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"<h4>{t['descriptive_stats']}</h4>", unsafe_allow_html=True)
                
                media = df_filtrado[variavel_analise].mean()
                desvio = df_filtrado[variavel_analise].std()
                mediana = df_filtrado[variavel_analise].median()
                moda = df_filtrado[variavel_analise].mode().iloc[0] if not df_filtrado[variavel_analise].mode().empty else 'N/A'
                variancia = df_filtrado[variavel_analise].var()
                cv = calcular_cv(media, desvio)
                q1 = df_filtrado[variavel_analise].quantile(0.25)
                q3 = df_filtrado[variavel_analise].quantile(0.75)
                iqr = q3 - q1
                amplitude_total = df_filtrado[variavel_analise].max() - df_filtrado[variavel_analise].min()
                assimetria = df_filtrado[variavel_analise].skew()
                curtose = df_filtrado[variavel_analise].kurtosis()
                
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
                        plot_bgcolor='rgba(30, 41, 59, 0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3b82f6', size=14),
                        showlegend=False,
                        yaxis_title=variavel_analise
                    )
                    fig_ic.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    fig_ic.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    st.plotly_chart(fig_ic, use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"<h4>{t['normality_test']}</h4>", unsafe_allow_html=True)
                
                dados_teste = df_filtrado[variavel_analise].dropna()
                n_teste = len(dados_teste)
                
                if n_teste < 3:
                    st.error("❌ " + ("Amostra muito pequena (n < 3)" if st.session_state.idioma == 'pt' else "Sample too small (n < 3)"))
                elif n_teste > 5000:
                    st.info("ℹ️ " + ("Amostra grande demais. Usando D'Agostino-Pearson." if st.session_state.idioma == 'pt' else "Sample too large. Using D'Agostino-Pearson."))
                    try:
                        _, p = stats.normaltest(dados_teste)
                        if p > 0.05:
                            status = f"✅ {t['normality_test'].split('🧪')[1] if '🧪' in t['normality_test'] else 'Dados normais'}"
                        else:
                            status = f"⚠️ {t['normality_test'].split('🧪')[1] if '🧪' in t['normality_test'] else 'Dados não normais'}"
                        st.info(status + f" (p={p:.4f})")
                    except:
                        st.warning("⚠️ " + ("Teste alternativo não disponível" if st.session_state.idioma == 'pt' else "Alternative test not available"))
                else:
                    try:
                        shapiro = stats.shapiro(dados_teste)
                        if shapiro.pvalue > 0.05:
                            status = f"✅ {t['normality_test'].split('🧪')[1] if '🧪' in t['normality_test'] else 'Dados normais'}"
                        else:
                            status = f"⚠️ {t['normality_test'].split('🧪')[1] if '🧪' in t['normality_test'] else 'Dados não normais'}"
                        st.info(status + f" (p={shapiro.pvalue:.4f})")
                    except:
                        st.error("❌ " + ("Erro no teste" if st.session_state.idioma == 'pt' else "Test error"))
                
                st.markdown("---")
                st.markdown(f"<h4>{t['summary_by_group']}</h4>", unsafe_allow_html=True)
                
                resumo = []
                for nome in atletas_selecionados:
                    for posicao in posicoes_selecionadas:
                        for periodo in periodos_selecionados:
                            dados = df_filtrado[
                                (df_filtrado['Nome'] == nome) & 
                                (df_filtrado['Posição'] == posicao) &
                                (df_filtrado['Período'] == periodo)
                            ]
                            if len(dados) > 0:
                                media_grupo = dados[variavel_analise].mean()
                                desvio_grupo = dados[variavel_analise].std()
                                cv_grupo = calcular_cv(media_grupo, desvio_grupo)
                                valor_max_grupo = dados[variavel_analise].max()
                                valor_min_grupo = dados[variavel_analise].min()
                                minuto_max_grupo = extrair_minuto_do_extremo(dados, variavel_analise, 'Minuto', 'max')
                                minuto_min_grupo = extrair_minuto_do_extremo(dados, variavel_analise, 'Minuto', 'min')
                                
                                resumo.append({
                                    'Atleta': nome,
                                    'Posição': posicao,
                                    'Período': periodo,
                                    f'Máx {variavel_analise}': valor_max_grupo,
                                    'Minuto do Máx': minuto_max_grupo,
                                    f'Mín {variavel_analise}': valor_min_grupo,
                                    'Minuto do Mín': minuto_min_grupo,
                                    'Amplitude': valor_max_grupo - valor_min_grupo,
                                    'Média': media_grupo,
                                    'CV (%)': cv_grupo,
                                    'Nº Amostras': len(dados)
                                })
                
                if resumo:
                    df_resumo = pd.DataFrame(resumo)
                    st.dataframe(
                        df_resumo.style.format({
                            f'Máx {variavel_analise}': '{:.2f}',
                            f'Mín {variavel_analise}': '{:.2f}',
                            'Amplitude': '{:.2f}',
                            'Média': '{:.2f}',
                            'CV (%)': '{:.1f}',
                            'Nº Amostras': '{:.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Explicação do IQR ao final da tabela
                    st.caption(f"📌 {t['iqr_title']}: {t['iqr_explanation']}")
            
            # ABA 3: BOXPLOTS
            with tabs[2]:
                st.markdown(f"<h3>{t['tab_boxplots']}</h3>", unsafe_allow_html=True)
                
                st.markdown(f"<h4>📍 {t['position']}</h4>", unsafe_allow_html=True)
                
                fig_box_pos = go.Figure()
                for posicao in posicoes_selecionadas:
                    dados_pos = df_filtrado[df_filtrado['Posição'] == posicao][variavel_analise]
                    if len(dados_pos) > 0:
                        fig_box_pos.add_trace(go.Box(
                            y=dados_pos,
                            name=posicao,
                            boxmean='sd',
                            marker_color='#3b82f6',
                            line_color='white',
                            fillcolor='rgba(59, 130, 246, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_pos.update_layout(
                    title=f"{t['position']} - {variavel_analise}",
                    plot_bgcolor='rgba(30, 41, 59, 0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3b82f6', size=16),
                    yaxis_title=variavel_analise,
                    showlegend=False
                )
                fig_box_pos.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                fig_box_pos.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                st.markdown(f"<h4>👥 {t['athlete']}</h4>", unsafe_allow_html=True)
                
                # Mostrar TODOS os atletas (sem limitação)
                fig_box_atl = go.Figure()
                for atleta in atletas_selecionados:
                    dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                    if len(dados_atl) > 0:
                        fig_box_atl.add_trace(go.Box(
                            y=dados_atl,
                            name=atleta[:20] + "..." if len(atleta) > 20 else atleta,
                            boxmean='sd',
                            marker_color='#8b5cf6',
                            line_color='white',
                            fillcolor='rgba(139, 92, 246, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                altura_boxplot = max(400, len(atletas_selecionados) * 25)
                
                fig_box_atl.update_layout(
                    title=f"{t['athlete']} - {variavel_analise}",
                    plot_bgcolor='rgba(30, 41, 59, 0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3b82f6', size=16),
                    yaxis_title=variavel_analise,
                    showlegend=False,
                    height=altura_boxplot
                )
                fig_box_atl.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'), tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                st.plotly_chart(fig_box_atl, use_container_width=True)
                
                with st.expander(f"📊 {t['descriptive_stats']} {t['athlete'].lower()}"):
                    stats_atletas = []
                    for atleta in atletas_selecionados:
                        dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                        if len(dados_atl) > 0:
                            q1_atl = dados_atl.quantile(0.25)
                            q3_atl = dados_atl.quantile(0.75)
                            iqr_atl = q3_atl - q1_atl
                            media_atl = dados_atl.mean()
                            desvio_atl = dados_atl.std()
                            cv_atl = calcular_cv(media_atl, desvio_atl)
                            valor_max_atl = dados_atl.max()
                            valor_min_atl = dados_atl.min()
                            minuto_max_atl = extrair_minuto_do_extremo(dados_atl, variavel_analise, 'Minuto', 'max')
                            minuto_min_atl = extrair_minuto_do_extremo(dados_atl, variavel_analise, 'Minuto', 'min')
                            
                            stats_atletas.append({
                                'Atleta': atleta,
                                'Média': media_atl,
                                'Mediana': dados_atl.median(),
                                'DP': desvio_atl,
                                'CV (%)': cv_atl,
                                'Mín': valor_min_atl,
                                'Minuto Mín': minuto_min_atl,
                                'Q1': q1_atl,
                                'Q3': q3_atl,
                                'Máx': valor_max_atl,
                                'Minuto Máx': minuto_max_atl,
                                'IQR': iqr_atl,
                                'Outliers': len(dados_atl[(dados_atl < q1_atl - 1.5*iqr_atl) | (dados_atl > q3_atl + 1.5*iqr_atl)]),
                                'N': len(dados_atl)
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
                    
                    # Explicação do IQR ao final
                    st.caption(f"📌 {t['iqr_title']}: {t['iqr_explanation']}")
            
            # ABA 4: CORRELAÇÕES
            with tabs[3]:
                st.markdown(f"<h3>{t['tab_correlation']}</h3>", unsafe_allow_html=True)
                
                if len(st.session_state.variaveis_quantitativas) > 1:
                    vars_corr = st.multiselect(
                        t['tab_correlation'].replace('🔥', '').strip(),
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
                            plot_bgcolor='rgba(30, 41, 59, 0.8)',
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
                                plot_bgcolor='rgba(30, 41, 59, 0.8)',
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
                        st.info("ℹ️ " + ("Selecione pelo menos 2 variáveis" if st.session_state.idioma == 'pt' else "Select at least 2 variables"))
                else:
                    st.info("ℹ️ " + ("São necessárias pelo menos 2 variáveis" if st.session_state.idioma == 'pt' else "At least 2 variables are needed"))
            
            # ABA 5: COMPARAÇÕES
            with tabs[4]:
                st.markdown(f"<h3>{t['tab_comparison']}</h3>", unsafe_allow_html=True)
                
                if len(posicoes_selecionadas) >= 2:
                    st.markdown(f"<h4>{t['position']}</h4>", unsafe_allow_html=True)
                    
                    col_comp1, col_comp2 = st.columns(2)
                    
                    with col_comp1:
                        grupo1 = st.selectbox(
                            f"{t['position']} 1:", 
                            posicoes_selecionadas, 
                            index=0, 
                            key="grupo1_select",
                            on_change=atualizar_grupo1
                        )
                    with col_comp2:
                        grupo2 = st.selectbox(
                            f"{t['position']} 2:", 
                            posicoes_selecionadas, 
                            index=min(1, len(posicoes_selecionadas)-1), 
                            key="grupo2_select",
                            on_change=atualizar_grupo2
                        )
                    
                    # Usar valores do session state
                    g1_atual = st.session_state.grupo1 if st.session_state.grupo1 is not None else grupo1
                    g2_atual = st.session_state.grupo2 if st.session_state.grupo2 is not None else grupo2
                    
                    if g1_atual and g2_atual and g1_atual != g2_atual:
                        resultado = comparar_grupos(df_filtrado, variavel_analise, g1_atual, g2_atual)
                        
                        if resultado:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>📊 {t['tab_comparison']}</h4>
                                <hr style="border-color: #334155;">
                                <p><strong>{t['position']} 1 ({g1_atual}):</strong> {resultado['media_g1']:.2f} ± {resultado['std_g1']:.2f} (n={resultado['n_g1']})</p>
                                <p><strong>{t['position']} 2 ({g2_atual}):</strong> {resultado['media_g2']:.2f} ± {resultado['std_g2']:.2f} (n={resultado['n_g2']})</p>
                                <p><strong>Teste:</strong> {resultado['teste']}</p>
                                <p><strong>p-valor:</strong> {resultado['p_valor']:.4f}</p>
                                <p><strong>Diferença:</strong> {resultado['media_g1'] - resultado['media_g2']:.2f}</p>
                                <p><strong>{'✅ Significativo' if resultado['significativo'] else '❌ Não significativo'}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig_comp = go.Figure()
                            fig_comp.add_trace(go.Box(
                                y=df_filtrado[df_filtrado['Posição'] == g1_atual][variavel_analise],
                                name=g1_atual,
                                boxmean='sd',
                                marker_color='#3b82f6',
                                line_color='white'
                            ))
                            fig_comp.add_trace(go.Box(
                                y=df_filtrado[df_filtrado['Posição'] == g2_atual][variavel_analise],
                                name=g2_atual,
                                boxmean='sd',
                                marker_color='#ef4444',
                                line_color='white'
                            ))
                            fig_comp.update_layout(
                                title=f"{g1_atual} vs {g2_atual} - {variavel_analise}",
                                plot_bgcolor='rgba(30, 41, 59, 0.8)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=11),
                                title_font=dict(color='#3b82f6', size=16),
                                yaxis_title=variavel_analise
                            )
                            fig_comp.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                            fig_comp.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                            st.plotly_chart(fig_comp, use_container_width=True)
                        else:
                            st.warning("⚠️ " + ("Dados insuficientes para comparação" if st.session_state.idioma == 'pt' else "Insufficient data for comparison"))
                    else:
                        st.info("ℹ️ " + ("Selecione grupos diferentes" if st.session_state.idioma == 'pt' else "Select different groups"))
                else:
                    st.info("ℹ️ " + ("Selecione pelo menos 2 posições" if st.session_state.idioma == 'pt' else "Select at least 2 positions"))
            
            with st.expander("📋 " + ("Visualizar dados brutos filtrados" if st.session_state.idioma == 'pt' else "View filtered raw data")):
                st.dataframe(df_filtrado, use_container_width=True)
    
    # Reset do botão após processamento
    st.session_state.processar_click = False

else:
    t = translations[st.session_state.idioma]
    if st.session_state.df_completo is None:
        st.info(t['step1'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(t['file_format'])
            
            exemplo_data = {
                'Nome-Período-Minuto': [
                    'Mariano-1 TEMPO 00:00-01:00',
                    'Maria-SEGUNDO TEMPO 05:00-06:00',
                    'Joao-2 TEMPO 44:00-45:00',
                    'Marta-PRIMEIRO TEMPO 11:00-12:00',
                    'Pedro-1 TEMPO 15:00-16:00',
                    'Ana-SEGUNDO TEMPO 22:00-23:00'
                ],
                'Posição': ['Atacante', 'Meio-campo', 'Zagueiro', 'Atacante', 'Goleiro', 'Meio-campo'],
                'Distancia Total': [250, 127, 200, 90, 45, 180],
                'Velocidade Maxima': [23, 29, 33, 27, 15, 31],
                'Aceleracao Max': [3.6, 4.2, 4.9, 3.1, 2.8, 4.5]
            }
            
            df_exemplo = pd.DataFrame(exemplo_data)
            st.dataframe(df_exemplo, use_container_width=True, hide_index=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h4>{t['components']}</h4>
                <hr style="border-color: #334155;">
                <p>{t['name_ex']}</p>
                <p>{t['period_ex']}</p>
                <p>{t['minute_ex']}</p>
                <p>{t['position_ex']}</p>
            </div>
            
            <div class="metric-container" style="margin-top: 20px;">
                <h4>{t['tip']}</h4>
                <hr style="border-color: #334155;">
                <p>{t['tip_text']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander(t['multi_file_ex']):
            st.markdown(t['multi_file_text'])
    else:
        st.info(t['step2'])
        
        with st.expander("📋 " + ("Preview dos dados carregados" if st.session_state.idioma == 'pt' else "Preview of loaded data")):
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