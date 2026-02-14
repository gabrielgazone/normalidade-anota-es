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
# CSS PERSONALIZADO - DESIGN CIENTÍFICO QUÂNTICO
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
    
    /* Background */
    .stApp {
        background: radial-gradient(ellipse at 50% 50%, #0a0f2a 0%, #000000 100%);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 2px solid #3b82f6;
    }
    
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
    }
    
    /* Cards */
    .quantum-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 20px;
        padding: 25px;
        border: 2px solid #3b82f6;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        margin-bottom: 15px;
    }
    
    .quantum-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(59,130,246,0.3);
    }
    
    .quantum-card .label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .quantum-card .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 10px 0;
    }
    
    .quantum-card .icon {
        font-size: 3rem;
    }
    
    /* Warning card */
    .quantum-warning {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        border-radius: 20px;
        padding: 20px;
        border: 2px solid #ef4444;
        text-align: center;
        color: white;
    }
    
    .quantum-warning .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Zone cards */
    .zone-card {
        background: #1e293b;
        border-radius: 15px;
        padding: 15px;
        margin: 5px 0;
        border-left: 6px solid;
        transition: all 0.3s ease;
    }
    
    .zone-card:hover {
        transform: translateX(10px);
    }
    
    .zone-card .zone-name {
        color: #94a3b8;
        font-size: 1rem;
        text-transform: uppercase;
    }
    
    .zone-card .zone-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        color: white;
    }
    
    .zone-card .zone-count {
        color: #3b82f6;
        font-size: 1rem;
    }
    
    /* Wave card */
    .wave-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 15px;
        padding: 15px;
        border-left: 6px solid;
        margin: 10px 0;
    }
    
    .wave-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
    }
    
    .wave-card .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        color: white;
    }
    
    .wave-card .sub-value {
        color: #64748b;
        font-size: 0.85rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #1e293b;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        border: 2px solid #3b82f6;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(59,130,246,0.5);
    }
    
    /* Metric container */
    .metric-container {
        background: #1e293b;
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #3b82f6;
        margin: 10px 0;
    }
    
    .metric-container h4 {
        color: #3b82f6;
        margin-bottom: 10px;
    }
    
    .metric-container p {
        color: #e2e8f0;
        margin: 5px 0;
    }
    
    /* Note card */
    .note-card {
        background: #1e293b;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
    }
    
    .note-card .note-date {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    .note-card .note-text {
        color: white;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 5px;
    }
    
    /* Dataframe */
    .dataframe {
        background: #1e293b !important;
        border-radius: 10px !important;
        border: 2px solid #3b82f6 !important;
        color: white !important;
    }
    
    .dataframe th {
        background: #0f172a !important;
        color: #3b82f6 !important;
        padding: 12px !important;
    }
    
    .dataframe td {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        padding: 10px !important;
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
    <div style="text-align: center; padding: 30px 0;">
        <h1>⚛️ SPORTS SCIENCE PRO</h1>
        <p style="color: #94a3b8; font-size: 1.4rem;">QUANTUM ANALYTICS PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# INTERNACIONALIZAÇÃO
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
        'process': '🚀 PROCESSAR ANÁLISE',
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
        'trend_analysis': 'Análise de Tendência'
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
        'process': '🚀 PROCESS ANALYSIS',
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
        'trend_analysis': 'Trend Analysis'
    }
}

# ============================================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================================

def parse_identificacao(series):
    """Parseia coluna de identificação: Nome-Período-Minuto"""
    nomes, periodos, minutos = [], [], []
    
    for item in series:
        try:
            texto = str(item).strip()
            partes = texto.split('-')
            
            if len(partes) >= 3:
                nomes.append(partes[0].strip())
                periodos.append('-'.join(partes[1:-1]).strip())
                minutos.append(partes[-1].strip())
            else:
                nomes.append(texto)
                periodos.append('')
                minutos.append('')
        except:
            nomes.append('')
            periodos.append('')
            minutos.append('')
    
    return nomes, periodos, minutos

def verificar_estruturas_arquivos(dataframes):
    """Verifica estrutura dos dataframes"""
    if not dataframes:
        return False, []
    
    primeira = dataframes[0].columns.tolist()
    
    for df in dataframes[1:]:
        if df.columns.tolist() != primeira:
            return False, primeira
    
    return True, primeira

def processar_upload(files):
    """Processa arquivos CSV"""
    if not files:
        return None, [], [], [], []
    
    dataframes = []
    nomes_arquivos = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.shape[1] >= 3:
                dataframes.append(df)
                nomes_arquivos.append(f.name)
        except:
            continue
    
    if not dataframes:
        return None, [], [], [], []
    
    # Verificar estrutura
    estrutura_ok, _ = verificar_estruturas_arquivos(dataframes)
    if not estrutura_ok:
        return None, [], [], [], []
    
    # Concatenar
    df_final = pd.concat(dataframes, ignore_index=True)
    
    # Parsear identificação
    primeira_coluna = df_final.iloc[:, 0].astype(str)
    nomes, periodos, minutos = parse_identificacao(primeira_coluna)
    
    # Posições
    posicoes = df_final.iloc[:, 1].astype(str)
    
    # Variáveis numéricas
    variaveis = []
    dados_var = {}
    
    for i in range(2, df_final.shape[1]):
        nome = df_final.columns[i]
        valores = pd.to_numeric(df_final.iloc[:, i], errors='coerce')
        if not valores.dropna().empty:
            variaveis.append(nome)
            dados_var[nome] = valores
    
    if not variaveis:
        return None, [], [], [], []
    
    # Dataframe estruturado
    df_estruturado = pd.DataFrame({
        'Nome': nomes,
        'Posição': posicoes,
        'Período': periodos,
        'Minuto': minutos
    })
    
    for var, vals in dados_var.items():
        df_estruturado[var] = vals
    
    df_estruturado = df_estruturado[df_estruturado['Nome'].str.len() > 0].reset_index(drop=True)
    
    periodos_unicos = sorted([p for p in df_estruturado['Período'].unique() if p])
    posicoes_unicas = sorted([p for p in df_estruturado['Posição'].unique() if p])
    
    return df_estruturado, variaveis, periodos_unicos, posicoes_unicas, nomes_arquivos

def media_movel(series, window):
    """Calcula média móvel"""
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()

def extrair_minuto_extremo(df, col_valor, col_minuto, extremo='max'):
    """Extrai minuto do valor extremo"""
    try:
        if df.empty:
            return 'N/A'
        if extremo == 'max':
            idx = df[col_valor].idxmax()
        else:
            idx = df[col_valor].idxmin()
        return str(df.loc[idx, col_minuto]) if pd.notna(idx) else 'N/A'
    except:
        return 'N/A'

def calcular_cv(media, desvio):
    """Coeficiente de variação"""
    return (desvio / abs(media)) * 100 if media != 0 else 0

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

def comparar_grupos(df, variavel, g1, g2):
    """Compara dois grupos"""
    try:
        d1 = df[df['Posição'] == g1][variavel].dropna()
        d2 = df[df['Posição'] == g2][variavel].dropna()
        
        if len(d1) < 3 or len(d2) < 3:
            return None
        
        if len(d1) <= 5000 and len(d2) <= 5000:
            _, p1 = stats.shapiro(d1)
            _, p2 = stats.shapiro(d2)
            
            if p1 > 0.05 and p2 > 0.05:
                stat, p = stats.ttest_ind(d1, d2)
                teste = "Teste t"
            else:
                stat, p = stats.mannwhitneyu(d1, d2)
                teste = "Mann-Whitney"
        else:
            stat, p = stats.mannwhitneyu(d1, d2)
            teste = "Mann-Whitney"
        
        return {
            'teste': teste,
            'p_valor': p,
            'significativo': p < 0.05,
            'media_g1': d1.mean(),
            'media_g2': d2.mean(),
            'std_g1': d1.std(),
            'std_g2': d2.std(),
            'n_g1': len(d1),
            'n_g2': len(d2)
        }
    except:
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
    <div style="background: #1e293b; border-radius: 12px; padding: 20px; border-left: 5px solid {cor};">
        <h4 style="color: white;">{status}</h4>
        <p style="color: #94a3b8;"><strong>Teste:</strong> {nome_teste}</p>
        <p style="color: #94a3b8;"><strong>p-valor:</strong> <span style="color: {cor};">{p_text}</span></p>
    </div>
    """, unsafe_allow_html=True)

def executive_card(titulo, valor, delta, icone, cor_status="#3b82f6"):
    """Card executivo"""
    delta_icon = "▲" if delta > 0 else "▼"
    delta_color = "#10b981" if delta > 0 else "#ef4444"
    
    st.markdown(f"""
    <div class="quantum-card" style="border-left-color: {cor_status};">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <p class="label">{titulo}</p>
                <p class="value">{valor}</p>
                <p style="color: {delta_color};">{delta_icon} {abs(delta):.1f}%</p>
            </div>
            <div class="icon">{icone}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def time_metric_card(label, valor, sub_label="", cor="#3b82f6"):
    """Card para métricas temporais"""
    st.markdown(f"""
    <div class="wave-card" style="border-left-color: {cor};">
        <p class="label">{label}</p>
        <p class="value">{valor}</p>
        <p class="sub-value">{sub_label}</p>
    </div>
    """, unsafe_allow_html=True)

def warning_card(titulo, valor, subtitulo, icone="⚠️"):
    """Card de aviso"""
    st.markdown(f"""
    <div class="quantum-warning">
        <p class="label">{icone} {titulo}</p>
        <p class="value">{valor}</p>
        <p>{subtitulo}</p>
    </div>
    """, unsafe_allow_html=True)

def comparar_atletas(df, atleta1, atleta2, variaveis, t):
    """Compara dois atletas"""
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
                <div style="background: #1e293b; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {cor};">
                    <span style="color: #94a3b8;">{var}:</span>
                    <span style="color: white; float: right;">{dados1[var]:.2f}</span>
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
                <div style="background: #1e293b; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {cor};">
                    <span style="color: #94a3b8;">{var}:</span>
                    <span style="color: white; float: right;">{dados2[var]:.2f}</span>
                    <br>
                    <span style="color: {cor};">{delta:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
    except:
        st.error("Erro na comparação")

def sistema_anotacoes(t):
    """Sistema de anotações"""
    with st.expander("📝 Anotações"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            nova = st.text_area("Nova anotação:", height=100, key="nova_anotacao")
        
        with col2:
            if st.button("➕ Adicionar"):
                if nova:
                    st.session_state.anotacoes.append({
                        'data': datetime.now().strftime("%d/%m/%Y %H:%M"),
                        'texto': nova
                    })
                    st.rerun()
        
        for anot in reversed(st.session_state.anotacoes):
            st.markdown(f"""
            <div class="note-card">
                <p class="note-date">{anot['data']}</p>
                <p class="note-text">{anot['texto']}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# FUNÇÃO PRINCIPAL DE TIMELINE
# ============================================================================

def criar_timeline_quantum(df_completo, atletas, periodos, variavel, window, t):
    """Cria timeline com todas as funcionalidades"""
    try:
        if not atletas or not periodos:
            return None, [], []
        
        df_filtrado = df_completo[
            df_completo['Nome'].isin(atletas) &
            df_completo['Período'].isin(periodos)
        ].copy()
        
        if df_filtrado.empty:
            return None, [], []
        
        df_filtrado = df_filtrado.sort_values('Minuto')
        
        combinacoes = df_filtrado.groupby(['Nome', 'Período']).size().reset_index()[['Nome', 'Período']]
        combinacoes_list = list(zip(combinacoes['Nome'], combinacoes['Período']))
        
        fig = go.Figure()
        cores = px.colors.qualitative.Set2 + px.colors.qualitative.Set1
        
        valor_max = df_filtrado[variavel].max()
        limiar_80 = valor_max * 0.8 if valor_max > 0 else 0
        
        # Linha do limiar
        fig.add_hline(
            y=limiar_80,
            line_dash="solid",
            line_color="#ef4444",
            line_width=3,
            annotation_text=f"🔴 {t['threshold_80']}: {limiar_80:.2f}",
            annotation_position="top left"
        )
        
        # Média móvel
        if st.session_state.show_moving_average:
            df_filtrado['Media_Movel'] = media_movel(df_filtrado[variavel], window)
            fig.add_trace(go.Scatter(
                x=df_filtrado['Minuto'],
                y=df_filtrado['Media_Movel'],
                mode='lines',
                name=f"📈 {t['moving_average']} ({window})",
                line=dict(color='white', width=4, dash='dash'),
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
            
            # Linha principal
            fig.add_trace(go.Scatter(
                x=df_combo['Minuto'],
                y=df_combo[variavel],
                mode='lines',
                name=f"{atleta[:15]} - {periodo[:10]}",
                line=dict(color=cor, width=2.5)
            ))
            
            # Pontos normais
            if not df_normal.empty:
                fig.add_trace(go.Scatter(
                    x=df_normal['Minuto'],
                    y=df_normal[variavel],
                    mode='markers',
                    marker=dict(size=8, color=cor, opacity=0.6, line=dict(color='white', width=1)),
                    showlegend=False
                ))
            
            # Pontos críticos
            if not df_critico.empty:
                fig.add_trace(go.Scatter(
                    x=df_critico['Minuto'],
                    y=df_critico[variavel],
                    mode='markers',
                    marker=dict(size=14, color='#ef4444', symbol='circle', line=dict(color='white', width=2)),
                    showlegend=False
                ))
        
        # Marcadores no eixo X
        if minutos_criticos and st.session_state.show_critical_markers:
            y_min = df_filtrado[variavel].min()
            y_range = df_filtrado[variavel].max() - y_min
            y_pos = y_min - (y_range * 0.1)
            
            fig.add_trace(go.Scatter(
                x=minutos_criticos,
                y=[y_pos] * len(minutos_criticos),
                mode='markers',
                name=t['critical_markers'],
                marker=dict(size=12, color='#ef4444', symbol='triangle-down', line=dict(color='white', width=1))
            ))
            
            for m in set(minutos_criticos):
                fig.add_vline(x=m, line_width=1, line_dash="dot", line_color="#ef4444", opacity=0.3)
        
        # Estatísticas
        media_global = df_filtrado[variavel].mean()
        desvio_global = df_filtrado[variavel].std()
        
        fig.add_hline(
            y=media_global,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"📊 {t['mean']}: {media_global:.2f}",
            annotation_position="bottom left"
        )
        
        fig.add_hrect(
            y0=media_global - desvio_global,
            y1=media_global + desvio_global,
            fillcolor="#3b82f6",
            opacity=0.1,
            line_width=0
        )
        
        fig.update_layout(
            title=f"📈 {t['trend_analysis']} - {variavel}",
            xaxis_title="Minuto",
            yaxis_title=variavel,
            hovermode='x unified',
            plot_bgcolor='rgba(15,23,42,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(15,23,42,0.9)',
                bordercolor='#3b82f6',
                borderwidth=2
            )
        )
        
        fig.update_xaxes(gridcolor='#334155', tickangle=-45)
        fig.update_yaxes(gridcolor='#334155')
        
        return fig, combinacoes_list, minutos_criticos
        
    except Exception as e:
        st.error(f"Erro na timeline: {str(e)}")
        return None, [], []

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<div class='sidebar-title'>🌐 IDIOMA</div>", unsafe_allow_html=True)
    
    idiomas = ['pt', 'en']
    idx = idiomas.index(st.session_state.idioma) if st.session_state.idioma in idiomas else 0
    idioma = st.selectbox("", idiomas, index=idx, label_visibility="collapsed", key="idioma_selector")
    
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
        key="file_uploader"
    )
    
    if uploaded_files and len(uploaded_files) > 0 and not st.session_state.upload_concluido:
        with st.spinner('Processando...'):
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
    
    if st.session_state.df_completo is not None:
        st.markdown("---")
        
        if st.session_state.variaveis_quantitativas:
            st.markdown(f"<div class='sidebar-title'>📈 {t['variable']}</div>", unsafe_allow_html=True)
            
            idx_var = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                idx_var = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            var_sel = st.selectbox(
                "",
                st.session_state.variaveis_quantitativas,
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
            
            select_all_pos = st.checkbox("Todos", value=len(st.session_state.posicoes_selecionadas) == len(st.session_state.todos_posicoes))
            
            if select_all_pos:
                if st.session_state.posicoes_selecionadas != st.session_state.todos_posicoes:
                    st.session_state.posicoes_selecionadas = st.session_state.todos_posicoes.copy()
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                pos_sel = st.multiselect(
                    "",
                    st.session_state.todos_posicoes,
                    default=st.session_state.posicoes_selecionadas,
                    label_visibility="collapsed"
                )
                if pos_sel != st.session_state.posicoes_selecionadas:
                    st.session_state.posicoes_selecionadas = pos_sel
                    st.session_state.dados_processados = False
                    st.rerun()
        
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown(f"<div class='sidebar-title'>📅 {t['period']}</div>", unsafe_allow_html=True)
            
            select_all_per = st.checkbox("Todos", value=len(st.session_state.periodos_selecionados) == len(st.session_state.todos_periodos))
            
            if select_all_per:
                if st.session_state.periodos_selecionados != st.session_state.todos_periodos:
                    st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                per_sel = st.multiselect(
                    "",
                    st.session_state.todos_periodos,
                    default=st.session_state.periodos_selecionados,
                    label_visibility="collapsed"
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
        
        atletas_disp = sorted(df_temp['Nome'].unique())
        
        if not st.session_state.atletas_selecionados and atletas_disp:
            st.session_state.atletas_selecionados = atletas_disp.copy()
            st.rerun()
        
        select_all_atl = st.checkbox("Todos", value=len(st.session_state.atletas_selecionados) == len(atletas_disp) and len(atletas_disp) > 0)
        
        if select_all_atl:
            if st.session_state.atletas_selecionados != atletas_disp:
                st.session_state.atletas_selecionados = atletas_disp
                st.session_state.dados_processados = False
                st.rerun()
        else:
            atl_sel = st.multiselect(
                "",
                atletas_disp,
                default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else (atletas_disp[:1] if atletas_disp else []),
                label_visibility="collapsed"
            )
            if atl_sel != st.session_state.atletas_selecionados:
                st.session_state.atletas_selecionados = atl_sel
                st.session_state.dados_processados = False
                st.rerun()
        
        st.markdown("---")
        st.markdown(f"<div class='sidebar-title'>⚙️ {t['config']}</div>", unsafe_allow_html=True)
        
        st.session_state.n_classes = st.slider("Classes:", 3, 20, st.session_state.n_classes)
        st.session_state.window_size = st.slider(f"{t['window_size']}:", 2, 10, st.session_state.window_size)
        
        st.session_state.show_moving_average = st.checkbox(t['moving_average'], st.session_state.show_moving_average)
        st.session_state.show_critical_markers = st.checkbox(t['critical_markers'], st.session_state.show_critical_markers)
        
        st.markdown("---")
        
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button(t['process'], use_container_width=True, disabled=not pode_processar):
            st.session_state.processar_click = True
            st.rerun()

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.processar_click and st.session_state.df_completo is not None:
    
    with st.spinner('Gerando análise...'):
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
            
            # Dashboard Executivo
            st.markdown(f"<h2>📊 {t['tab_executive']}</h2>", unsafe_allow_html=True)
            
            media_global = df_filtrado[variavel].mean()
            valor_max = df_filtrado[variavel].max()
            valor_min = df_filtrado[variavel].min()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="quantum-card">
                    <p class="label">{t['mean']}</p>
                    <p class="value">{media_global:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="quantum-card">
                    <p class="label">{t['max_value']}</p>
                    <p class="value">{valor_max:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="quantum-card">
                    <p class="label">{t['min_value']}</p>
                    <p class="value">{valor_min:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="quantum-card">
                    <p class="label">{t['observations']}</p>
                    <p class="value">{len(df_filtrado)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Abas
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
                        marker_color='#3b82f6',
                        opacity=0.8
                    ))
                    fig_hist.add_vline(x=media_global, line_dash="dash", line_color="#ef4444")
                    fig_hist.update_layout(
                        title=f"Histograma - {variavel}",
                        plot_bgcolor='rgba(15,23,42,0.95)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    fig_hist.update_xaxes(gridcolor='#334155')
                    fig_hist.update_yaxes(gridcolor='#334155')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    dados = df_filtrado[variavel].dropna()
                    quantis_teoricos = stats.norm.ppf(np.linspace(0.01, 0.99, len(dados)))
                    quantis_observados = np.sort(dados)
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=quantis_observados,
                        mode='markers',
                        marker=dict(color='#3b82f6', size=6)
                    ))
                    fig_qq.update_layout(
                        title=f"QQ Plot - {variavel}",
                        plot_bgcolor='rgba(15,23,42,0.95)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    fig_qq.update_xaxes(gridcolor='#334155')
                    fig_qq.update_yaxes(gridcolor='#334155')
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
                    time_metric_card(t['mean'], f"{media_global:.2f}", "", "#3b82f6")
                with col4:
                    time_metric_card(t['threshold_80'], f"{limiar_80:.2f}", "", "#f59e0b")
                with col5:
                    warning_card(t['critical_events'], f"{eventos_criticos}", f"{percentual_critico:.1f}% {t['above_threshold']}")
                
                st.markdown("---")
                
                # Zonas de intensidade
                st.markdown(f"<h4>{t['intensity_zones']}</h4>", unsafe_allow_html=True)
                
                metodo = st.radio(
                    t['zone_method'],
                    [t['percentiles'], t['based_on_max']],
                    horizontal=True,
                    key="metodo_zona_radio"
                )
                
                st.session_state.metodo_zona = 'percentis' if metodo == t['percentiles'] else 'based_on_max'
                
                zonas = criar_zonas_intensidade(df_filtrado, variavel, st.session_state.metodo_zona)
                cores_zonas = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981']
                
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
                            <p class="zone-name">{zona}</p>
                            <p class="zone-value">{limite:.1f}</p>
                            <p class="zone-count">{count} ({pct:.0f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Timeline
                st.markdown(f"<h4>⏱️ {t['trend_analysis']}</h4>", unsafe_allow_html=True)
                
                resultado = criar_timeline_quantum(df, atletas, periodos, variavel, window, t)
                
                if resultado and resultado[0]:
                    fig_timeline, combinacoes, minutos_criticos = resultado
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"✅ {len(combinacoes)} combinações")
                    with col2:
                        st.warning(f"⚠️ {len(set(minutos_criticos))} minutos críticos")
                    with col3:
                        st.info(f"📊 {len(df_filtrado)} observações")
                
                st.markdown("---")
                
                # Estatísticas descritivas
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
                    <div class="metric-container">
                        <h4>{t['mean']}</h4>
                        <p><strong>{t['mean']}:</strong> {media:.3f}</p>
                        <p><strong>{t['median']}:</strong> {mediana:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_e2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{t['std']}</h4>
                        <p><strong>{t['std']}:</strong> {desvio:.3f}</p>
                        <p><strong>{t['cv']}:</strong> {calcular_cv(media, desvio):.1f}%</p>
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
                    <div class="metric-container">
                        <h4>{t['kurtosis']}</h4>
                        <p><strong>Valor:</strong> {curtose:.3f}</p>
                        <p><strong>Interpretação:</strong> {interp_curt}</p>
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
                        error_y=dict(type='constant', value=(ic_sup - media), color='#ef4444', thickness=3, width=15)
                    ))
                    
                    fig_ic.update_layout(
                        title=t['confidence_interval'],
                        plot_bgcolor='rgba(15,23,42,0.95)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False
                    )
                    fig_ic.update_xaxes(gridcolor='#334155')
                    fig_ic.update_yaxes(gridcolor='#334155')
                    
                    st.plotly_chart(fig_ic, use_container_width=True)
                
                st.markdown("---")
                
                # Teste de normalidade
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
                    st.dataframe(df_resumo, use_container_width=True, hide_index=True)
            
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
                            line_color='white'
                        ))
                
                fig_box_pos.update_layout(
                    plot_bgcolor='rgba(15,23,42,0.95)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                fig_box_pos.update_xaxes(gridcolor='#334155')
                fig_box_pos.update_yaxes(gridcolor='#334155')
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                # Boxplot por atleta
                st.markdown(f"<h4>👤 {t['athlete']}</h4>", unsafe_allow_html=True)
                
                fig_box_atl = go.Figure()
                for atl in atletas[:10]:
                    dados = df_filtrado[df_filtrado['Nome'] == atl][variavel]
                    if len(dados) > 0:
                        fig_box_atl.add_trace(go.Box(
                            y=dados,
                            name=atl[:15] + "..." if len(atl) > 15 else atl,
                            boxmean='sd',
                            marker_color='#8b5cf6',
                            line_color='white'
                        ))
                
                fig_box_atl.update_layout(
                    plot_bgcolor='rgba(15,23,42,0.95)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=max(400, len(atletas[:10]) * 30)
                )
                fig_box_atl.update_xaxes(gridcolor='#334155', tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#334155')
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
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1
                        )
                        fig_corr.update_layout(
                            plot_bgcolor='rgba(15,23,42,0.95)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            height=500
                        )
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
                            <div class="metric-container">
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
        <div class="quantum-card">
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