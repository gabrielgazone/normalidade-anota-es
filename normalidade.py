import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import io
import base64
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Sports Science Analytics Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🏃"
)

# ============================================================================
# INTERNACIONALIZAÇÃO
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
        'tab_executive': '📋 Executivo',
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
        'above_threshold': 'acima do limiar de 80%',
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
        'iqr_explanation': 'O IQR (Intervalo Interquartil) é a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1). Representa a amplitude dos 50% centrais dos dados, sendo uma medida robusta de dispersão menos sensível a outliers.',
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
        'tab_executive': '📋 Executive',
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
        'above_threshold': 'above 80% threshold',
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
        'iqr_explanation': 'IQR (Interquartile Range) is the difference between the third quartile (Q3) and the first quartile (Q1). It represents the range of the middle 50% of the data, being a robust measure of dispersion.',
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
    },
    'es': {
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Dashboard Profesional para Análisis de Rendimiento Deportivo',
        'upload': 'Carga de Datos',
        'variable': 'Variable',
        'position': 'Posición',
        'period': 'Período',
        'athlete': 'Atleta',
        'config': 'Configuración',
        'tab_distribution': '📊 Distribución',
        'tab_temporal': '📈 Estadísticas & Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlaciones',
        'tab_comparison': '⚖️ Comparaciones',
        'tab_executive': '📋 Ejecutivo',
        'positions': 'Posiciones',
        'periods': 'Períodos',
        'athletes': 'Atletas',
        'observations': 'Observaciones',
        'mean': 'Media',
        'median': 'Mediana',
        'mode': 'Moda',
        'std': 'Desviación Estándar',
        'variance': 'Varianza',
        'cv': 'Coeficiente de Variación',
        'min': 'Mínimo',
        'max': 'Máximo',
        'amplitude': 'Amplitud',
        'q1': 'Q1 (25%)',
        'q3': 'Q3 (75%)',
        'iqr': 'IQR',
        'skewness': 'Asimetría',
        'kurtosis': 'Curtosis',
        'max_value': 'VALOR MÁXIMO',
        'min_value': 'VALOR MÍNIMO',
        'minute_of_max': 'Minuto del Máx',
        'minute_of_min': 'Minuto del Mín',
        'threshold_80': 'UMBRAL 80%',
        'critical_events': 'EVENTOS CRÍTICOS',
        'above_threshold': 'por encima del umbral 80%',
        'intensity_zones': '🎚️ Zonas de Intensidad',
        'zone_method': 'Método de definición',
        'percentiles': 'Percentiles',
        'based_on_max': 'Basado en Máximo',
        'very_low': 'Muy Baja',
        'low': 'Baja',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muy Alta',
        'process': '🚀 Procesar Análisis',
        'descriptive_stats': '📊 Estadísticas Descriptivas',
        'confidence_interval': '🎯 Intervalo de Confianza (95%)',
        'normality_test': '🧪 Prueba de Normalidad',
        'summary_by_group': '🏃 Resumen por Atleta, Posición y Período',
        'symmetric': 'Aproximadamente simétrica',
        'moderate_skew': 'Moderadamente asimétrica',
        'high_skew': 'Fuertemente asimétrica',
        'leptokurtic': 'Leptocúrtica (colas pesadas)',
        'platykurtic': 'Platicúrtica (colas ligeras)',
        'mesokurtic': 'Mesocúrtica (normal)',
        'strong_positive': 'Correlación fuerte positiva',
        'moderate_positive': 'Correlación moderada positiva',
        'weak_positive': 'Correlación débil positiva',
        'very_weak_positive': 'Correlación muy débil positiva',
        'very_weak_negative': 'Correlación muy débil negativa',
        'weak_negative': 'Correlación débil negativa',
        'moderate_negative': 'Correlación moderada negativa',
        'strong_negative': 'Correlación fuerte negativa',
        'iqr_title': '📌 ¿Qué es IQR?',
        'iqr_explanation': 'IQR (Rango Intercuartil) es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1). Representa la amplitud del 50% central de los datos, siendo una medida robusta de dispersión.',
        'step1': '👈 **Paso 1:** Cargue uno o más archivos CSV para comenzar',
        'step2': '👈 **Paso 2:** Seleccione los filtros y haga clic en Procesar Análisis',
        'file_format': '### 📋 Formato esperado del archivo:',
        'col1_desc': '**Primera columna:** Identificación en formato `Nombre-Período-Minuto`',
        'col2_desc': '**Segunda columna:** Posición del atleta',
        'col3_desc': '**Demás columnas (3+):** Variables numéricas para análisis',
        'components': '📌 Componentes',
        'name_ex': 'Nombre: Mariano, Maria, Joao...',
        'period_ex': 'Período: 1 TEMPO, SEGUNDO TEMPO...',
        'minute_ex': 'Minuto: 00:00-01:00, 05:00-06:00...',
        'position_ex': 'Posición: Atacante, Meio-campo...',
        'tip': '💡 Consejo',
        'tip_text': 'Puede seleccionar múltiples archivos CSV con la misma estructura.',
        'multi_file_ex': '📁 Ejemplo con múltiples archivos',
        'multi_file_text': '''
            ### Cargando múltiples archivos:
            1. Prepare sus archivos CSV con la **misma estructura** de columnas
            2. Seleccione todos los archivos deseados
            3. El sistema verificará compatibilidad y concatenará automáticamente
        '''
    }
}

# ============================================================================
# CSS PERSONALIZADO - DESIGN CIENTÍFICO INOVADOR
# ============================================================================

st.markdown("""
<style>
    /* Tema base científico com gradiente dinâmico */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #0a0f1e 0%, #1a1f2f 50%, #0f1422 100%);
        position: relative;
        overflow-x: hidden;
    }
    
    /* Efeito de partículas científicas */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #3b82f6, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 40px 70px, #8b5cf6, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 90px 40px, #10b981, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 160px 120px, #ef4444, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 230px 80px, #3b82f6, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 300px 190px, #8b5cf6, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 380px 250px, #10b981, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 450px 320px, #ef4444, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 520px 150px, #3b82f6, rgba(0,0,0,0));
        background-repeat: repeat;
        opacity: 0.15;
        pointer-events: none;
        z-index: 0;
        animation: float 20s infinite linear;
    }
    
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
        100% { transform: translateY(0px) rotate(360deg); }
    }
    
    /* Sidebar elegante com efeito glassmorphism */
    .css-1d391kg, .css-1wrcr25 {
        background: rgba(2, 6, 23, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 10px 0 30px -10px rgba(0,0,0,0.5);
    }
    
    .sidebar-title {
        color: #f8fafc !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid transparent;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) border-box;
        -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        text-transform: uppercase;
        letter-spacing: 2px;
        animation: glow 3s infinite;
    }
    
    @keyframes glow {
        0% { border-bottom-color: #3b82f6; }
        50% { border-bottom-color: #8b5cf6; }
        100% { border-bottom-color: #3b82f6; }
    }
    
    /* Cards executivos com design científico */
    .executive-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 25px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5), 0 0 0 1px rgba(59, 130, 246, 0.1) inset;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
    }
    
    .executive-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.6s;
        pointer-events: none;
    }
    
    .executive-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 30px 50px -20px #3b82f6, 0 0 0 2px rgba(59, 130, 246, 0.3) inset;
    }
    
    .executive-card:hover::before {
        opacity: 1;
        animation: rotate 10s infinite linear;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .executive-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .executive-card .value {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
        background: linear-gradient(135deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    
    /* Timeline cards com gradiente dinâmico */
    .time-metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 16px;
        border-left: 4px solid transparent;
        margin: 10px 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    /* Warning card com efeito de alerta */
    .warning-card {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.9), rgba(185, 28, 28, 0.95));
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 0 30px rgba(220, 38, 38, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        text-align: center;
        color: white;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: alertPulse 2s infinite;
        position: relative;
    }
    
    @keyframes alertPulse {
        0% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.3); }
        50% { box-shadow: 0 0 50px rgba(220, 38, 38, 0.6); }
        100% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.3); }
    }
    
    /* Zone cards com design moderno */
    .zone-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 16px;
        margin: 5px 0;
        border-left: 4px solid;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .zone-card:hover {
        transform: translateX(5px) scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    
    .zone-card .zone-name {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .zone-card .zone-value {
        font-size: 1.4rem;
        color: white;
        font-weight: 600;
    }
    
    .zone-card .zone-count {
        font-size: 1rem;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 500;
    }
    
    /* Títulos com efeito holográfico */
    h1 {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899, #ef4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        text-shadow: 0 0 50px rgba(59, 130, 246, 0.3);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h2 {
        font-size: 2.2rem;
        font-weight: 700;
        border-bottom: 2px solid transparent;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) border-box;
        -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        padding-bottom: 15px;
        margin-bottom: 30px;
    }
    
    h3 {
        font-size: 1.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Abas com design futurista */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        padding: 8px;
        border-radius: 50px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 12px 25px;
        font-weight: 600;
        color: #94a3b8 !important;
        transition: all 0.3s ease;
        font-size: 1rem;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 5px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Containers de métricas com efeito vidro */
    .metric-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 15px 35px -10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-container:hover {
        border-color: #3b82f6;
        box-shadow: 0 20px 40px -10px #3b82f6;
        transform: translateY(-2px);
    }
    
    /* Botões com efeito científico */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* Scrollbar científica */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
        box-shadow: 0 0 20px #3b82f6;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
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
    <div style="text-align: center; padding: 30px 0; position: relative; z-index: 10;">
        <h1>🏃 Sports Science Analytics Pro</h1>
        <p style="color: #94a3b8; font-size: 1.3rem; margin-top: 10px; letter-spacing: 2px;">
            Elite Performance Analysis Platform
        </p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 25px;">
            <span style="background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 8px 20px; border-radius: 50px; font-size: 0.9rem; box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);">⚡ Neural Analytics</span>
            <span style="background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; padding: 8px 20px; border-radius: 50px; font-size: 0.9rem; box-shadow: 0 5px 15px rgba(139, 92, 246, 0.3);">📊 Predictive Modeling</span>
            <span style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 8px 20px; border-radius: 50px; font-size: 0.9rem; box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);">🎯 Quantum Precision</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
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
    if 'ordem_personalizada' not in st.session_state:
        st.session_state.ordem_personalizada = []
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
    if 'zona_key' not in st.session_state:
        st.session_state.zona_key = 0
    if 'anotacoes' not in st.session_state:
        st.session_state.anotacoes = []
    if 'n_classes' not in st.session_state:
        st.session_state.n_classes = 5
    if 'upload_concluido' not in st.session_state:
        st.session_state.upload_concluido = False
    if 'atleta1_comp' not in st.session_state:
        st.session_state.atleta1_comp = None
    if 'atleta2_comp' not in st.session_state:
        st.session_state.atleta2_comp = None
    if 'vars_comp' not in st.session_state:
        st.session_state.vars_comp = []
    if 'window_size' not in st.session_state:
        st.session_state.window_size = 3

init_session_state()

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def interpretar_teste(p_valor, nome_teste, t):
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

def extrair_periodo(texto):
    try:
        texto = str(texto)
        partes = texto.split('-')
        if len(partes) >= 3:
            return partes[1].strip()
        return ""
    except:
        return ""

def extrair_minuto(texto):
    try:
        texto = str(texto)
        partes = texto.split('-')
        if len(partes) >= 3:
            return partes[2].strip()
        return ""
    except:
        return ""

def extrair_nome(texto):
    try:
        texto = str(texto)
        partes = texto.split('-')
        return partes[0].strip()
    except:
        return ""

def verificar_estruturas_arquivos(dataframes):
    if not dataframes:
        return False, []
    
    primeira_estrutura = dataframes[0].columns.tolist()
    
    for i, df in enumerate(dataframes[1:], 1):
        if df.columns.tolist() != primeira_estrutura:
            return False, primeira_estrutura
    
    return True, primeira_estrutura

def executive_card(titulo, valor, delta, icone, cor_status="#3b82f6"):
    """Card profissional estilo dashboard executivo"""
    delta_icon = "▲" if delta > 0 else "▼"
    delta_color = "#10b981" if delta > 0 else "#ef4444"
    
    st.markdown(f"""
    <div class="executive-card" style="border-left-color: {cor_status};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p class="label">{titulo}</p>
                <p class="value">{valor}</p>
                <p class="delta" style="color: {delta_color};">
                    {delta_icon} {abs(delta):.1f}% vs. média
                </p>
            </div>
            <div class="icon">{icone}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def time_metric_card(label, valor, sub_label="", cor="#3b82f6"):
    st.markdown(f"""
    <div class="time-metric-card" style="border-left-color: {cor};">
        <div class="label">{label}</div>
        <div class="value">{valor}</div>
        <div class="sub-value">{sub_label}</div>
    </div>
    """, unsafe_allow_html=True)

def warning_card(titulo, valor, subtitulo, icone="⚠️"):
    st.markdown(f"""
    <div class="warning-card">
        <div class="label">{icone} {titulo}</div>
        <div class="value">{valor}</div>
        <div class="sub-label">{subtitulo}</div>
    </div>
    """, unsafe_allow_html=True)

def calcular_cv(media, desvio):
    if media != 0 and not np.isnan(media) and not np.isnan(desvio):
        return (desvio / media) * 100
    return 0

def extrair_minuto_do_extremo(df, coluna_valor, coluna_minuto, extremo='max'):
    try:
        if df.empty or len(df) == 0:
            return "N/A"
        
        if extremo == 'max':
            idx_extremo = df[coluna_valor].idxmax()
        else:
            idx_extremo = df[coluna_valor].idxmin()
        
        if pd.notna(idx_extremo):
            return df.loc[idx_extremo, coluna_minuto]
        
        return "N/A"
    except:
        return "N/A"

def criar_zonas_intensidade(df, variavel, metodo='percentis'):
    if metodo == 'percentis':
        return {
            'Muito Baixa': df[variavel].quantile(0.2),
            'Baixa': df[variavel].quantile(0.4),
            'Moderada': df[variavel].quantile(0.6),
            'Alta': df[variavel].quantile(0.8),
            'Muito Alta': df[variavel].quantile(1.0)
        }
    else:  # based_on_max
        max_val = df[variavel].max()
        return {
            'Muito Baixa': max_val * 0.2,
            'Baixa': max_val * 0.4,
            'Moderada': max_val * 0.6,
            'Alta': max_val * 0.8,
            'Muito Alta': max_val
        }

def comparar_grupos(df, variavel, grupo1, grupo2):
    try:
        dados_grupo1 = df[df['Posição'] == grupo1][variavel].dropna()
        dados_grupo2 = df[df['Posição'] == grupo2][variavel].dropna()
        
        if len(dados_grupo1) < 3 or len(dados_grupo2) < 3:
            return None
        
        _, p1 = stats.shapiro(dados_grupo1)
        _, p2 = stats.shapiro(dados_grupo2)
        
        if p1 > 0.05 and p2 > 0.05:
            t_stat, p_valor = stats.ttest_ind(dados_grupo1, dados_grupo2)
            teste = "Teste t de Student"
        else:
            u_stat, p_valor = stats.mannwhitneyu(dados_grupo1, dados_grupo2)
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

def calcular_media_movel(df, variavel, window):
    """Calcula média móvel com janela especificada"""
    return df[variavel].rolling(window=window, min_periods=1, center=True).mean()

# ============================================================================
# FUNÇÃO CORRIGIDA: criar_timeline_filtrada
# ============================================================================

def criar_timeline_filtrada(df_completo, atletas_selecionados, periodos_selecionados, variavel, window_size, t):
    """
    Timeline com:
    - Linhas por atleta-período
    - Pontos vermelhos para eventos críticos
    - Marcadores vermelhos no eixo X para eventos críticos
    - Média móvel (linha tracejada branca)
    """
    if not atletas_selecionados or not periodos_selecionados:
        return None, [], []
    
    # Filtrar dados apenas para combinações válidas
    df_filtrado = df_completo[
        df_completo['Nome'].isin(atletas_selecionados) & 
        df_completo['Período'].isin(periodos_selecionados)
    ].copy()
    
    if df_filtrado.empty:
        return None, [], []
    
    # Ordenar por minuto para garantir sequência temporal
    df_filtrado = df_filtrado.sort_values('Minuto')
    
    # Identificar todas as combinações únicas atleta-período
    combinacoes = df_filtrado.groupby(['Nome', 'Período']).size().reset_index()[['Nome', 'Período']]
    combinacoes_list = list(zip(combinacoes['Nome'], combinacoes['Período']))
    
    fig = go.Figure()
    
    # Cores para cada atleta
    cores = px.colors.qualitative.Set2
    
    # Calcular valor máximo e limiar de 80% para linha vermelha
    valor_maximo = df_filtrado[variavel].max()
    limiar_80 = valor_maximo * 0.8
    
    # Adicionar linha vermelha do limiar de 80%
    fig.add_hline(
        y=limiar_80,
        line_dash="solid",
        line_color="#ef4444",
        line_width=3,
        annotation_text=f"Limiar 80%: {limiar_80:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="white", size=12)
    )
    
    # Calcular média móvel global
    df_filtrado['Media_Movel'] = calcular_media_movel(df_filtrado, variavel, window_size)
    
    # Adicionar linha da média móvel global
    fig.add_trace(go.Scatter(
        x=df_filtrado['Minuto'],
        y=df_filtrado['Media_Movel'],
        mode='lines',
        name=f'Média Móvel (janela={window_size})',
        line=dict(color='white', width=3, dash='dash'),
        opacity=0.9,
        hovertemplate='<b>Média Móvel</b><br>' +
                      '<b>Minuto:</b> %{x}<br>' +
                      '<b>Valor:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Lista para armazenar minutos de eventos críticos
    minutos_criticos = []
    
    # Plotar cada combinação atleta-período
    for i, (atleta, periodo) in enumerate(combinacoes_list):
        # Filtrar dados específicos para esta combinação
        df_combo = df_filtrado[
            (df_filtrado['Nome'] == atleta) & 
            (df_filtrado['Período'] == periodo)
        ].copy().sort_values('Minuto')
        
        if df_combo.empty:
            continue
        
        # Determinar a cor baseada no atleta
        if atleta in atletas_selecionados:
            atleta_idx = atletas_selecionados.index(atleta)
            cor_atleta = cores[atleta_idx % len(cores)]
        else:
            cor_atleta = '#3b82f6'
        
        # Separar pontos normais e críticos
        mask_critico = df_combo[variavel] > limiar_80
        df_normal = df_combo[~mask_critico]
        df_critico = df_combo[mask_critico]
        
        # Adicionar minutos críticos à lista
        minutos_criticos.extend(df_critico['Minuto'].tolist())
        
        # Plotar linha conectando todos os pontos
        fig.add_trace(go.Scatter(
            x=df_combo['Minuto'],
            y=df_combo[variavel],
            mode='lines',
            name=f"{atleta} - {periodo}",
            line=dict(color=cor_atleta, width=2),
            legendgroup=f"{atleta}_{periodo}",
            showlegend=True,
            hovertemplate='<b>Atleta:</b> ' + atleta + '<br>' +
                          '<b>Período:</b> ' + periodo + '<br>' +
                          '<b>Minuto:</b> %{x}<br>' +
                          '<b>Valor:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Plotar pontos normais
        if not df_normal.empty:
            fig.add_trace(go.Scatter(
                x=df_normal['Minuto'],
                y=df_normal[variavel],
                mode='markers',
                name=f"{atleta} - {periodo} (normal)",
                marker=dict(
                    size=8,
                    color=cor_atleta,
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                legendgroup=f"{atleta}_{periodo}",
                showlegend=False,
                hovertemplate='<b>Atleta:</b> ' + atleta + '<br>' +
                              '<b>Período:</b> ' + periodo + '<br>' +
                              '<b>Minuto:</b> %{x}<br>' +
                              '<b>Valor:</b> %{y:.2f}<br>' +
                              '<b>Status:</b> Normal<extra></extra>'
            ))
        
        # Plotar pontos críticos em VERMELHO
        if not df_critico.empty:
            for _, row in df_critico.iterrows():
                percent_acima = ((row[variavel] / limiar_80) - 1) * 100
                fig.add_trace(go.Scatter(
                    x=[row['Minuto']],
                    y=[row[variavel]],
                    mode='markers',
                    name=f"{atleta} - {periodo} (crítico)",
                    marker=dict(
                        size=12,
                        color='#ef4444',
                        symbol='circle',
                        opacity=1,
                        line=dict(color='white', width=2)
                    ),
                    legendgroup=f"{atleta}_{periodo}",
                    showlegend=False,
                    hovertemplate='<b style="color:red;">⚠️ EVENTO CRÍTICO</b><br>' +
                                  '<b>Atleta:</b> ' + atleta + '<br>' +
                                  '<b>Período:</b> ' + periodo + '<br>' +
                                  '<b>Minuto:</b> %{x}<br>' +
                                  '<b>Valor:</b> %{y:.2f}<br>' +
                                  f'<b>Acima do limiar:</b> {percent_acima:.1f}%<extra></extra>'
                ))
    
    # Adicionar marcadores vermelhos no eixo X para eventos críticos
    if minutos_criticos:
        # Encontrar o valor mínimo para posicionar os marcadores
        y_min = df_filtrado[variavel].min()
        y_range = df_filtrado[variavel].max() - y_min
        y_pos = y_min - (y_range * 0.05)  # Posicionar 5% abaixo do mínimo
        
        fig.add_trace(go.Scatter(
            x=minutos_criticos,
            y=[y_pos] * len(minutos_criticos),
            mode='markers',
            name='Eventos Críticos (eixo X)',
            marker=dict(
                size=12,
                color='#ef4444',
                symbol='triangle-down',
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            showlegend=True,
            hovertemplate='<b style="color:red;">⚠️ EVENTO CRÍTICO</b><br>' +
                          '<b>Minuto:</b> %{x}<br>' +
                          '<b>Valor acima do limiar</b><extra></extra>'
        ))
        
        # Adicionar linhas verticais tracejadas nos minutos críticos
        for minuto in minutos_criticos:
            fig.add_vline(
                x=minuto,
                line_width=1,
                line_dash="dot",
                line_color="#ef4444",
                opacity=0.3
            )
    
    # Calcular média global
    media_global = df_filtrado[variavel].mean()
    desvio_global = df_filtrado[variavel].std()
    
    # Adicionar linha da média global
    fig.add_hline(
        y=media_global, 
        line_dash="dash", 
        line_color="#94a3b8",
        annotation_text=f"Média Global: {media_global:.2f}", 
        annotation_position="bottom left",
        annotation_font=dict(color="white", size=10)
    )
    
    # Adicionar área de desvio padrão
    fig.add_hrect(
        y0=media_global-desvio_global, 
        y1=media_global+desvio_global,
        fillcolor="#3b82f6", 
        opacity=0.1, 
        line_width=0,
        annotation_text="±1 DP",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=dict(
            text=f"📈 Análise Temporal - {variavel}",
            font=dict(size=24, color='#3b82f6'),
            x=0.5
        ),
        xaxis_title="Minuto",
        yaxis_title=variavel,
        hovermode='closest',
        plot_bgcolor='rgba(30,41,59,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        showlegend=True,
        legend=dict(
            font=dict(color='white', size=10),
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(30,41,59,0.9)',
            bordercolor='#3b82f6',
            borderwidth=2
        ),
        height=700,
        margin=dict(l=50, r=150, t=80, b=50),
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

def criar_tabela_destaque(df, colunas_destaque):
    """Tabela com células destacadas baseado em valores"""
    styled_df = df.style
    
    # Aplicar gradiente nas colunas numéricas
    for col in colunas_destaque:
        if col in df.select_dtypes(include=[np.number]).columns:
            styled_df = styled_df.background_gradient(
                subset=[col],
                cmap='viridis'
            )
    
    return styled_df

def comparar_atletas(df, atleta1, atleta2, variaveis, t):
    """Comparação lado a lado de dois atletas"""
    
    dados1 = df[df['Nome'] == atleta1][variaveis].mean()
    dados2 = df[df['Nome'] == atleta2][variaveis].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {atleta1}")
        for var in variaveis:
            delta = ((dados1[var] - dados2[var]) / dados2[var]) * 100 if dados2[var] != 0 else 0
            cor = "#10b981" if delta > 0 else "#ef4444"
            st.markdown(f"""
            <div style="background: #1e293b; padding: 15px; border-radius: 12px; margin: 5px 0;
                        border-left: 4px solid {cor}; border: 1px solid rgba(59, 130, 246, 0.2);">
                <span style="color: #94a3b8;">{var}:</span>
                <span style="color: white; font-weight: bold; float: right;">{dados1[var]:.2f}</span>
                <br>
                <span style="color: {cor}; font-size: 0.9rem;">
                    {delta:+.1f}% vs {atleta2}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### {atleta2}")
        for var in variaveis:
            delta = ((dados2[var] - dados1[var]) / dados1[var]) * 100 if dados1[var] != 0 else 0
            cor = "#10b981" if delta > 0 else "#ef4444"
            st.markdown(f"""
            <div style="background: #1e293b; padding: 15px; border-radius: 12px; margin: 5px 0;
                        border-left: 4px solid {cor}; border: 1px solid rgba(59, 130, 246, 0.2);">
                <span style="color: #94a3b8;">{var}:</span>
                <span style="color: white; font-weight: bold; float: right;">{dados2[var]:.2f}</span>
                <br>
                <span style="color: {cor}; font-size: 0.9rem;">
                    {delta:+.1f}% vs {atleta1}
                </span>
            </div>
            """, unsafe_allow_html=True)

def sistema_anotacoes(t):
    """Sistema de anotações profissionais"""
    
    with st.expander("📝 Anotações da Análise"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            nova_anotacao = st.text_area("Nova anotação", height=100, key="nova_anotacao")
        
        with col2:
            if st.button("➕ Adicionar", use_container_width=True):
                if nova_anotacao:
                    st.session_state.anotacoes.append({
                        'data': datetime.now().strftime("%d/%m/%Y %H:%M"),
                        'texto': nova_anotacao
                    })
                    st.rerun()
        
        # Listar anotações
        for i, anotacao in enumerate(reversed(st.session_state.anotacoes)):
            st.markdown(f"""
            <div style="background: #1e293b; padding: 12px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #3b82f6;">
                <p style="color: #94a3b8; font-size: 0.85rem;">{anotacao['data']}</p>
                <p style="color: white; margin: 5px 0;">{anotacao['texto']}</p>
            </div>
            """, unsafe_allow_html=True)

def time_range_selector(t):
    """Seletor de período estilo Google Analytics"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        periodo = st.selectbox(
            "Período",
            ["Hoje", "Últimos 7 dias", "Últimos 30 dias", "Este mês", "Personalizado"],
            index=2,
            key="periodo_selector"
        )
    
    data_inicio = None
    data_fim = None
    
    if periodo == "Personalizado":
        with col2:
            data_inicio = st.date_input("Data inicial", key="data_inicio")
        with col3:
            data_fim = st.date_input("Data final", key="data_fim")
    else:
        # Simulação - em um caso real, você usaria datas reais dos dados
        data_fim = datetime.now()
        if periodo == "Hoje":
            data_inicio = data_fim
        elif periodo == "Últimos 7 dias":
            data_inicio = data_fim - timedelta(days=7)
        elif periodo == "Últimos 30 dias":
            data_inicio = data_fim - timedelta(days=30)
        elif periodo == "Este mês":
            data_inicio = data_fim.replace(day=1)
    
    return data_inicio, data_fim

# ============================================================================
# CALLBACKS - ATUALIZAÇÃO INSTANTÂNEA
# ============================================================================

def atualizar_metodo_zona():
    """Callback para atualizar método de zona"""
    valor_radio = st.session_state.metodo_zona_radio
    if valor_radio in ["Percentis", "Percentiles"]:
        st.session_state.metodo_zona = 'percentis'
    else:
        st.session_state.metodo_zona = 'based_on_max'
    st.session_state.zona_key += 1

def atualizar_grupos():
    """Callback para atualizar grupos de comparação"""
    st.session_state.grupo1 = st.session_state.grupo1_select
    st.session_state.grupo2 = st.session_state.grupo2_select

def atualizar_comparacao_atletas1():
    """Callback para atualizar atleta 1"""
    st.session_state.atleta1_comp = st.session_state.atleta1_comp_select

def atualizar_comparacao_atletas2():
    """Callback para atualizar atleta 2"""
    st.session_state.atleta2_comp = st.session_state.atleta2_comp_select

def atualizar_vars_comp():
    """Callback para atualizar variáveis de comparação"""
    st.session_state.vars_comp = st.session_state.vars_comp_select

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🌐 Idioma / Language</h2>", unsafe_allow_html=True)
    
    idioma_opcoes = ['pt', 'en', 'es']
    idioma_idx = idioma_opcoes.index(st.session_state.idioma) if st.session_state.idioma in idioma_opcoes else 0
    
    idioma = st.selectbox(
        "", 
        idioma_opcoes,
        index=idioma_idx,
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
    
    # Processar upload apenas uma vez
    if upload_files and len(upload_files) > 0 and not st.session_state.upload_concluido:
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
                    except Exception as e:
                        continue
                
                if dataframes:
                    estruturas_ok, _ = verificar_estruturas_arquivos(dataframes)
                    
                    if not estruturas_ok:
                        st.error("❌ " + ("Arquivos com estruturas diferentes" if st.session_state.idioma == 'pt' else 
                                        "Files with different structures" if st.session_state.idioma == 'en' else
                                        "Archivos con estructuras diferentes"))
                        st.stop()
                    
                    data = pd.concat(dataframes, ignore_index=True)
                    
                    if data.shape[1] >= 3 and not data.empty:
                        primeira_coluna = data.iloc[:, 0].astype(str)
                        segunda_coluna = data.iloc[:, 1].astype(str)
                        
                        nomes = primeira_coluna.apply(extrair_nome)
                        minutos = primeira_coluna.apply(extrair_minuto)
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
                                st.session_state.ordem_personalizada = periodos_unicos.copy()
                                st.session_state.upload_files_names = arquivos_validos
                                st.session_state.upload_concluido = True
                                
                                if variaveis_quant and st.session_state.variavel_selecionada is None:
                                    st.session_state.variavel_selecionada = variaveis_quant[0]
                                
                                sucesso_msg = ("arquivo(s) carregado(s)" if st.session_state.idioma == 'pt' else
                                              "file(s) loaded" if st.session_state.idioma == 'en' else
                                              "archivo(s) cargado(s)")
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
                f"Selecionar todas as {t['position'].lower()}s" if st.session_state.idioma == 'pt' else
                f"Select all {t['position'].lower()}s" if st.session_state.idioma == 'en' else
                f"Seleccionar todas las {t['position'].lower()}s",
                value=len(st.session_state.posicoes_selecionadas) == len(st.session_state.todos_posicoes),
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
                f"Selecionar todos os {t['period'].lower()}s" if st.session_state.idioma == 'pt' else
                f"Select all {t['period'].lower()}s" if st.session_state.idioma == 'en' else
                f"Seleccionar todos los {t['period'].lower()}s",
                value=len(st.session_state.periodos_selecionados) == len(st.session_state.todos_periodos),
                key="todos_periodos_check"
            )
            
            if selecionar_todos:
                if st.session_state.periodos_selecionados != st.session_state.todos_periodos:
                    st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                    st.session_state.ordem_personalizada = st.session_state.todos_periodos.copy()
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
        
        # Seção de atletas
        st.markdown("---")
        st.markdown(f"<h2 class='sidebar-title'>👤 {t['athlete']}</h2>", unsafe_allow_html=True)
        
        df_temp = st.session_state.df_completo.copy()
        if st.session_state.posicoes_selecionadas:
            df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
        if st.session_state.periodos_selecionados:
            df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
        
        atletas_disponiveis = sorted(df_temp['Nome'].unique())
        
        # Inicializar atletas_selecionados se estiver vazio
        if not st.session_state.atletas_selecionados and atletas_disponiveis:
            st.session_state.atletas_selecionados = atletas_disponiveis.copy()
            st.rerun()
        
        # Checkbox para selecionar todos
        selecionar_todos_atletas = st.checkbox(
            f"Selecionar todos os {t['athlete'].lower()}s" if st.session_state.idioma == 'pt' else
            f"Select all {t['athlete'].lower()}s" if st.session_state.idioma == 'en' else
            f"Seleccionar todos los {t['athlete'].lower()}s",
            value=len(st.session_state.atletas_selecionados) == len(atletas_disponiveis) and len(atletas_disponiveis) > 0,
            key="todos_atletas_check"
        )
        
        if selecionar_todos_atletas:
            if st.session_state.atletas_selecionados != atletas_disponiveis:
                st.session_state.atletas_selecionados = atletas_disponiveis
                st.session_state.dados_processados = False
                st.rerun()
        
        # Multiselect
        atletas_sel = st.multiselect(
            "",
            options=atletas_disponiveis,
            default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else (atletas_disponiveis[:1] if atletas_disponiveis else []),
            label_visibility="collapsed",
            key="atletas_selector"
        )
        
        if atletas_sel != st.session_state.atletas_selecionados:
            st.session_state.atletas_selecionados = atletas_sel
            st.session_state.dados_processados = False
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"<h2 class='sidebar-title'>⚙️ {t['config']}</h2>", unsafe_allow_html=True)
        
        n_classes = st.slider(f"Classes do histograma:", 3, 20, st.session_state.n_classes, key="classes_slider")
        if n_classes != st.session_state.n_classes:
            st.session_state.n_classes = n_classes
            st.rerun()
        
        window_size = st.slider("Janela da Média Móvel:", 2, 10, st.session_state.window_size, key="window_slider")
        if window_size != st.session_state.window_size:
            st.session_state.window_size = window_size
            st.rerun()
        
        st.markdown("---")
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
    
    with st.spinner('🔄 ' + ("Gerando análises..." if st.session_state.idioma == 'pt' else 
                             "Generating analysis..." if st.session_state.idioma == 'en' else
                             "Generando análisis...")):
        time.sleep(0.5)
        
        df_completo = st.session_state.df_completo
        atletas_selecionados = st.session_state.atletas_selecionados
        posicoes_selecionadas = st.session_state.posicoes_selecionadas
        periodos_selecionados = st.session_state.periodos_selecionados
        variavel_analise = st.session_state.variavel_selecionada
        n_classes = st.session_state.n_classes
        window_size = st.session_state.window_size
        
        df_filtrado = df_completo[
            df_completo['Nome'].isin(atletas_selecionados) & 
            df_completo['Posição'].isin(posicoes_selecionadas) &
            df_completo['Período'].isin(periodos_selecionados)
        ].copy()
        
        df_filtrado = df_filtrado.dropna(subset=[variavel_analise])
        
        if df_filtrado.empty:
            st.warning("⚠️ " + ("Nenhum dado encontrado" if st.session_state.idioma == 'pt' else 
                               "No data found" if st.session_state.idioma == 'en' else
                               "No se encontraron datos"))
        else:
            st.session_state.dados_processados = True
            t = translations[st.session_state.idioma]
            
            # ====================================================================
            # DASHBOARD EXECUTIVO - VISÃO GERAL
            # ====================================================================
            st.markdown(f"<h2>📊 Executive Dashboard</h2>", unsafe_allow_html=True)
            
            media_global = df_filtrado[variavel_analise].mean()
            media_posicoes = df_filtrado.groupby('Posição')[variavel_analise].mean()
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
            
            data_inicio, data_fim = time_range_selector(t)
            
            st.markdown("---")
            
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
                st.markdown(f"<h4>📋 {t['tab_distribution']} ({n_classes} classes)</h4>", unsafe_allow_html=True)
                
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
                
                opcoes = [t['percentiles'], t['based_on_max']]
                idx_atual = 0 if st.session_state.metodo_zona == 'percentis' else 1
                
                metodo_zona = st.radio(
                    t['zone_method'],
                    opcoes,
                    index=idx_atual,
                    key="metodo_zona_radio",
                    on_change=atualizar_metodo_zona
                )
                
                zonas = criar_zonas_intensidade(df_filtrado, variavel_analise, st.session_state.metodo_zona)
                
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
                            <div class="zone-card" style="border-left-color: {cores_zonas[i]};" key="zona_{i}_{st.session_state.zona_key}">
                                <div class="zone-name">{zona}</div>
                                <div class="zone-value">{limite:.1f}</div>
                                <div class="zone-count">{n_obs} obs ({n_obs/len(df_filtrado)*100:.0f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"<h4>⏱️ Evolução Temporal com Média Móvel</h4>", unsafe_allow_html=True)
                st.caption(f"Mostrando dados filtrados com linha vermelha do limiar de 80%. Pontos em VERMELHO são eventos críticos. Marcadores vermelhos no eixo X indicam momentos críticos. Média móvel com janela={window_size}.")
                
                # USAR A FUNÇÃO CORRIGIDA
                resultado = criar_timeline_filtrada(
                    df_completo, 
                    atletas_selecionados, 
                    periodos_selecionados, 
                    variavel_analise,
                    window_size,
                    t
                )
                
                if resultado and resultado[0] is not None:
                    fig_tempo_filtrada, combinacoes, minutos_criticos = resultado
                    st.plotly_chart(fig_tempo_filtrada, use_container_width=True)
                    
                    # Mostrar estatísticas das combinações
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"✅ **{len(combinacoes)} combinações atleta-período**")
                    with col2:
                        st.warning(f"⚠️ **{len(minutos_criticos)} eventos críticos**")
                    with col3:
                        st.info(f"📊 **{len(df_filtrado)} observações totais**")
                    
                    # Tabela de combinações
                    df_combinacoes = pd.DataFrame(combinacoes, columns=['Atleta', 'Período'])
                    df_combinacoes = df_combinacoes.sort_values(['Atleta', 'Período']).reset_index(drop=True)
                    
                    with st.expander("📋 Ver combinações atleta-período"):
                        st.dataframe(df_combinacoes, use_container_width=True, hide_index=True)
                    
                    # Lista de minutos críticos
                    if minutos_criticos:
                        with st.expander("⚠️ Minutos com eventos críticos"):
                            dados_criticos = []
                            for m in set(minutos_criticos):
                                valor = df_filtrado[df_filtrado['Minuto'] == m][variavel_analise].values
                                if len(valor) > 0:
                                    percent_acima = ((valor[0] / limiar_80) - 1) * 100
                                    dados_criticos.append({
                                        'Minuto': m,
                                        'Valor': valor[0],
                                        'Acima do limiar (%)': percent_acima
                                    })
                            df_criticos = pd.DataFrame(dados_criticos).sort_values('Minuto')
                            if not df_criticos.empty:
                                st.dataframe(df_criticos.style.format({
                                    'Valor': '{:.2f}',
                                    'Acima do limiar (%)': '{:.1f}'
                                }), use_container_width=True, hide_index=True)
                    
                    # Estatísticas adicionais
                    st.info(f"ℹ️ Estatísticas: Média={media_tempo:.2f} | Desvio={df_tempo[variavel_analise].std():.2f} | Limiar 80%={limiar_80:.2f}")
                else:
                    st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados.")
                
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
                    st.error("❌ " + ("Amostra muito pequena (n < 3)" if st.session_state.idioma == 'pt' else 
                                    "Sample too small (n < 3)" if st.session_state.idioma == 'en' else
                                    "Muestra muy pequeña (n < 3)"))
                elif n_teste > 5000:
                    st.info("ℹ️ " + ("Amostra grande demais. Usando D'Agostino-Pearson." if st.session_state.idioma == 'pt' else
                                    "Sample too large. Using D'Agostino-Pearson." if st.session_state.idioma == 'en' else
                                    "Muestra demasiado grande. Usando D'Agostino-Pearson."))
                    try:
                        k2, p = stats.normaltest(dados_teste)
                        interpretar_teste(p, "D'Agostino-Pearson", t)
                    except:
                        st.warning("⚠️ " + ("Teste alternativo não disponível" if st.session_state.idioma == 'pt' else
                                          "Alternative test not available" if st.session_state.idioma == 'en' else
                                          "Prueba alternativa no disponible"))
                else:
                    try:
                        shapiro = stats.shapiro(dados_teste)
                        interpretar_teste(shapiro.pvalue, "Shapiro-Wilk", t)
                    except:
                        st.error("❌ " + ("Erro no teste" if st.session_state.idioma == 'pt' else
                                        "Test error" if st.session_state.idioma == 'en' else
                                        "Error en la prueba"))
                
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
                    
                    styled_df = criar_tabela_destaque(df_resumo, ['Média', f'Máx {variavel_analise}', f'Mín {variavel_analise}', 'CV (%)'])
                    
                    st.dataframe(
                        styled_df.format({
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
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.8); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
                        <h5 style="color: #3b82f6;">{t['iqr_title']}</h5>
                        <p style="color: #94a3b8;">{t['iqr_explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
                    
                    st.caption(f"📌 {t['iqr_title']}: {t['iqr_explanation']}")
            
            # ABA 4: CORRELAÇÕES
            with tabs[3]:
                st.markdown(f"<h3>{t['tab_correlation']}</h3>", unsafe_allow_html=True)
                
                if len(st.session_state.variaveis_quantitativas) > 1:
                    vars_corr = st.multiselect(
                        t['tab_correlation'],
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
                        st.info("ℹ️ " + ("Selecione pelo menos 2 variáveis" if st.session_state.idioma == 'pt' else 
                                       "Select at least 2 variables" if st.session_state.idioma == 'en' else
                                       "Seleccione al menos 2 variables"))
                else:
                    st.info("ℹ️ " + ("São necessárias pelo menos 2 variáveis" if st.session_state.idioma == 'pt' else 
                                   "At least 2 variables are needed" if st.session_state.idioma == 'en' else
                                   "Se necesitan al menos 2 variables"))
            
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
                            on_change=atualizar_grupos
                        )
                    with col_comp2:
                        grupo2 = st.selectbox(
                            f"{t['position']} 2:", 
                            posicoes_selecionadas, 
                            index=min(1, len(posicoes_selecionadas)-1), 
                            key="grupo2_select",
                            on_change=atualizar_grupos
                        )
                    
                    grupo1_atual = st.session_state.grupo1 if st.session_state.grupo1 is not None else grupo1
                    grupo2_atual = st.session_state.grupo2 if st.session_state.grupo2 is not None else grupo2
                    
                    if grupo1_atual and grupo2_atual and grupo1_atual != grupo2_atual:
                        resultado = comparar_grupos(df_filtrado, variavel_analise, grupo1_atual, grupo2_atual)
                        
                        if resultado:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>📊 {t['tab_comparison']}</h4>
                                <hr style="border-color: #334155;">
                                <p><strong>{t['position']} 1 ({grupo1_atual}):</strong> {resultado['media_g1']:.2f} ± {resultado['std_g1']:.2f} (n={resultado['n_g1']})</p>
                                <p><strong>{t['position']} 2 ({grupo2_atual}):</strong> {resultado['media_g2']:.2f} ± {resultado['std_g2']:.2f} (n={resultado['n_g2']})</p>
                                <p><strong>Teste:</strong> {resultado['teste']}</p>
                                <p><strong>p-valor:</strong> {resultado['p_valor']:.4f}</p>
                                <p><strong>Diferença:</strong> {resultado['media_g1'] - resultado['media_g2']:.2f}</p>
                                <p><strong>{'✅ Significativo' if resultado['significativo'] else '❌ Não significativo'}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig_comp = go.Figure()
                            
                            dados_comp1 = df_filtrado[df_filtrado['Posição'] == grupo1_atual][variavel_analise]
                            dados_comp2 = df_filtrado[df_filtrado['Posição'] == grupo2_atual][variavel_analise]
                            
                            fig_comp.add_trace(go.Box(
                                y=dados_comp1,
                                name=grupo1_atual,
                                boxmean='sd',
                                marker_color='#3b82f6',
                                line_color='white'
                            ))
                            
                            fig_comp.add_trace(go.Box(
                                y=dados_comp2,
                                name=grupo2_atual,
                                boxmean='sd',
                                marker_color='#ef4444',
                                line_color='white'
                            ))
                            
                            fig_comp.update_layout(
                                title=f"{grupo1_atual} vs {grupo2_atual} - {variavel_analise}",
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
                            st.warning("⚠️ " + ("Dados insuficientes para comparação" if st.session_state.idioma == 'pt' else 
                                              "Insufficient data for comparison" if st.session_state.idioma == 'en' else
                                              "Datos insuficientes para comparación"))
                    else:
                        st.info("ℹ️ " + ("Selecione grupos diferentes" if st.session_state.idioma == 'pt' else 
                                       "Select different groups" if st.session_state.idioma == 'en' else
                                       "Seleccione grupos diferentes"))
                else:
                    st.info("ℹ️ " + ("Selecione pelo menos 2 posições" if st.session_state.idioma == 'pt' else 
                                   "Select at least 2 positions" if st.session_state.idioma == 'en' else
                                   "Seleccione al menos 2 posiciones"))
            
            # ABA 6: EXECUTIVO
            with tabs[5]:
                st.markdown(f"<h3>{t['tab_executive']}</h3>", unsafe_allow_html=True)
                
                st.markdown("### 🆚 Comparação de Atletas")
                if len(atletas_selecionados) >= 2:
                    col_atl1, col_atl2 = st.columns(2)
                    with col_atl1:
                        atleta1_comp = st.selectbox(
                            "Atleta 1", 
                            atletas_selecionados, 
                            index=0, 
                            key="atleta1_comp_select",
                            on_change=atualizar_comparacao_atletas1
                        )
                    with col_atl2:
                        atleta2_comp = st.selectbox(
                            "Atleta 2", 
                            atletas_selecionados, 
                            index=min(1, len(atletas_selecionados)-1), 
                            key="atleta2_comp_select",
                            on_change=atualizar_comparacao_atletas2
                        )
                    
                    a1_atual = st.session_state.atleta1_comp if st.session_state.atleta1_comp is not None else atleta1_comp
                    a2_atual = st.session_state.atleta2_comp if st.session_state.atleta2_comp is not None else atleta2_comp
                    
                    if a1_atual != a2_atual:
                        vars_comp = st.multiselect(
                            "Variáveis para comparar",
                            st.session_state.variaveis_quantitativas,
                            default=st.session_state.vars_comp if st.session_state.vars_comp else st.session_state.variaveis_quantitativas[:3],
                            key="vars_comp_select",
                            on_change=atualizar_vars_comp
                        )
                        
                        if len(vars_comp) >= 1:
                            comparar_atletas(df_filtrado, a1_atual, a2_atual, vars_comp, t)
                    else:
                        st.info("Selecione atletas diferentes para comparação")
                else:
                    st.info("Selecione pelo menos 2 atletas para comparação")
                
                st.markdown("---")
                
                sistema_anotacoes(t)
            
            with st.expander("📋 " + ("Visualizar dados brutos filtrados" if st.session_state.idioma == 'pt' else 
                                     "View filtered raw data" if st.session_state.idioma == 'en' else
                                     "Ver datos brutos filtrados")):
                st.dataframe(df_filtrado, use_container_width=True)
    
    st.session_state.processar_click = False

elif st.session_state.df_completo is None:
    t = translations[st.session_state.idioma]
    st.info(t['step1'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(t['file_format'])
        
        exemplo_data = {
            'Nome-Período-Minuto': [
                'Mariano-1 TEMPO-00:00-01:00',
                'Maria-SEGUNDO TEMPO-05:00-06:00',
                'Joao-2 TEMPO-44:00-45:00',
                'Marta-PRIMEIRO TEMPO-11:00-12:00',
                'Pedro-1 TEMPO-15:00-16:00',
                'Ana-SEGUNDO TEMPO-22:00-23:00'
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
        
elif st.session_state.dados_processados:
    t = translations[st.session_state.idioma]
    st.info(t['step2'])
    
    with st.expander("📋 " + ("Preview dos dados carregados" if st.session_state.idioma == 'pt' else 
                             "Preview of loaded data" if st.session_state.idioma == 'en' else
                             "Vista previa de datos cargados")):
        if st.session_state.upload_files_names:
            st.caption(f"**{t['upload']}:** {', '.join(st.session_state.upload_files_names)}")
            st.markdown("---")
        
        st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
        st.caption(f"**{t['observations']}:** {len(st.session_state.df_completo)}")
        st.caption(f"**{t['variable']}s:** {', '.join(st.session_session.variaveis_quantitativas)}")
        if st.session_state.todos_posicoes:
            st.caption(f"**{t['positions']}:** {', '.join(st.session_state.todos_posicoes)}")
        if st.session_state.todos_periodos:
            st.caption(f"**{t['periods']}:** {', '.join(st.session_state.todos_periodos)}")