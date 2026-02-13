import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io
import base64
from datetime import datetime
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
# INTERNACIONALIZAÇÃO (i18n)
# ============================================================================

translations = {
    'pt': {
        # Títulos e menus
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Dashboard Profissional para Análise de Desempenho Esportivo',
        'upload': 'Upload dos Dados',
        'variable': 'Variável',
        'position': 'Posição',
        'period': 'Período',
        'athlete': 'Atleta',
        'config': 'Configurações',
        
        # Abas
        'tab_distribution': '📊 Distribuição',
        'tab_temporal': '📈 Estatísticas & Temporal',
        'tab_boxplots': '📦 Boxplots',
        'tab_correlation': '🔥 Correlações',
        'tab_comparison': '⚖️ Comparações',
        'tab_clusters': '🎯 Clusters',
        'tab_fatigue': '⚡ Análise de Fadiga',
        'tab_progression': '📈 Progressão Individual',
        'tab_radar': '🕷️ Radar Chart',
        'tab_report': '📋 Relatório',
        
        # Métricas
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
        
        # Temporal
        'max_value': 'VALOR MÁXIMO',
        'minute_of_max': 'Minuto do Máx',
        'threshold_80': 'LIMIAR 80%',
        'critical_events': 'EVENTOS CRÍTICOS',
        'above_threshold': 'acima do limiar de 80%',
        
        # Zonas de intensidade
        'intensity_zones': '🎚️ Zonas de Intensidade',
        'zone_method': 'Método de definição',
        'percentiles': 'Percentis',
        'manual': 'Manual',
        'based_on_max': 'Baseado no Máximo',
        'very_low': 'Muito Baixa',
        'low': 'Baixa',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muito Alta',
        
        # Botões
        'process': '🚀 Processar Análise',
        'generate_report': '📄 Gerar Relatório PDF',
        'download_report': '📥 Download Relatório',
        
        # Estatísticas
        'descriptive_stats': '📊 Estatísticas Descritivas',
        'confidence_interval': '🎯 Intervalo de Confiança (95%)',
        'normality_test': '🧪 Teste de Normalidade',
        'summary_by_group': '🏃 Resumo por Atleta, Posição e Período',
        
        # Interpretações
        'symmetric': 'Aproximadamente simétrica',
        'moderate_skew': 'Moderadamente assimétrica',
        'high_skew': 'Fortemente assimétrica',
        'leptokurtic': 'Leptocúrtica (caudas pesadas)',
        'platykurtic': 'Platicúrtica (caudas leves)',
        'mesokurtic': 'Mesocúrtica (normal)',
        
        # Correlação
        'strong_positive': 'Correlação forte positiva',
        'moderate_positive': 'Correlação moderada positiva',
        'weak_positive': 'Correlação fraca positiva',
        'very_weak_positive': 'Correlação muito fraca positiva',
        'very_weak_negative': 'Correlação muito fraca negativa',
        'weak_negative': 'Correlação fraca negativa',
        'moderate_negative': 'Correlação moderada negativa',
        'strong_negative': 'Correlação forte negativa',
        
        # IQR explanation
        'iqr_title': '📌 O que é IQR?',
        'iqr_explanation': '''
            <strong>IQR (Intervalo Interquartil)</strong> é a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1). 
            Ele representa a amplitude dos 50% centrais dos dados, sendo uma medida robusta de dispersão menos sensível a outliers.
            Um IQR maior indica maior variabilidade nos dados.
        ''',
        
        # Fatigue analysis
        'fatigue_analysis': '⚡ Análise de Fadiga',
        'baseline': 'Baseline',
        'fatigue_threshold': 'Limiar Fadiga (-15%)',
        'fatigue_periods': 'Períodos de Fadiga',
        'avg_decline': 'Declínio Médio',
        'max_decline': 'Declínio Máximo',
        'fatigue_minutes': 'Minutos em Fadiga',
        'fatigue_percentage': 'Percentual em Fadiga',
        
        # Cluster analysis
        'cluster_analysis': '🎯 Análise de Clusters',
        'n_clusters': 'Número de clusters',
        'cluster_vars': 'Variáveis para clusterização',
        'cluster_plot': 'Visualização dos Clusters (PCA)',
        'cluster_stats': 'Estatísticas por Cluster',
        
        # Progression
        'individual_progression': '📈 Progressão Individual',
        'select_athlete': 'Selecione o atleta',
        'progression_rate': 'Taxa de Variação',
        'trend_line': 'Tendência',
        
        # Radar
        'radar_chart': '🕷️ Radar Chart',
        'select_vars_radar': 'Selecione as variáveis para o radar',
        'team_average': 'Média do Time',
        
        # Report
        'report_title': 'Relatório de Análise de Desempenho',
        'report_date': 'Data do relatório',
        'report_variable': 'Variável analisada',
        
        # Steps
        'step1': '👈 **Passo 1:** Faça upload de um ou mais arquivos CSV para começar',
        'step2': '👈 **Passo 2:** Selecione os filtros e clique em Processar Análise',
        
        # File format
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
        'tip_text': 'Você pode selecionar múltiplos arquivos CSV com a mesma estrutura. O sistema verificará automaticamente a compatibilidade.',
        'multi_file_ex': '📁 Exemplo de uso com múltiplos arquivos',
        'multi_file_text': '''
            ### Carregando múltiplos arquivos:
            1. Prepare seus arquivos CSV com a **mesma estrutura** de colunas
            2. Selecione todos os arquivos desejados no seletor de arquivos
            3. O sistema irá verificar compatibilidade e concatenar automaticamente
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
        'tab_clusters': '🎯 Clusters',
        'tab_fatigue': '⚡ Fatigue Analysis',
        'tab_progression': '📈 Individual Progression',
        'tab_radar': '🕷️ Radar Chart',
        'tab_report': '📋 Report',
        
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
        'minute_of_max': 'Max Minute',
        'threshold_80': '80% THRESHOLD',
        'critical_events': 'CRITICAL EVENTS',
        'above_threshold': 'above 80% threshold',
        
        'intensity_zones': '🎚️ Intensity Zones',
        'zone_method': 'Definition method',
        'percentiles': 'Percentiles',
        'manual': 'Manual',
        'based_on_max': 'Based on Maximum',
        'very_low': 'Very Low',
        'low': 'Low',
        'moderate': 'Moderate',
        'high': 'High',
        'very_high': 'Very High',
        
        'process': '🚀 Process Analysis',
        'generate_report': '📄 Generate PDF Report',
        'download_report': '📥 Download Report',
        
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
        'iqr_explanation': '''
            <strong>IQR (Interquartile Range)</strong> is the difference between the third quartile (Q3) and the first quartile (Q1).
            It represents the range of the middle 50% of the data, being a robust measure of dispersion less sensitive to outliers.
            A larger IQR indicates greater variability in the data.
        ''',
        
        'fatigue_analysis': '⚡ Fatigue Analysis',
        'baseline': 'Baseline',
        'fatigue_threshold': 'Fatigue Threshold (-15%)',
        'fatigue_periods': 'Fatigue Periods',
        'avg_decline': 'Average Decline',
        'max_decline': 'Maximum Decline',
        'fatigue_minutes': 'Minutes in Fatigue',
        'fatigue_percentage': 'Fatigue Percentage',
        
        'cluster_analysis': '🎯 Cluster Analysis',
        'n_clusters': 'Number of clusters',
        'cluster_vars': 'Variables for clustering',
        'cluster_plot': 'Cluster Visualization (PCA)',
        'cluster_stats': 'Statistics by Cluster',
        
        'individual_progression': '📈 Individual Progression',
        'select_athlete': 'Select athlete',
        'progression_rate': 'Rate of Change',
        'trend_line': 'Trend Line',
        
        'radar_chart': '🕷️ Radar Chart',
        'select_vars_radar': 'Select variables for radar',
        'team_average': 'Team Average',
        
        'report_title': 'Performance Analysis Report',
        'report_date': 'Report date',
        'report_variable': 'Analyzed variable',
        
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
        'tip_text': 'You can select multiple CSV files with the same structure. The system will automatically check compatibility.',
        'multi_file_ex': '📁 Example with multiple files',
        'multi_file_text': '''
            ### Loading multiple files:
            1. Prepare your CSV files with the **same column structure**
            2. Select all desired files in the file selector
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
        'tab_clusters': '🎯 Clústers',
        'tab_fatigue': '⚡ Análisis de Fatiga',
        'tab_progression': '📈 Progresión Individual',
        'tab_radar': '🕷️ Gráfico Radar',
        'tab_report': '📋 Informe',
        
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
        'minute_of_max': 'Minuto del Máx',
        'threshold_80': 'UMBRAL 80%',
        'critical_events': 'EVENTOS CRÍTICOS',
        'above_threshold': 'por encima del umbral 80%',
        
        'intensity_zones': '🎚️ Zonas de Intensidad',
        'zone_method': 'Método de definición',
        'percentiles': 'Percentiles',
        'manual': 'Manual',
        'based_on_max': 'Basado en Máximo',
        'very_low': 'Muy Baja',
        'low': 'Baja',
        'moderate': 'Moderada',
        'high': 'Alta',
        'very_high': 'Muy Alta',
        
        'process': '🚀 Procesar Análisis',
        'generate_report': '📄 Generar Informe PDF',
        'download_report': '📥 Descargar Informe',
        
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
        'iqr_explanation': '''
            <strong>IQR (Rango Intercuartil)</strong> es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1).
            Representa la amplitud del 50% central de los datos, siendo una medida robusta de dispersión menos sensible a valores atípicos.
            Un IQR mayor indica mayor variabilidad en los datos.
        ''',
        
        'fatigue_analysis': '⚡ Análisis de Fatiga',
        'baseline': 'Línea base',
        'fatigue_threshold': 'Umbral de Fatiga (-15%)',
        'fatigue_periods': 'Períodos de Fatiga',
        'avg_decline': 'Declive Medio',
        'max_decline': 'Declive Máximo',
        'fatigue_minutes': 'Minutos en Fatiga',
        'fatigue_percentage': 'Porcentaje de Fatiga',
        
        'cluster_analysis': '🎯 Análisis de Clústers',
        'n_clusters': 'Número de clústers',
        'cluster_vars': 'Variables para clusterización',
        'cluster_plot': 'Visualización de Clústers (PCA)',
        'cluster_stats': 'Estadísticas por Clúster',
        
        'individual_progression': '📈 Progresión Individual',
        'select_athlete': 'Seleccionar atleta',
        'progression_rate': 'Tasa de Variación',
        'trend_line': 'Línea de Tendencia',
        
        'radar_chart': '🕷️ Gráfico Radar',
        'select_vars_radar': 'Seleccionar variables para el radar',
        'team_average': 'Promedio del Equipo',
        
        'report_title': 'Informe de Análisis de Rendimiento',
        'report_date': 'Fecha del informe',
        'report_variable': 'Variable analizada',
        
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
        'tip_text': 'Puede seleccionar múltiples archivos CSV con la misma estructura. El sistema verificará automáticamente la compatibilidad.',
        'multi_file_ex': '📁 Ejemplo con múltiples archivos',
        'multi_file_text': '''
            ### Cargando múltiples archivos:
            1. Prepare sus archivos CSV con la **misma estructura** de columnas
            2. Seleccione todos los archivos deseados en el selector
            3. El sistema verificará compatibilidad y concatenará automáticamente
        '''
    }
}

# ============================================================================
# DETECÇÃO DE DISPOSITIVO MÓVEL
# ============================================================================

def is_mobile():
    """Detecta se está em dispositivo móvel"""
    try:
        user_agent = st.query_params.get('user_agent', [''])[0]
        mobile_keywords = ['android', 'iphone', 'ipad', 'mobile']
        return any(keyword in user_agent.lower() for keyword in mobile_keywords)
    except:
        return False

# ============================================================================
# CSS PERSONALIZADO COM TEMA PROFISSIONAL
# ============================================================================

st.markdown("""
<style>
    /* Tema dark profissional */
    .stApp {
        background: #0f172a;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Sidebar elegante */
    .css-1d391kg, .css-1wrcr25 {
        background: #020617 !important;
        border-right: 1px solid #334155;
    }
    
    /* Títulos na sidebar */
    .sidebar-title {
        color: #f8fafc !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #3b82f6;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Cards principais com efeito glassmorphism */
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
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .metric-card:hover::before {
        left: 100%;
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
        font-family: 'Inter', sans-serif;
    }
    
    /* Cards para métricas temporais */
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
        font-family: 'Inter', sans-serif;
    }
    
    .time-metric-card .sub-value {
        color: #64748b;
        font-size: 0.8rem;
    }
    
    /* Card de alerta para eventos críticos */
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
        font-family: 'Inter', sans-serif;
    }
    
    .warning-card .sub-label {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    /* Zonas de intensidade */
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
    
    /* Títulos principais */
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
        font-family: 'Inter', sans-serif;
    }
    
    h2 {
        color: white !important;
        font-size: 1.8rem;
        font-weight: 600;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
        margin-bottom: 25px;
        font-family: 'Inter', sans-serif;
    }
    
    h3 {
        color: #3b82f6 !important;
        font-size: 1.4rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    h4 {
        color: #8b5cf6 !important;
        font-size: 1.2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    /* Abas personalizadas */
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
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        color: white !important;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 0 0 12px 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-top: none;
        padding: 15px;
    }
    
    /* Containers de métricas */
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
    
    /* Dataframe */
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
    
    /* Texto geral */
    p, li, .caption, .stMarkdown {
        color: #cbd5e1 !important;
        line-height: 1.6;
    }
    
    /* Info boxes */
    .stInfo, .stWarning, .stError {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-left-color: #3b82f6 !important;
        color: #e2e8f0 !important;
        border-radius: 12px;
        padding: 15px !important;
    }
    
    /* Botões */
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
    
    /* Slider */
    .stSlider label {
        color: #e2e8f0 !important;
    }
    
    /* Selectbox e Multiselect */
    .stSelectbox, .stMultiselect {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    
    .stSelectbox div, .stMultiselect div {
        color: #e2e8f0 !important;
    }
    
    /* Animações */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Scrollbar personalizada */
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
# DETECÇÃO DE DISPOSITIVO E AJUSTE DE LAYOUT
# ============================================================================

mobile = is_mobile()
n_colunas = 1 if mobile else 4

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
if 'ordem_personalizada' not in st.session_state:
    st.session_state.ordem_personalizada = []
if 'upload_files_names' not in st.session_state:
    st.session_state.upload_files_names = []
if 'idioma' not in st.session_state:
    st.session_state.idioma = 'pt'

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def interpretar_teste(p_valor, nome_teste, t):
    """Interpreta resultados do teste de normalidade"""
    if p_valor < 0.0001:
        p_text = f"{p_valor:.2e}"
    else:
        p_text = f"{p_valor:.5f}"
    
    if p_valor > 0.05:
        status = f"✅ {t['normality_test'].split('🧪')[1] if '🧪' in t['normality_test'] else 'Dados normais'}"
        cor = "#10b981"
    else:
        status = f"⚠️ {t['normality_test'].split('🧪')[1] if '🧪' in t['normality_test'] else 'Dados não normais'}"
        cor = "#ef4444"
    
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.8); border-radius: 12px; padding: 20px; border-left: 5px solid {cor}; backdrop-filter: blur(10px);">
        <h4 style="color: white; margin: 0 0 10px 0;">{status}</h4>
        <p style="color: #94a3b8; margin: 5px 0;"><strong>Teste:</strong> {nome_teste}</p>
        <p style="color: #94a3b8; margin: 5px 0;"><strong>p-valor:</strong> <span style="color: {cor};">{p_text}</span></p>
    </div>
    """, unsafe_allow_html=True)

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
    
    for i, df in enumerate(dataframes[1:], 1):
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

def extrair_minuto_do_maximo(df, coluna_valor, coluna_minuto):
    """
    Extrai com segurança o minuto correspondente ao valor máximo
    """
    try:
        if df.empty or len(df) == 0:
            return "N/A"
        
        # Resetar o índice para evitar problemas
        df_reset = df.reset_index(drop=True)
        idx_max = df_reset[coluna_valor].idxmax()
        
        if pd.notna(idx_max) and idx_max < len(df_reset):
            return df_reset.loc[idx_max, coluna_minuto]
        
        return "N/A"
    except:
        try:
            # Método alternativo
            df_sorted = df.sort_values(coluna_valor, ascending=False)
            return df_sorted.iloc[0][coluna_minuto]
        except:
            return "N/A"

def criar_zonas_intensidade(df, variavel, metodo='percentis', limites=None):
    """Cria zonas de intensidade baseadas em diferentes métodos"""
    if metodo == 'percentis':
        return {
            'Muito Baixa': df[variavel].quantile(0.2),
            'Baixa': df[variavel].quantile(0.4),
            'Moderada': df[variavel].quantile(0.6),
            'Alta': df[variavel].quantile(0.8),
            'Muito Alta': df[variavel].quantile(1.0)
        }
    elif metodo == 'maximo':
        max_val = df[variavel].max()
        return {
            'Muito Baixa': max_val * 0.2,
            'Baixa': max_val * 0.4,
            'Moderada': max_val * 0.6,
            'Alta': max_val * 0.8,
            'Muito Alta': max_val
        }
    else:
        return limites

def comparar_grupos(df, variavel, grupo1, grupo2):
    """Compara dois grupos com teste estatístico apropriado"""
    try:
        dados_grupo1 = df[df['Posição'] == grupo1][variavel].dropna()
        dados_grupo2 = df[df['Posição'] == grupo2][variavel].dropna()
        
        if len(dados_grupo1) < 3 or len(dados_grupo2) < 3:
            return None
        
        # Teste de normalidade para cada grupo
        _, p1 = stats.shapiro(dados_grupo1)
        _, p2 = stats.shapiro(dados_grupo2)
        
        if p1 > 0.05 and p2 > 0.05:
            # Dados normais - usar teste t
            t_stat, p_valor = stats.ttest_ind(dados_grupo1, dados_grupo2)
            teste = "Teste t de Student"
        else:
            # Dados não normais - usar Mann-Whitney
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

def analisar_fadiga(df, variavel, baseline_minutos=5):
    """Analisa sinais de fadiga baseado na variável"""
    try:
        df = df.copy().sort_values('Minuto')
        
        # Identificar primeiros minutos como baseline
        minutos_unicos = df['Minuto'].unique()
        if len(minutos_unicos) <= baseline_minutos:
            baseline = df[variavel].mean()
        else:
            baseline_minutos_list = minutos_unicos[:baseline_minutos]
            baseline = df[df['Minuto'].isin(baseline_minutos_list)][variavel].mean()
        
        # Calcular declínio
        df['Declinio'] = (df[variavel] - baseline) / baseline * 100
        
        # Identificar momentos de fadiga (declínio > 15%)
        df['Fadiga'] = df['Declinio'] < -15
        
        # Estatísticas de fadiga
        stats_fadiga = {
            'baseline': baseline,
            'declinio_medio': df['Declinio'].mean(),
            'declinio_maximo': df['Declinio'].min(),
            'minutos_fadiga': df['Fadiga'].sum(),
            'percentual_fadiga': df['Fadiga'].mean() * 100 if len(df) > 0 else 0
        }
        
        return df, stats_fadiga
    except:
        return df, None

def analise_clusters(df, variaveis, n_clusters=3):
    """Realiza análise de clusters nos atletas"""
    try:
        # Preparar dados
        df_cluster = df.groupby('Nome')[variaveis].mean().reset_index()
        X = df_cluster[variaveis].values
        
        if len(X) < n_clusters:
            return None, None, None
        
        # Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_cluster['Cluster'] = clusters
        
        # PCA para visualização
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        df_cluster['PC1'] = X_pca[:, 0]
        df_cluster['PC2'] = X_pca[:, 1]
        
        # Estatísticas por cluster
        stats_cluster = df_cluster.groupby('Cluster')[variaveis].agg(['mean', 'std']).round(3)
        
        return df_cluster, stats_cluster, kmeans
    except:
        return None, None, None

def criar_radar_chart(df, atleta, variaveis, t):
    """Cria gráfico de radar para um atleta específico"""
    try:
        dados_atleta = df[df['Nome'] == atleta][variaveis].mean()
        
        # Normalizar dados (0-100%)
        min_vals = df[variaveis].min()
        max_vals = df[variaveis].max()
        dados_norm = 100 * (dados_atleta - min_vals) / (max_vals - min_vals)
        
        # Média do time
        media_time = df[variaveis].mean()
        media_norm = 100 * (media_time - min_vals) / (max_vals - min_vals)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=dados_norm.values,
            theta=variaveis,
            fill='toself',
            name=atleta,
            line_color='#3b82f6',
            fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=media_norm.values,
            theta=variaveis,
            fill='toself',
            name=t['team_average'],
            line_color='#ef4444',
            fillcolor='rgba(239, 68, 68, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='#334155'
                ),
                bgcolor='rgba(30, 41, 59, 0.8)'
            ),
            showlegend=True,
            title=f"{t['radar_chart']} - {atleta}",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=11)
        )
        
        return fig
    except:
        return None

def plot_progressao_atleta(df, atleta, variavel, t):
    """Plota a progressão de um atleta específico"""
    try:
        df_atleta = df[df['Nome'] == atleta].sort_values('Minuto')
        
        if len(df_atleta) < 3:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{t['individual_progression']} - {variavel}", t['progression_rate']),
            vertical_spacing=0.15
        )
        
        # Gráfico principal
        fig.add_trace(
            go.Scatter(
                x=df_atleta['Minuto'], 
                y=df_atleta[variavel],
                mode='lines+markers', 
                name=variavel,
                line=dict(color='#3b82f6', width=3),
                marker=dict(color='#3b82f6', size=8)
            ),
            row=1, col=1
        )
        
        # Linha de tendência
        z = np.polyfit(range(len(df_atleta)), df_atleta[variavel], 1)
        tendencia = np.poly1d(z)(range(len(df_atleta)))
        fig.add_trace(
            go.Scatter(
                x=df_atleta['Minuto'], 
                y=tendencia,
                mode='lines', 
                name=t['trend_line'],
                line=dict(color='#ef4444', dash='dash', width=2)
            ),
            row=1, col=1
        )
        
        # Taxa de variação
        variacao = df_atleta[variavel].diff()
        cores_variacao = ['#10b981' if v > 0 else '#ef4444' for v in variacao[1:]]
        fig.add_trace(
            go.Bar(
                x=df_atleta['Minuto'][1:], 
                y=variacao[1:],
                name='Variação',
                marker_color=cores_variacao,
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 41, 59, 0.8)',
            font=dict(color='white', size=11)
        )
        
        fig.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'), tickangle=-45)
        fig.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
        
        return fig
    except:
        return None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Seletor de idioma
    st.markdown("<h2 class='sidebar-title'>🌐 Idioma / Language</h2>", unsafe_allow_html=True)
    idioma = st.selectbox("", ['pt', 'en', 'es'], index=['pt', 'en', 'es'].index(st.session_state.idioma), label_visibility="collapsed")
    st.session_state.idioma = idioma
    t = translations[idioma]
    
    st.markdown("---")
    st.markdown(f"<h2 class='sidebar-title'>📂 {t['upload']}</h2>", unsafe_allow_html=True)
    
    upload_files = st.file_uploader(
        "",
        type=['csv'],
        accept_multiple_files=True,
        help=t['tip_text']
    )
    
    if upload_files:
        with st.spinner('🔄 Processando...'):
            time.sleep(0.5)
            try:
                dataframes = []
                arquivos_validos = []
                arquivos_invalidos = []
                
                for uploaded_file in upload_files:
                    try:
                        data = pd.read_csv(uploaded_file)
                        
                        if data.shape[1] >= 3 and not data.empty:
                            dataframes.append(data)
                            arquivos_validos.append(uploaded_file.name)
                        else:
                            arquivos_invalidos.append(f"{uploaded_file.name}")
                    except:
                        arquivos_invalidos.append(f"{uploaded_file.name}")
                
                if dataframes:
                    estruturas_ok, estrutura_referencia = verificar_estruturas_arquivos(dataframes)
                    
                    if not estruturas_ok:
                        st.error("❌ " + ("Arquivos com estruturas diferentes" if idioma == 'pt' else 
                                        "Files with different structures" if idioma == 'en' else 
                                        "Archivos con estructuras diferentes"))
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
                                st.session_state.ordem_personalizada = periodos_unicos.copy()
                                st.session_state.upload_files_names = arquivos_validos
                                
                                if variaveis_quant and st.session_state.variavel_selecionada is None:
                                    st.session_state.variavel_selecionada = variaveis_quant[0]
                                
                                st.success(f"✅ {len(arquivos_validos)} " + 
                                         ("arquivo(s) carregado(s)" if idioma == 'pt' else
                                          "file(s) loaded" if idioma == 'en' else
                                          "archivo(s) cargado(s)"))
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
    
    if st.session_state.df_completo is not None:
        st.markdown("---")
        
        # Seleção de Variável
        if st.session_state.variaveis_quantitativas:
            st.markdown(f"<h2 class='sidebar-title'>📈 {t['variable']}</h2>", unsafe_allow_html=True)
            
            current_index = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                current_index = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            variavel_selecionada = st.selectbox(
                "",
                options=st.session_state.variaveis_quantitativas,
                index=current_index,
                label_visibility="collapsed"
            )
            st.session_state.variavel_selecionada = variavel_selecionada
            
            df_temp = st.session_state.df_completo[variavel_selecionada].dropna()
            if not df_temp.empty:
                st.caption(f"📊 {len(df_temp)} obs | {t['mean']}: {df_temp.mean():.2f}")
        
        # Filtro por Posição
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown(f"<h2 class='sidebar-title'>📍 {t['position']}</h2>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox(f"Selecionar todas as {t['position'].lower()}s" if idioma == 'pt' 
                                          else f"Select all {t['position'].lower()}s" if idioma == 'en'
                                          else f"Seleccionar todas las {t['position'].lower()}s", value=True)
            if selecionar_todos:
                st.session_state.posicoes_selecionadas = st.session_state.todos_posicoes.copy()
            else:
                st.session_state.posicoes_selecionadas = st.multiselect(
                    "",
                    options=st.session_state.todos_posicoes,
                    default=st.session_state.posicoes_selecionadas,
                    label_visibility="collapsed"
                )
        
        # Filtro por Período
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown(f"<h2 class='sidebar-title'>📅 {t['period']}</h2>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox(f"Selecionar todos os {t['period'].lower()}s" if idioma == 'pt'
                                          else f"Select all {t['period'].lower()}s" if idioma == 'en'
                                          else f"Seleccionar todos los {t['period'].lower()}s", value=True)
            if selecionar_todos:
                st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                st.session_state.ordem_personalizada = st.session_state.todos_periodos.copy()
            else:
                st.session_state.periodos_selecionados = st.multiselect(
                    "",
                    options=st.session_state.todos_periodos,
                    default=st.session_state.periodos_selecionados,
                    label_visibility="collapsed"
                )
        
        # Filtro por Atleta
        if st.session_state.atletas_selecionados:
            st.markdown("---")
            st.markdown(f"<h2 class='sidebar-title'>👤 {t['athlete']}</h2>", unsafe_allow_html=True)
            
            df_temp = st.session_state.df_completo.copy()
            if st.session_state.posicoes_selecionadas:
                df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
            if st.session_state.periodos_selecionados:
                df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
            
            atletas_disponiveis = sorted(df_temp['Nome'].unique())
            
            selecionar_todos = st.checkbox(f"Selecionar todos os {t['athlete'].lower()}s" if idioma == 'pt'
                                          else f"Select all {t['athlete'].lower()}s" if idioma == 'en'
                                          else f"Seleccionar todos los {t['athlete'].lower()}s", value=True)
            if selecionar_todos:
                st.session_state.atletas_selecionados = atletas_disponiveis
            else:
                st.session_state.atletas_selecionados = st.multiselect(
                    "",
                    options=atletas_disponiveis,
                    default=[a for a in st.session_state.atletas_selecionados if a in atletas_disponiveis],
                    label_visibility="collapsed"
                )
        
        # Configurações
        st.markdown("---")
        st.markdown(f"<h2 class='sidebar-title'>⚙️ {t['config']}</h2>", unsafe_allow_html=True)
        
        n_classes = st.slider(t['config'].replace('⚙️ ', '') + ":", 3, 20, 5)
        
        # Botão Processar
        st.markdown("---")
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button(t['process'], use_container_width=True, disabled=not pode_processar):
            st.session_state.process_button = True
            st.rerun()

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.get('process_button', False) and st.session_state.df_completo is not None:
    
    with st.spinner('🔄 ' + ("Gerando análises..." if idioma == 'pt' else "Generating analysis..." if idioma == 'en' else "Generando análisis...")):
        time.sleep(0.5)
        
        df_completo = st.session_state.df_completo
        atletas_selecionados = st.session_state.atletas_selecionados
        posicoes_selecionadas = st.session_state.posicoes_selecionadas
        periodos_selecionados = st.session_state.periodos_selecionados
        variavel_analise = st.session_state.variavel_selecionada
        
        df_filtrado = df_completo[
            df_completo['Nome'].isin(atletas_selecionados) & 
            df_completo['Posição'].isin(posicoes_selecionadas) &
            df_completo['Período'].isin(periodos_selecionados)
        ].copy()
        
        df_filtrado = df_filtrado.dropna(subset=[variavel_analise])
        
        if df_filtrado.empty:
            st.warning("⚠️ " + ("Nenhum dado encontrado" if idioma == 'pt' else "No data found" if idioma == 'en' else "No se encontraron datos"))
        else:
            # Métricas principais
            st.markdown(f"<h2>📊 {t['title'].split('Pro')[0] if 'Pro' in t['title'] else 'Visão Geral'}</h2>", unsafe_allow_html=True)
            
            cols = st.columns(n_colunas)
            with cols[0]:
                metric_card(t['positions'], len(posicoes_selecionadas), "📍", "linear-gradient(135deg, #3b82f6, #2563eb)")
            with cols[1 % n_colunas]:
                metric_card(t['periods'], len(periodos_selecionados), "📅", "linear-gradient(135deg, #8b5cf6, #7c3aed)")
            with cols[2 % n_colunas]:
                metric_card(t['athletes'], len(atletas_selecionados), "👥", "linear-gradient(135deg, #ef4444, #dc2626)")
            with cols[3 % n_colunas]:
                metric_card(t['observations'], len(df_filtrado), "📊", "linear-gradient(135deg, #f59e0b, #d97706)")
            
            st.markdown("---")
            
            # Organizar em abas
            tab_titles = [
                t['tab_distribution'], t['tab_temporal'], t['tab_boxplots'], 
                t['tab_correlation'], t['tab_comparison'], t['tab_clusters'],
                t['tab_fatigue'], t['tab_progression'], t['tab_radar'], t['tab_report']
            ]
            
            tabs = st.tabs(tab_titles)
            
            # Aba 1: Distribuição
            with tabs[0]:
                st.markdown(f"<h3>{t['tab_distribution']}</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma
                    dados_hist = df_filtrado[variavel_analise].dropna()
                    
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=dados_hist,
                        nbinsx=n_classes,
                        name='Frequência',
                        marker_color='#3b82f6',
                        opacity=0.8,
                        hovertemplate='Faixa: %{x}<br>Frequência: %{y}<extra></extra>'
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
                    # QQ Plot
                    dados_qq = df_filtrado[variavel_analise].dropna()
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
                        marker=dict(color='#3b82f6', size=8, opacity=0.7),
                        hovertemplate='Teórico: %{x:.2f}<br>Observado: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=linha_ref(quantis_teoricos),
                        mode='lines',
                        name=f'Referência (R² = {r2:.3f})',
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
                
                # Tabela de Frequência
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
            
            # Aba 2: Estatísticas & Temporal
            with tabs[1]:
                st.markdown(f"<h3>{t['tab_temporal']}</h3>", unsafe_allow_html=True)
                
                # Calcular métricas temporais
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                valor_maximo = df_tempo[variavel_analise].max()
                minuto_maximo = extrair_minuto_do_maximo(df_tempo, variavel_analise, 'Minuto')
                media_tempo = df_tempo[variavel_analise].mean()
                limiar_80 = valor_maximo * 0.8
                
                # Contar eventos acima do limiar
                eventos_acima_80 = (df_tempo[variavel_analise] > limiar_80).sum()
                percentual_acima_80 = (eventos_acima_80 / len(df_tempo)) * 100 if len(df_tempo) > 0 else 0
                
                # Cards temporais
                cols_t = st.columns(4)
                with cols_t[0]:
                    time_metric_card(t['max_value'], f"{valor_maximo:.2f}", f"{t['minute_of_max']}: {minuto_maximo}", "#ef4444")
                with cols_t[1]:
                    time_metric_card(t['mean'], f"{media_tempo:.2f}", t['mean'], "#3b82f6")
                with cols_t[2]:
                    time_metric_card(t['threshold_80'], f"{limiar_80:.2f}", f"80% do máx ({valor_maximo:.2f})", "#f59e0b")
                with cols_t[3]:
                    warning_card(t['critical_events'], f"{eventos_acima_80}", f"{percentual_acima_80:.1f}% {t['above_threshold']}", "⚠️")
                
                # Zonas de intensidade
                st.markdown("---")
                st.markdown(f"<h4>{t['intensity_zones']}</h4>", unsafe_allow_html=True)
                
                col_z1, col_z2 = st.columns([1, 3])
                
                with col_z1:
                    metodo_zona = st.radio(
                        t['zone_method'],
                        [t['percentiles'], t['manual'], t['based_on_max']],
                        index=0
                    )
                
                with col_z2:
                    if metodo_zona == t['percentiles']:
                        zonas = criar_zonas_intensidade(df_filtrado, variavel_analise, 'percentis')
                    elif metodo_zona == t['based_on_max']:
                        zonas = criar_zonas_intensidade(df_filtrado, variavel_analise, 'maximo')
                    else:
                        st.info("Modo manual - defina os limites:")
                        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                        with col_l1:
                            l1 = st.number_input(t['very_low'], value=0.0, step=0.1)
                        with col_l2:
                            l2 = st.number_input(t['low'], value=25.0, step=0.1)
                        with col_l3:
                            l3 = st.number_input(t['moderate'], value=50.0, step=0.1)
                        with col_l4:
                            l4 = st.number_input(t['high'], value=75.0, step=0.1)
                        zonas = {
                            t['very_low']: l1,
                            t['low']: l2,
                            t['moderate']: l3,
                            t['high']: l4,
                            t['very_high']: df_filtrado[variavel_analise].max()
                        }
                
                # Visualizar zonas
                if zonas:
                    cores_zonas = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981']
                    st.markdown("##### Limiares das Zonas:")
                    cols_zone = st.columns(5)
                    for i, (zona, limite) in enumerate(zonas.items()):
                        with cols_zone[i]:
                            count = (df_filtrado[variavel_analise] > (list(zonas.values())[i-1] if i > 0 else -np.inf)) & \
                                    (df_filtrado[variavel_analise] <= limite)
                            n_obs = count.sum()
                            st.markdown(f"""
                            <div class="zone-card" style="border-left-color: {cores_zonas[i]};">
                                <div class="zone-name">{zona}</div>
                                <div class="zone-value">{limite:.1f}</div>
                                <div class="zone-count">{n_obs} obs ({n_obs/len(df_filtrado)*100:.0f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Gráfico temporal com suavização
                st.markdown("---")
                st.markdown(f"<h4>{t['tab_temporal']}</h4>", unsafe_allow_html=True)
                
                fig_tempo = go.Figure()
                
                # Dados originais
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
                
                # Linha suavizada
                if len(df_tempo) > 5:
                    y_suave = savgol_filter(df_tempo[variavel_analise], min(5, len(df_tempo)-1 if len(df_tempo)%2==0 else len(df_tempo)), 2)
                    fig_tempo.add_trace(go.Scatter(
                        x=df_tempo['Minuto'],
                        y=y_suave,
                        mode='lines',
                        name='Tendência (suavizada)',
                        line=dict(color='#f59e0b', width=3)
                    ))
                
                # Linhas de referência
                fig_tempo.add_hline(y=media_tempo, line_dash="dash", line_color="#3b82f6", line_width=2,
                                   annotation_text=f"{t['mean']}: {media_tempo:.2f}", annotation_position="left")
                fig_tempo.add_hline(y=limiar_80, line_dash="dot", line_color="#f59e0b", line_width=2,
                                   annotation_text=f"{t['threshold_80']}: {limiar_80:.2f}", annotation_position="right")
                fig_tempo.add_hline(y=valor_maximo, line_dash="solid", line_color="#ef4444", line_width=2,
                                   annotation_text=f"{t['max_value']}: {valor_maximo:.2f}", annotation_position="right")
                
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
                
                # Estatísticas descritivas
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
                        <p><small>{t['iqr_explanation']}</small></p>
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
                
                # Intervalo de Confiança
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
                
                # Teste de Normalidade
                st.markdown("---")
                st.markdown(f"<h4>{t['normality_test']}</h4>", unsafe_allow_html=True)
                
                dados_teste = df_filtrado[variavel_analise].dropna()
                n_teste = len(dados_teste)
                
                if n_teste < 3:
                    st.error("❌ " + ("Amostra muito pequena (n < 3)" if idioma == 'pt' else "Sample too small (n < 3)" if idioma == 'en' else "Muestra muy pequeña (n < 3)"))
                elif n_teste > 5000:
                    st.info("ℹ️ " + ("Amostra grande demais. Usando D'Agostino-Pearson." if idioma == 'pt' else "Sample too large. Using D'Agostino-Pearson." if idioma == 'en' else "Muestra demasiado grande. Usando D'Agostino-Pearson."))
                    try:
                        k2, p = stats.normaltest(dados_teste)
                        interpretar_teste(p, "D'Agostino-Pearson", t)
                    except:
                        st.warning("⚠️ " + ("Teste alternativo não disponível" if idioma == 'pt' else "Alternative test not available" if idioma == 'en' else "Prueba alternativa no disponible"))
                else:
                    try:
                        shapiro = stats.shapiro(dados_teste)
                        interpretar_teste(shapiro.pvalue, "Shapiro-Wilk", t)
                    except:
                        st.error("❌ " + ("Erro no teste" if idioma == 'pt' else "Test error" if idioma == 'en' else "Error en la prueba"))
                
                # Resumo por grupos
                st.markdown("---")
                st.markdown(f"<h4>{t['summary_by_group']}</h4>", unsafe_allow_html=True)
                
                resumo = []
                for nome in atletas_selecionados[:10]:
                    for posicao in posicoes_selecionadas[:5]:
                        for periodo in periodos_selecionados[:5]:
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
                                minuto_max_grupo = extrair_minuto_do_maximo(dados, variavel_analise, 'Minuto')
                                
                                resumo.append({
                                    'Atleta': nome,
                                    'Posição': posicao,
                                    'Período': periodo,
                                    f'Máx {variavel_analise}': valor_max_grupo,
                                    'Minuto do Máx': minuto_max_grupo,
                                    f'Mín {variavel_analise}': dados[variavel_analise].min(),
                                    'Amplitude': valor_max_grupo - dados[variavel_analise].min(),
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
            
            # Aba 3: Boxplots
            with tabs[2]:
                st.markdown(f"<h3>{t['tab_boxplots']}</h3>", unsafe_allow_html=True)
                
                # Boxplot por posição
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
                
                # Boxplot por atleta
                st.markdown(f"<h4>👥 {t['athlete']}</h4>", unsafe_allow_html=True)
                
                atletas_plot = atletas_selecionados[:10]
                if len(atletas_selecionados) > 10:
                    st.info(f"ℹ️ " + (f"Mostrando 10 de {len(atletas_selecionados)} atletas" if idioma == 'pt' 
                                    else f"Showing 10 of {len(atletas_selecionados)} athletes" if idioma == 'en'
                                    else f"Mostrando 10 de {len(atletas_selecionados)} atletas"))
                
                fig_box_atl = go.Figure()
                for atleta in atletas_plot:
                    dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                    if len(dados_atl) > 0:
                        fig_box_atl.add_trace(go.Box(
                            y=dados_atl,
                            name=atleta[:15] + "..." if len(atleta) > 15 else atleta,
                            boxmean='sd',
                            marker_color='#8b5cf6',
                            line_color='white',
                            fillcolor='rgba(139, 92, 246, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_atl.update_layout(
                    title=f"{t['athlete']} - {variavel_analise}",
                    plot_bgcolor='rgba(30, 41, 59, 0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3b82f6', size=16),
                    yaxis_title=variavel_analise,
                    showlegend=False,
                    height=400
                )
                fig_box_atl.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'), tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                st.plotly_chart(fig_box_atl, use_container_width=True)
                
                # Estatísticas por atleta
                with st.expander(f"📊 {t['descriptive_stats']} {t['athlete'].lower()}"):
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.8); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
                        <h5 style="color: #3b82f6;">{t['iqr_title']}</h5>
                        <p style="color: #94a3b8;">{t['iqr_explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_atletas = []
                    for atleta in atletas_selecionados[:20]:
                        dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                        if len(dados_atl) > 0:
                            q1_atl = dados_atl.quantile(0.25)
                            q3_atl = dados_atl.quantile(0.75)
                            iqr_atl = q3_atl - q1_atl
                            media_atl = dados_atl.mean()
                            desvio_atl = dados_atl.std()
                            cv_atl = calcular_cv(media_atl, desvio_atl)
                            valor_max_atl = dados_atl.max()
                            minuto_max_atl = extrair_minuto_do_maximo(dados_atl, variavel_analise, 'Minuto')
                            
                            stats_atletas.append({
                                'Atleta': atleta,
                                'Média': media_atl,
                                'Mediana': dados_atl.median(),
                                'DP': desvio_atl,
                                'CV (%)': cv_atl,
                                'Mín': dados_atl.min(),
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
            
            # Aba 4: Correlações
            with tabs[3]:
                st.markdown(f"<h3>{t['tab_correlation']}</h3>", unsafe_allow_html=True)
                
                if len(st.session_state.variaveis_quantitativas) > 1:
                    vars_corr = st.multiselect(
                        t['tab_correlation'].replace('🔥', '').strip(),
                        options=st.session_state.variaveis_quantitativas,
                        default=st.session_state.variaveis_quantitativas[:min(5, len(st.session_state.variaveis_quantitativas))]
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
                        st.info("ℹ️ " + ("Selecione pelo menos 2 variáveis" if idioma == 'pt' else "Select at least 2 variables" if idioma == 'en' else "Seleccione al menos 2 variables"))
                else:
                    st.info("ℹ️ " + ("São necessárias pelo menos 2 variáveis" if idioma == 'pt' else "At least 2 variables are needed" if idioma == 'en' else "Se necesitan al menos 2 variables"))
            
            # Aba 5: Comparações
            with tabs[4]:
                st.markdown(f"<h3>{t['tab_comparison']}</h3>", unsafe_allow_html=True)
                
                if len(posicoes_selecionadas) >= 2:
                    st.markdown(f"<h4>{t['position']}</h4>", unsafe_allow_html=True)
                    
                    col_comp1, col_comp2 = st.columns(2)
                    
                    with col_comp1:
                        grupo1 = st.selectbox(f"{t['position']} 1:", posicoes_selecionadas, index=0)
                    with col_comp2:
                        grupo2 = st.selectbox(f"{t['position']} 2:", posicoes_selecionadas, index=min(1, len(posicoes_selecionadas)-1))
                    
                    if grupo1 != grupo2:
                        resultado = comparar_grupos(df_filtrado, variavel_analise, grupo1, grupo2)
                        
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
                            
                            # Boxplot comparativo
                            fig_comp = go.Figure()
                            
                            dados_comp1 = df_filtrado[df_filtrado['Posição'] == grupo1][variavel_analise]
                            dados_comp2 = df_filtrado[df_filtrado['Posição'] == grupo2][variavel_analise]
                            
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
                                title=f"{grupo1} vs {grupo2} - {variavel_analise}",
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
                            st.warning("⚠️ " + ("Dados insuficientes para comparação" if idioma == 'pt' else "Insufficient data for comparison" if idioma == 'en' else "Datos insuficientes para comparación"))
                    else:
                        st.info("ℹ️ " + ("Selecione grupos diferentes" if idioma == 'pt' else "Select different groups" if idioma == 'en' else "Seleccione grupos diferentes"))
                else:
                    st.info("ℹ️ " + ("Selecione pelo menos 2 posições" if idioma == 'pt' else "Select at least 2 positions" if idioma == 'en' else "Seleccione al menos 2 posiciones"))
            
            # Aba 6: Clusters
            with tabs[5]:
                st.markdown(f"<h3>{t['tab_clusters']}</h3>", unsafe_allow_html=True)
                
                vars_cluster = st.multiselect(
                    t['cluster_vars'],
                    options=st.session_state.variaveis_quantitativas,
                    default=st.session_state.variaveis_quantitativas[:min(3, len(st.session_state.variaveis_quantitativas))]
                )
                
                if len(vars_cluster) >= 2:
                    n_clusters = st.slider(t['n_clusters'], 2, 5, 3)
                    
                    df_cluster, stats_cluster, _ = analise_clusters(df_filtrado, vars_cluster, n_clusters)
                    
                    if df_cluster is not None:
                        fig_cluster = px.scatter(
                            df_cluster,
                            x='PC1',
                            y='PC2',
                            color='Cluster',
                            hover_data=['Nome'] + vars_cluster,
                            title=t['cluster_plot'],
                            color_continuous_scale=px.colors.qualitative.Set1
                        )
                        fig_cluster.update_layout(
                            plot_bgcolor='rgba(30, 41, 59, 0.8)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', size=11),
                            title_font=dict(color='#3b82f6', size=16),
                            height=500
                        )
                        fig_cluster.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                        fig_cluster.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                        st.plotly_chart(fig_cluster, use_container_width=True)
                        
                        st.markdown(f"<h4>{t['cluster_stats']}</h4>", unsafe_allow_html=True)
                        st.dataframe(stats_cluster, use_container_width=True)
                        
                        # Distribuição dos clusters
                        st.markdown(f"<h4>{t['position']}</h4>", unsafe_allow_html=True)
                        
                        df_cluster_pos = df_cluster.merge(
                            df_filtrado[['Nome', 'Posição']].drop_duplicates(),
                            on='Nome',
                            how='left'
                        )
                        
                        cluster_pos = pd.crosstab(df_cluster_pos['Cluster'], df_cluster_pos['Posição'])
                        st.dataframe(cluster_pos, use_container_width=True)
                    else:
                        st.warning("⚠️ " + ("Não foi possível criar clusters" if idioma == 'pt' else "Could not create clusters" if idioma == 'en' else "No se pudieron crear clústers"))
                else:
                    st.info("ℹ️ " + ("Selecione pelo menos 2 variáveis" if idioma == 'pt' else "Select at least 2 variables" if idioma == 'en' else "Seleccione al menos 2 variables"))
            
            # Aba 7: Análise de Fadiga
            with tabs[6]:
                st.markdown(f"<h3>{t['tab_fatigue']}</h3>", unsafe_allow_html=True)
                
                baseline_min = st.slider(f"{t['baseline']} (minutos iniciais):", 1, 10, 5)
                
                df_fadiga, stats_fadiga = analisar_fadiga(df_filtrado, variavel_analise, baseline_min)
                
                if stats_fadiga is not None:
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    
                    with col_f1:
                        time_metric_card(t['baseline'], f"{stats_fadiga['baseline']:.2f}", "", "#3b82f6")
                    with col_f2:
                        time_metric_card(t['avg_decline'], f"{stats_fadiga['declinio_medio']:.1f}%", "", "#f59e0b")
                    with col_f3:
                        time_metric_card(t['max_decline'], f"{stats_fadiga['declinio_maximo']:.1f}%", "", "#ef4444")
                    with col_f4:
                        warning_card(t['fatigue_minutes'], f"{stats_fadiga['minutos_fadiga']:.0f}", 
                                   f"{stats_fadiga['percentual_fadiga']:.1f}% {t['fatigue_percentage'].lower()}", "⚡")
                    
                    # Gráfico de fadiga
                    fig_fadiga = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f"{variavel_analise} com {t['baseline']}", t['fatigue_analysis']),
                        vertical_spacing=0.15
                    )
                    
                    fig_fadiga.add_trace(
                        go.Scatter(
                            x=df_fadiga['Minuto'],
                            y=df_fadiga[variavel_analise],
                            mode='lines+markers',
                            name=variavel_analise,
                            line=dict(color='#3b82f6', width=2),
                            marker=dict(color=['#ef4444' if f else '#3b82f6' for f in df_fadiga['Fadiga']], size=8)
                        ),
                        row=1, col=1
                    )
                    fig_fadiga.add_hline(y=stats_fadiga['baseline'], line_dash="dash", 
                                        line_color="#3b82f6", annotation_text=t['baseline'], row=1, col=1)
                    
                    fig_fadiga.add_trace(
                        go.Bar(
                            x=df_fadiga['Minuto'],
                            y=df_fadiga['Declinio'],
                            name='Declínio %',
                            marker_color=['#ef4444' if f else '#3b82f6' for f in df_fadiga['Fadiga']]
                        ),
                        row=2, col=1
                    )
                    fig_fadiga.add_hline(y=-15, line_dash="dot", line_color="#ef4444",
                                        annotation_text=t['fatigue_threshold'], row=2, col=1)
                    
                    fig_fadiga.update_layout(
                        height=600,
                        plot_bgcolor='rgba(30, 41, 59, 0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3b82f6', size=16),
                        showlegend=False
                    )
                    fig_fadiga.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'), tickangle=-45)
                    fig_fadiga.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    
                    st.plotly_chart(fig_fadiga, use_container_width=True)
                    
                    # Tabela de períodos de fadiga
                    df_fadiga_periodos = df_fadiga[df_fadiga['Fadiga']][['Minuto', variavel_analise, 'Declinio']]
                    if not df_fadiga_periodos.empty:
                        st.markdown(f"<h4>{t['fatigue_periods']}</h4>", unsafe_allow_html=True)
                        st.dataframe(df_fadiga_periodos, use_container_width=True)
                else:
                    st.warning("⚠️ " + ("Não foi possível analisar fadiga" if idioma == 'pt' else "Could not analyze fatigue" if idioma == 'en' else "No se pudo analizar fatiga"))
            
            # Aba 8: Progressão Individual
            with tabs[7]:
                st.markdown(f"<h3>{t['tab_progression']}</h3>", unsafe_allow_html=True)
                
                atleta_prog = st.selectbox(t['select_athlete'], atletas_selecionados)
                
                fig_prog = plot_progressao_atleta(df_filtrado, atleta_prog, variavel_analise, t)
                
                if fig_prog is not None:
                    st.plotly_chart(fig_prog, use_container_width=True)
                    
                    # Estatísticas do atleta
                    dados_atleta_prog = df_filtrado[df_filtrado['Nome'] == atleta_prog][variavel_analise]
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📊 {atleta_prog}</h4>
                        <hr style="border-color: #334155;">
                        <p><strong>{t['mean']}:</strong> {dados_atleta_prog.mean():.2f}</p>
                        <p><strong>{t['max']}:</strong> {dados_atleta_prog.max():.2f}</p>
                        <p><strong>{t['min']}:</strong> {dados_atleta_prog.min():.2f}</p>
                        <p><strong>{t['std']}:</strong> {dados_atleta_prog.std():.2f}</p>
                        <p><strong>{t['cv']}:</strong> {calcular_cv(dados_atleta_prog.mean(), dados_atleta_prog.std()):.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ " + ("Dados insuficientes para progressão" if idioma == 'pt' else "Insufficient data for progression" if idioma == 'en' else "Datos insuficientes para progresión"))
            
            # Aba 9: Radar Chart
            with tabs[8]:
                st.markdown(f"<h3>{t['tab_radar']}</h3>", unsafe_allow_html=True)
                
                vars_radar = st.multiselect(
                    t['select_vars_radar'],
                    options=st.session_state.variaveis_quantitativas,
                    default=st.session_state.variaveis_quantitativas[:min(4, len(st.session_state.variaveis_quantitativas))]
                )
                
                if len(vars_radar) >= 3:
                    atleta_radar = st.selectbox(t['select_athlete'], atletas_selecionados, key="radar_athlete")
                    
                    fig_radar = criar_radar_chart(df_filtrado, atleta_radar, vars_radar, t)
                    
                    if fig_radar is not None:
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Tabela comparativa
                        st.markdown(f"<h4>📊 {t['tab_radar']}</h4>", unsafe_allow_html=True)
                        
                        dados_radar = df_filtrado.groupby('Nome')[vars_radar].mean().round(2)
                        st.dataframe(dados_radar.style.highlight_max(color='#10b981').highlight_min(color='#ef4444'), use_container_width=True)
                    else:
                        st.warning("⚠️ " + ("Não foi possível criar radar chart" if idioma == 'pt' else "Could not create radar chart" if idioma == 'en' else "No se pudo crear gráfico radar"))
                else:
                    st.info("ℹ️ " + ("Selecione pelo menos 3 variáveis" if idioma == 'pt' else "Select at least 3 variables" if idioma == 'en' else "Seleccione al menos 3 variables"))
            
            # Aba 10: Relatório
            with tabs[9]:
                st.markdown(f"<h3>{t['tab_report']}</h3>", unsafe_allow_html=True)
                
                if st.button(t['generate_report'], use_container_width=True):
                    with st.spinner("📄 " + ("Gerando relatório..." if idioma == 'pt' else "Generating report..." if idioma == 'en' else "Generando informe...")):
                        time.sleep(2)
                        
                        # Criar HTML do relatório
                        html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>{t['report_title']}</title>
                            <style>
                                body {{ font-family: 'Arial', sans-serif; background: #0f172a; color: #e2e8f0; padding: 30px; }}
                                h1 {{ color: #3b82f6; text-align: center; }}
                                h2 {{ color: #8b5cf6; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
                                .metric {{ background: #1e293b; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #3b82f6; }}
                                .critical {{ border-left-color: #ef4444; }}
                                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                                th {{ background: #3b82f6; color: white; padding: 10px; }}
                                td {{ background: #1e293b; padding: 8px; border: 1px solid #334155; }}
                                .footer {{ text-align: center; margin-top: 30px; color: #94a3b8; }}
                            </style>
                        </head>
                        <body>
                            <h1>{t['report_title']}</h1>
                            <p style="text-align: center;">{t['report_date']}: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                            
                            <h2>📊 {t['title'].split('Pro')[0]}</h2>
                            <div class="metric">
                                <p><strong>{t['variable']}:</strong> {variavel_analise}</p>
                                <p><strong>{t['positions']}:</strong> {len(posicoes_selecionadas)}</p>
                                <p><strong>{t['periods']}:</strong> {len(periodos_selecionados)}</p>
                                <p><strong>{t['athletes']}:</strong> {len(atletas_selecionados)}</p>
                                <p><strong>{t['observations']}:</strong> {len(df_filtrado)}</p>
                            </div>
                            
                            <h2>📈 {t['descriptive_stats']}</h2>
                            <div class="metric">
                                <p><strong>{t['mean']}:</strong> {media:.3f}</p>
                                <p><strong>{t['median']}:</strong> {mediana:.3f}</p>
                                <p><strong>{t['std']}:</strong> {desvio:.3f}</p>
                                <p><strong>{t['cv']}:</strong> {cv:.1f}%</p>
                                <p><strong>{t['min']}:</strong> {df_filtrado[variavel_analise].min():.3f}</p>
                                <p><strong>{t['max']}:</strong> {df_filtrado[variavel_analise].max():.3f}</p>
                            </div>
                            
                            <h2>🎯 {t['confidence_interval']}</h2>
                            <div class="metric">
                                <p><strong>{t['mean']}:</strong> {media:.3f}</p>
                                <p><strong>IC 95%:</strong> [{ic_inf:.3f}, {ic_sup:.3f}]</p>
                            </div>
                            
                            <h2>⚡ {t['tab_fatigue']}</h2>
                            <div class="metric critical">
                                <p><strong>{t['critical_events']}:</strong> {eventos_acima_80}</p>
                                <p><strong>{t['fatigue_percentage']}:</strong> {percentual_acima_80:.1f}%</p>
                                <p><strong>{t['max_value']}:</strong> {valor_maximo:.2f} ({t['minute_of_max']}: {minuto_maximo})</p>
                            </div>
                            
                            <h2>📋 {t['summary_by_group']}</h2>
                            <table>
                                <tr>
                                    <th>{t['position']}</th>
                                    <th>{t['athlete']}</th>
                                    <th>{t['period']}</th>
                                    <th>{t['mean']}</th>
                                    <th>{t['max']}</th>
                                    <th>N</th>
                                </tr>
                        """
                        
                        for item in resumo[:20]:
                            html_content += f"""
                                <tr>
                                    <td>{item['Posição']}</td>
                                    <td>{item['Atleta']}</td>
                                    <td>{item['Período']}</td>
                                    <td>{item['Média']:.2f}</td>
                                    <td>{item[f'Máx {variavel_analise}']:.2f}</td>
                                    <td>{item['Nº Amostras']}</td>
                                </tr>
                            """
                        
                        html_content += f"""
                            </table>
                            
                            <div class="footer">
                                <p>{t['report_title']} - {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                                <p>Sports Science Analytics Pro</p>
                            </div>
                        </body>
                        </html>
                        """
                        
                        # Download do relatório
                        b64 = base64.b64encode(html_content.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="relatorio_{variavel_analise}_{datetime.now().strftime("%Y%m%d_%H%M")}.html">📥 {t["download_report"]}</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("✅ " + ("Relatório gerado com sucesso!" if idioma == 'pt' else "Report generated successfully!" if idioma == 'en' else "Informe generado con éxito!"))
            
            # Dados brutos
            with st.expander("📋 " + ("Visualizar dados brutos filtrados" if idioma == 'pt' else "View filtered raw data" if idioma == 'en' else "Ver datos brutos filtrados")):
                st.dataframe(df_filtrado, use_container_width=True)
    
    # Reset do botão
    st.session_state.process_button = False

else:
    # Tela inicial
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
        
        with st.expander("📋 " + ("Preview dos dados carregados" if idioma == 'pt' else "Preview of loaded data" if idioma == 'en' else "Vista previa de datos cargados")):
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