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
import subprocess
import sys
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Instalar scikit-learn se não estiver disponível
try:
    from sklearn.cluster import KMeans
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Sports Science Analytics Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🏃"
)

# ============================================================================
# CORES ACESSÍVEIS PARA DALTÔNICOS
# ============================================================================
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'skyblue': '#56B4E9',
    'black': '#000000',
    'gray': '#999999',
    'darkblue': '#003366'
}

COR_PRIMARIA = COLORS['blue']
COR_SECUNDARIA = COLORS['orange']
COR_SUCESSO = COLORS['green']
COR_ALERTA = COLORS['red']
COR_DESTAQUE = COLORS['purple']
COR_NEUTRA = COLORS['gray']

# Cores para os temas
THEMES = {
    'dark': {
        'bg_primary': '#0f172a',
        'bg_secondary': '#1e293b',
        'bg_card': 'rgba(30, 41, 59, 0.8)',
        'text_primary': '#ffffff',
        'text_secondary': '#94a3b8',
        'border': '#334155'
    },
    'light': {
        'bg_primary': '#f8fafc',
        'bg_secondary': '#ffffff',
        'bg_card': 'rgba(255, 255, 255, 0.9)',
        'text_primary': '#0f172a',
        'text_secondary': '#475569',
        'border': '#e2e8f0'
    }
}

# ============================================================================
# CORES PARA HEATMAP DE CORRELAÇÃO
# ============================================================================
CORRELATION_COLORS = [
    [0, 'rgb(213, 94, 0)'],
    [0.25, 'rgb(230, 159, 0)'],
    [0.5, 'rgb(255, 255, 255)'],
    [0.75, 'rgb(86, 180, 233)'],
    [1, 'rgb(0, 114, 178)']
]

# ============================================================================
# INTERNACIONALIZAÇÃO - MÚLTIPLOS IDIOMAS
# ============================================================================

translations = {
    'pt_br': {
        'name': 'Português (Brasil)',
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
        'tab_kmeans': '🤖 K-means Clusters',
        'tab_comparador': '🆚 Comparador de Atletas',
        'tab_mbi': '🔬 Análise MBI',
        'tab_executive': '📋 Executivo',
        'tab_individual': '👤 Individual',
        'tab_trends': '📈 Tendências ML',
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
        'threshold_75': 'LIMIAR 75%',
        'threshold_50': 'LIMIAR 50%',
        'high_intensity_events': 'EVENTOS DE ALTA INTENSIDADE',
        'medium_high_intensity_events': 'EVENTOS DE INTENSIDADE MÉDIA-ALTA',
        'above_threshold_75': 'acima do limiar de 75%',
        'between_50_75': 'entre 50% e 75%',
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
        'iqr_explanation': 'O IQR (Intervalo Interquartil) é a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1). Representa a amplitude dos 50% centrais dos dados.',
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
        'multi_file_text': '### Carregando múltiplos arquivos:\n1. Prepare seus arquivos CSV com a **mesma estrutura** de colunas\n2. Selecione todos os arquivos desejados\n3. O sistema verificará compatibilidade e concatenará automaticamente',
        'select_period_timeline': 'Selecione o período para visualização temporal',
        'all_periods': 'Todos os períodos (gráfico único)',
        'compare_periods': 'Comparar períodos (múltiplos gráficos)',
        'export_report': '📄 Exportar Relatório',
        'export_excel': '📊 Exportar Excel',
        'export_pdf': '📑 Exportar PDF',
        'presentation_mode': '🎬 Modo Apresentação',
        'theme': '🎨 Tema',
        'dark_theme': 'Escuro',
        'light_theme': 'Claro',
        'alerts': '⚠️ Alertas',
        'outlier_alert': 'Outlier Detectado',
        'trend_alert': 'Tendência Preocupante',
        'forecast': 'Previsão',
        'confidence_interval_forecast': 'Intervalo de Confiança',
        'athlete_profile': 'Perfil do Atleta',
        'performance_history': 'Histórico de Desempenho',
        'weekly_summary': 'Resumo Semanal',
        'download_excel': 'Baixar Excel',
        'download_pdf': 'Baixar PDF'
    },
    'es_mx': {
        'name': 'Español (México)',
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
        'tab_kmeans': '🤖 Clústeres K-means',
        'tab_comparador': '🆚 Comparador de Atletas',
        'tab_mbi': '🔬 Análisis MBI',
        'tab_executive': '📋 Ejecutivo',
        'tab_individual': '👤 Individual',
        'tab_trends': '📈 Tendencias ML',
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
        'threshold_75': 'UMBRAL 75%',
        'threshold_50': 'UMBRAL 50%',
        'high_intensity_events': 'EVENTOS DE ALTA INTENSIDAD',
        'medium_high_intensity_events': 'EVENTOS DE INTENSIDAD MEDIA-ALTA',
        'above_threshold_75': 'por encima del umbral 75%',
        'between_50_75': 'entre 50% y 75%',
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
        'iqr_explanation': 'IQR (Rango Intercuartil) es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1). Representa la amplitud del 50% central de los datos.',
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
        'multi_file_text': '### Cargando múltiples archivos:\n1. Prepare sus archivos CSV con la **misma estructura** de columnas\n2. Seleccione todos los archivos deseados\n3. El sistema verificará compatibilidad y concatenará automáticamente',
        'select_period_timeline': 'Seleccione el período para visualización temporal',
        'all_periods': 'Todos los períodos (gráfico único)',
        'compare_periods': 'Comparar períodos (múltiples gráficos)',
        'export_report': '📄 Exportar Reporte',
        'export_excel': '📊 Exportar Excel',
        'export_pdf': '📑 Exportar PDF',
        'presentation_mode': '🎬 Modo Presentación',
        'theme': '🎨 Tema',
        'dark_theme': 'Oscuro',
        'light_theme': 'Claro',
        'alerts': '⚠️ Alertas',
        'outlier_alert': 'Outlier Detectado',
        'trend_alert': 'Tendencia Preocupante',
        'forecast': 'Pronóstico',
        'confidence_interval_forecast': 'Intervalo de Confianza',
        'athlete_profile': 'Perfil del Atleta',
        'performance_history': 'Historial de Rendimiento',
        'weekly_summary': 'Resumen Semanal',
        'download_excel': 'Descargar Excel',
        'download_pdf': 'Descargar PDF'
    },
    'es_la': {
        'name': 'Español (Latinoamérica)',
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
        'tab_kmeans': '🤖 Clústeres K-means',
        'tab_comparador': '🆚 Comparador de Atletas',
        'tab_mbi': '🔬 Análisis MBI',
        'tab_executive': '📋 Ejecutivo',
        'tab_individual': '👤 Individual',
        'tab_trends': '📈 Tendencias ML',
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
        'threshold_75': 'UMBRAL 75%',
        'threshold_50': 'UMBRAL 50%',
        'high_intensity_events': 'EVENTOS DE ALTA INTENSIDAD',
        'medium_high_intensity_events': 'EVENTOS DE INTENSIDAD MEDIA-ALTA',
        'above_threshold_75': 'por encima del umbral 75%',
        'between_50_75': 'entre 50% y 75%',
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
        'iqr_explanation': 'IQR (Rango Intercuartil) es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1). Representa la amplitud del 50% central de los datos.',
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
        'multi_file_text': '### Cargando múltiples archivos:\n1. Prepare sus archivos CSV con la **misma estructura** de columnas\n2. Seleccione todos los archivos deseados\n3. El sistema verificará compatibilidad y concatenará automáticamente',
        'select_period_timeline': 'Seleccione el período para visualización temporal',
        'all_periods': 'Todos los períodos (gráfico único)',
        'compare_periods': 'Comparar períodos (múltiples gráficos)',
        'export_report': '📄 Exportar Informe',
        'export_excel': '📊 Exportar Excel',
        'export_pdf': '📑 Exportar PDF',
        'presentation_mode': '🎬 Modo Presentación',
        'theme': '🎨 Tema',
        'dark_theme': 'Oscuro',
        'light_theme': 'Claro',
        'alerts': '⚠️ Alertas',
        'outlier_alert': 'Outlier Detectado',
        'trend_alert': 'Tendencia Preocupante',
        'forecast': 'Pronóstico',
        'confidence_interval_forecast': 'Intervalo de Confianza',
        'athlete_profile': 'Perfil del Atleta',
        'performance_history': 'Historial de Rendimiento',
        'weekly_summary': 'Resumen Semanal',
        'download_excel': 'Descargar Excel',
        'download_pdf': 'Descargar PDF'
    },
    'en_us': {
        'name': 'English (US)',
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
        'tab_kmeans': '🤖 K-means Clusters',
        'tab_comparador': '🆚 Athlete Comparator',
        'tab_mbi': '🔬 MBI Analysis',
        'tab_executive': '📋 Executive',
        'tab_individual': '👤 Individual',
        'tab_trends': '📈 ML Trends',
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
        'threshold_75': '75% THRESHOLD',
        'threshold_50': '50% THRESHOLD',
        'high_intensity_events': 'HIGH INTENSITY EVENTS',
        'medium_high_intensity_events': 'MEDIUM-HIGH INTENSITY EVENTS',
        'above_threshold_75': 'above 75% threshold',
        'between_50_75': 'between 50% and 75%',
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
        'iqr_explanation': 'IQR (Interquartile Range) is the difference between the third quartile (Q3) and the first quartile (Q1). It represents the range of the middle 50% of the data.',
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
        'multi_file_text': '### Loading multiple files:\n1. Prepare your CSV files with the **same column structure**\n2. Select all desired files\n3. The system will check compatibility and concatenate automatically',
        'select_period_timeline': 'Select period for temporal visualization',
        'all_periods': 'All periods (single chart)',
        'compare_periods': 'Compare periods (multiple charts)',
        'export_report': '📄 Export Report',
        'export_excel': '📊 Export Excel',
        'export_pdf': '📑 Export PDF',
        'presentation_mode': '🎬 Presentation Mode',
        'theme': '🎨 Theme',
        'dark_theme': 'Dark',
        'light_theme': 'Light',
        'alerts': '⚠️ Alerts',
        'outlier_alert': 'Outlier Detected',
        'trend_alert': 'Concerning Trend',
        'forecast': 'Forecast',
        'confidence_interval_forecast': 'Confidence Interval',
        'athlete_profile': 'Athlete Profile',
        'performance_history': 'Performance History',
        'weekly_summary': 'Weekly Summary',
        'download_excel': 'Download Excel',
        'download_pdf': 'Download PDF'
    },
    'en_uk': {
        'name': 'English (UK)',
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
        'tab_kmeans': '🤖 K-means Clusters',
        'tab_comparador': '🆚 Athlete Comparator',
        'tab_mbi': '🔬 MBI Analysis',
        'tab_executive': '📋 Executive',
        'tab_individual': '👤 Individual',
        'tab_trends': '📈 ML Trends',
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
        'threshold_75': '75% THRESHOLD',
        'threshold_50': '50% THRESHOLD',
        'high_intensity_events': 'HIGH INTENSITY EVENTS',
        'medium_high_intensity_events': 'MEDIUM-HIGH INTENSITY EVENTS',
        'above_threshold_75': 'above 75% threshold',
        'between_50_75': 'between 50% and 75%',
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
        'iqr_explanation': 'IQR (Interquartile Range) is the difference between the third quartile (Q3) and the first quartile (Q1). It represents the range of the middle 50% of the data.',
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
        'multi_file_text': '### Loading multiple files:\n1. Prepare your CSV files with the **same column structure**\n2. Select all desired files\n3. The system will check compatibility and concatenate automatically',
        'select_period_timeline': 'Select period for temporal visualization',
        'all_periods': 'All periods (single chart)',
        'compare_periods': 'Compare periods (multiple charts)',
        'export_report': '📄 Export Report',
        'export_excel': '📊 Export Excel',
        'export_pdf': '📑 Export PDF',
        'presentation_mode': '🎬 Presentation Mode',
        'theme': '🎨 Theme',
        'dark_theme': 'Dark',
        'light_theme': 'Light',
        'alerts': '⚠️ Alerts',
        'outlier_alert': 'Outlier Detected',
        'trend_alert': 'Concerning Trend',
        'forecast': 'Forecast',
        'confidence_interval_forecast': 'Confidence Interval',
        'athlete_profile': 'Athlete Profile',
        'performance_history': 'Performance History',
        'weekly_summary': 'Weekly Summary',
        'download_excel': 'Download Excel',
        'download_pdf': 'Download PDF'
    },
    'fr': {
        'name': 'Français',
        'title': 'Sports Science Analytics Pro',
        'subtitle': 'Tableau de Bord Professionnel pour l\'Analyse de la Performance Sportive',
        'upload': 'Téléchargement des Données',
        'variable': 'Variable',
        'position': 'Position',
        'period': 'Période',
        'athlete': 'Athlète',
        'config': 'Configuration',
        'tab_distribution': '📊 Distribution',
        'tab_temporal': '📈 Statistiques & Temporel',
        'tab_boxplots': '📦 Boîtes à Moustaches',
        'tab_correlation': '🔥 Corrélations',
        'tab_kmeans': '🤖 Clusters K-means',
        'tab_comparador': '🆚 Comparateur d\'Athlètes',
        'tab_mbi': '🔬 Analyse MBI',
        'tab_executive': '📋 Exécutif',
        'tab_individual': '👤 Individuel',
        'tab_trends': '📈 Tendances ML',
        'positions': 'Positions',
        'periods': 'Périodes',
        'athletes': 'Athlètes',
        'observations': 'Observations',
        'mean': 'Moyenne',
        'median': 'Médiane',
        'mode': 'Mode',
        'std': 'Écart Type',
        'variance': 'Variance',
        'cv': 'Coefficient de Variation',
        'min': 'Minimum',
        'max': 'Maximum',
        'amplitude': 'Amplitude',
        'q1': 'Q1 (25%)',
        'q3': 'Q3 (75%)',
        'iqr': 'IQR',
        'skewness': 'Asymétrie',
        'kurtosis': 'Kurtosis',
        'max_value': 'VALEUR MAXIMALE',
        'min_value': 'VALEUR MINIMALE',
        'minute_of_max': 'Minute du Max',
        'minute_of_min': 'Minute du Min',
        'threshold_75': 'SEUIL 75%',
        'threshold_50': 'SEUIL 50%',
        'high_intensity_events': 'ÉVÉNEMENTS DE HAUTE INTENSITÉ',
        'medium_high_intensity_events': 'ÉVÉNEMENTS D\'INTENSITÉ MOYENNE-ÉLEVÉE',
        'above_threshold_75': 'au-dessus du seuil de 75%',
        'between_50_75': 'entre 50% et 75%',
        'intensity_zones': '🎚️ Zones d\'Intensité',
        'zone_method': 'Méthode de définition',
        'percentiles': 'Percentiles',
        'based_on_max': 'Basé sur le Maximum',
        'very_low': 'Très Faible',
        'low': 'Faible',
        'moderate': 'Modérée',
        'high': 'Élevée',
        'very_high': 'Très Élevée',
        'process': '🚀 Traiter l\'Analyse',
        'descriptive_stats': '📊 Statistiques Descriptives',
        'confidence_interval': '🎯 Intervalle de Confiance (95%)',
        'normality_test': '🧪 Test de Normalité',
        'summary_by_group': '🏃 Résumé par Athlète, Position et Période',
        'symmetric': 'Approximativement symétrique',
        'moderate_skew': 'Modérément asymétrique',
        'high_skew': 'Fortement asymétrique',
        'leptokurtic': 'Leptokurtique (queues lourdes)',
        'platykurtic': 'Platykurtique (queues légères)',
        'mesokurtic': 'Mésokurtique (normale)',
        'strong_positive': 'Corrélation forte positive',
        'moderate_positive': 'Corrélation modérée positive',
        'weak_positive': 'Corrélation faible positive',
        'very_weak_positive': 'Corrélation très faible positive',
        'very_weak_negative': 'Corrélation très faible négative',
        'weak_negative': 'Corrélation faible négative',
        'moderate_negative': 'Corrélation modérée négative',
        'strong_negative': 'Corrélation forte négative',
        'iqr_title': '📌 Qu\'est-ce que l\'IQR ?',
        'iqr_explanation': 'L\'IQR (Intervalle Interquartile) est la différence entre le troisième quartile (Q3) et le premier quartile (Q1). Il représente l\'amplitude des 50% centraux des données.',
        'step1': '👈 **Étape 1:** Téléchargez un ou plusieurs fichiers CSV pour commencer',
        'step2': '👈 **Étape 2:** Sélectionnez les filtres et cliquez sur Traiter l\'Analyse',
        'file_format': '### 📋 Format de fichier attendu :',
        'col1_desc': '**Première colonne:** Identification au format `Nom-Période-Minute`',
        'col2_desc': '**Deuxième colonne:** Position de l\'athlète',
        'col3_desc': '**Autres colonnes (3+):** Variables numériques pour l\'analyse',
        'components': '📌 Composants',
        'name_ex': 'Nom: Mariano, Maria, Joao...',
        'period_ex': 'Période: 1 TEMPO, SEGUNDO TEMPO...',
        'minute_ex': 'Minute: 00:00-01:00, 05:00-06:00...',
        'position_ex': 'Position: Atacante, Meio-campo...',
        'tip': '💡 Astuce',
        'tip_text': 'Vous pouvez sélectionner plusieurs fichiers CSV avec la même structure.',
        'multi_file_ex': '📁 Exemple avec plusieurs fichiers',
        'multi_file_text': '### Chargement de plusieurs fichiers :\n1. Préparez vos fichiers CSV avec la **même structure** de colonnes\n2. Sélectionnez tous les fichiers souhaités\n3. Le système vérifiera la compatibilité et concaténera automatiquement',
        'select_period_timeline': 'Sélectionnez la période pour la visualisation temporelle',
        'all_periods': 'Toutes les périodes (graphique unique)',
        'compare_periods': 'Comparer les périodes (graphiques multiples)',
        'export_report': '📄 Exporter le Rapport',
        'export_excel': '📊 Exporter Excel',
        'export_pdf': '📑 Exporter PDF',
        'presentation_mode': '🎬 Mode Présentation',
        'theme': '🎨 Thème',
        'dark_theme': 'Sombre',
        'light_theme': 'Clair',
        'alerts': '⚠️ Alertes',
        'outlier_alert': 'Valeur Aberrante Détectée',
        'trend_alert': 'Tendance Préoccupante',
        'forecast': 'Prévision',
        'confidence_interval_forecast': 'Intervalle de Confiance',
        'athlete_profile': 'Profil de l\'Athlète',
        'performance_history': 'Historique des Performances',
        'weekly_summary': 'Résumé Hebdomadaire',
        'download_excel': 'Télécharger Excel',
        'download_pdf': 'Télécharger PDF'
    },
    'zh': {
        'name': '中文',
        'title': 'Sports Science Analytics Pro',
        'subtitle': '专业运动表现分析仪表板',
        'upload': '数据上传',
        'variable': '变量',
        'position': '位置',
        'period': '时间段',
        'athlete': '运动员',
        'config': '设置',
        'tab_distribution': '📊 分布',
        'tab_temporal': '📈 统计与时间',
        'tab_boxplots': '📦 箱线图',
        'tab_correlation': '🔥 相关性',
        'tab_kmeans': '🤖 K-means聚类',
        'tab_comparador': '🆚 运动员比较',
        'tab_mbi': '🔬 MBI分析',
        'tab_executive': '📋 执行摘要',
        'tab_individual': '👤 个人分析',
        'tab_trends': '📈 机器学习趋势',
        'positions': '位置',
        'periods': '时间段',
        'athletes': '运动员',
        'observations': '观测值',
        'mean': '平均值',
        'median': '中位数',
        'mode': '众数',
        'std': '标准差',
        'variance': '方差',
        'cv': '变异系数',
        'min': '最小值',
        'max': '最大值',
        'amplitude': '极差',
        'q1': '第一四分位数 (25%)',
        'q3': '第三四分位数 (75%)',
        'iqr': '四分位距',
        'skewness': '偏度',
        'kurtosis': '峰度',
        'max_value': '最大值',
        'min_value': '最小值',
        'minute_of_max': '最大值时刻',
        'minute_of_min': '最小值时刻',
        'threshold_75': '75% 阈值',
        'threshold_50': '50% 阈值',
        'high_intensity_events': '高强度事件',
        'medium_high_intensity_events': '中高强度事件',
        'above_threshold_75': '高于75%阈值',
        'between_50_75': '介于50%和75%之间',
        'intensity_zones': '🎚️ 强度区间',
        'zone_method': '定义方法',
        'percentiles': '百分位数',
        'based_on_max': '基于最大值',
        'very_low': '非常低',
        'low': '低',
        'moderate': '中等',
        'high': '高',
        'very_high': '非常高',
        'process': '🚀 处理分析',
        'descriptive_stats': '📊 描述性统计',
        'confidence_interval': '🎯 置信区间 (95%)',
        'normality_test': '🧪 正态性检验',
        'summary_by_group': '🏃 按运动员、位置和时间段汇总',
        'symmetric': '近似对称',
        'moderate_skew': '中度偏斜',
        'high_skew': '高度偏斜',
        'leptokurtic': '尖峰分布（重尾）',
        'platykurtic': '平峰分布（轻尾）',
        'mesokurtic': '正态峰度',
        'strong_positive': '强正相关',
        'moderate_positive': '中度正相关',
        'weak_positive': '弱正相关',
        'very_weak_positive': '非常弱正相关',
        'very_weak_negative': '非常弱负相关',
        'weak_negative': '弱负相关',
        'moderate_negative': '中度负相关',
        'strong_negative': '强负相关',
        'iqr_title': '📌 什么是四分位距？',
        'iqr_explanation': '四分位距是第三四分位数(Q3)与第一四分位数(Q1)之差。它代表数据中间50%的范围。',
        'step1': '👈 **第一步：** 上传一个或多个CSV文件开始',
        'step2': '👈 **第二步：** 选择筛选条件并点击"处理分析"',
        'file_format': '### 📋 期望的文件格式：',
        'col1_desc': '**第一列：** 格式为`姓名-时间段-分钟`的标识',
        'col2_desc': '**第二列：** 运动员位置',
        'col3_desc': '**其他列（3+）：** 用于分析的数值变量',
        'components': '📌 组成部分',
        'name_ex': '姓名：Mariano, Maria, Joao...',
        'period_ex': '时间段：1 TEMPO, SEGUNDO TEMPO...',
        'minute_ex': '分钟：00:00-01:00, 05:00-06:00...',
        'position_ex': '位置：Atacante, Meio-campo...',
        'tip': '💡 提示',
        'tip_text': '您可以选择多个具有相同结构的CSV文件。',
        'multi_file_ex': '📁 多文件示例',
        'multi_file_text': '### 加载多个文件：\n1. 准备具有**相同列结构**的CSV文件\n2. 选择所有需要的文件\n3. 系统将检查兼容性并自动合并',
        'select_period_timeline': '选择时间段进行时间可视化',
        'all_periods': '所有时间段（单一图表）',
        'compare_periods': '比较时间段（多个图表）',
        'export_report': '📄 导出报告',
        'export_excel': '📊 导出Excel',
        'export_pdf': '📑 导出PDF',
        'presentation_mode': '🎬 演示模式',
        'theme': '🎨 主题',
        'dark_theme': '深色',
        'light_theme': '浅色',
        'alerts': '⚠️ 警报',
        'outlier_alert': '检测到异常值',
        'trend_alert': '令人担忧的趋势',
        'forecast': '预测',
        'confidence_interval_forecast': '置信区间',
        'athlete_profile': '运动员档案',
        'performance_history': '表现历史',
        'weekly_summary': '周总结',
        'download_excel': '下载Excel',
        'download_pdf': '下载PDF'
    }
}

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
        st.session_state.idioma = 'pt_br'
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
    if 'modo_timeline' not in st.session_state:
        st.session_state.modo_timeline = 'unico'
    if 'periodo_timeline' not in st.session_state:
        st.session_state.periodo_timeline = None
    if 'df_filtrado' not in st.session_state:
        st.session_state.df_filtrado = None
    if 'kmeans_ativo' not in st.session_state:
        st.session_state.kmeans_ativo = False
    if 'kmeans_var_x' not in st.session_state:
        st.session_state.kmeans_var_x = None
    if 'kmeans_var_y' not in st.session_state:
        st.session_state.kmeans_var_y = None
    if 'kmeans_n_clusters' not in st.session_state:
        st.session_state.kmeans_n_clusters = 3
    if 'kmeans_resultados' not in st.session_state:
        st.session_state.kmeans_resultados = None
    if 'tema' not in st.session_state:
        st.session_state.tema = 'dark'
    if 'modo_apresentacao' not in st.session_state:
        st.session_state.modo_apresentacao = False
    if 'alertas' not in st.session_state:
        st.session_state.alertas = []

init_session_state()

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def extrair_periodo(texto):
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
    if not dataframes:
        return False, []
    
    primeira_estrutura = dataframes[0].columns.tolist()
    
    for i, df in enumerate(dataframes[1:], 1):
        if df.columns.tolist() != primeira_estrutura:
            return False, primeira_estrutura
    
    return True, primeira_estrutura

def calcular_cv(media, desvio):
    if media != 0 and not np.isnan(media) and not np.isnan(desvio):
        return (desvio / media) * 100
    return 0

def extrair_minuto_do_extremo(df, coluna_valor, coluna_minuto, extremo='max'):
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
        try:
            df_sorted = df.sort_values(coluna_valor, ascending=(extremo=='min'))
            return df_sorted.iloc[0][coluna_minuto]
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
    else:
        max_val = df[variavel].max()
        return {
            'Muito Baixa': max_val * 0.2,
            'Baixa': max_val * 0.4,
            'Moderada': max_val * 0.6,
            'Alta': max_val * 0.8,
            'Muito Alta': max_val
        }

def criar_tabela_destaque(df, colunas_destaque):
    styled_df = df.style
    
    for col in colunas_destaque:
        if col in df.select_dtypes(include=[np.number]).columns:
            styled_df = styled_df.background_gradient(
                subset=[col],
                cmap='viridis'
            )
    
    if 'Média' in df.columns:
        def highlight_max_row(row):
            if row.name == df['Média'].idxmax():
                return ['background-color: rgba(0, 158, 115, 0.2)'] * len(row)
            return [''] * len(row)
        
        styled_df = styled_df.apply(highlight_max_row, axis=1)
    
    return styled_df

def comparar_atletas(df, atleta1, atleta2, variaveis, t):
    dados1 = df[df['Nome'] == atleta1][variaveis].mean()
    dados2 = df[df['Nome'] == atleta2][variaveis].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {atleta1}")
        for var in variaveis:
            delta = ((dados1[var] - dados2[var]) / dados2[var]) * 100 if dados2[var] != 0 else 0
            cor = COR_SUCESSO if delta > 0 else COR_ALERTA
            st.markdown(f"""
            <div style="background: #1e293b; padding: 10px; border-radius: 8px; margin: 5px 0;
                        border-left: 3px solid {cor};">
                <span style="color: #94a3b8;">{var}:</span>
                <span style="color: white; font-weight: bold; float: right;">{dados1[var]:.2f}</span>
                <br>
                <span style="color: {cor}; font-size: 0.8rem;">
                    {delta:+.1f}% vs {atleta2}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### {atleta2}")
        for var in variaveis:
            delta = ((dados2[var] - dados1[var]) / dados1[var]) * 100 if dados1[var] != 0 else 0
            cor = COR_SUCESSO if delta > 0 else COR_ALERTA
            st.markdown(f"""
            <div style="background: #1e293b; padding: 10px; border-radius: 8px; margin: 5px 0;
                        border-left: 3px solid {cor};">
                <span style="color: #94a3b8;">{var}:</span>
                <span style="color: white; font-weight: bold; float: right;">{dados2[var]:.2f}</span>
                <br>
                <span style="color: {cor}; font-size: 0.8rem;">
                    {delta:+.1f}% vs {atleta1}
                </span>
            </div>
            """, unsafe_allow_html=True)

def sistema_anotacoes(t):
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
        
        for i, anotacao in enumerate(reversed(st.session_state.anotacoes)):
            st.markdown(f"""
            <div class="note-card">
                <p class="note-date">{anotacao['data']}</p>
                <p class="note-text">{anotacao['texto']}</p>
            </div>
            """, unsafe_allow_html=True)

def time_range_selector(t):
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
# FUNÇÕES PARA TIMELINE
# ============================================================================

def atualizar_modo_timeline():
    valor_radio = st.session_state.modo_timeline_radio
    if valor_radio == "Gráfico único":
        st.session_state.modo_timeline = 'unico'
    else:
        st.session_state.modo_timeline = 'multiplo'

def criar_timeline_multipla(df, variavel, periodos, t):
    n_periodos = len(periodos)
    fig = make_subplots(
        rows=n_periodos, 
        cols=1,
        subplot_titles=[f"Período: {p}" for p in periodos],
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    for i, periodo in enumerate(periodos, 1):
        df_periodo = df[df['Período'] == periodo].sort_values('Minuto').copy()
        
        if df_periodo.empty:
            continue
        
        valor_maximo = df_periodo[variavel].max()
        limiar_75 = valor_maximo * 0.75
        
        fig.add_trace(
            go.Scatter(
                x=df_periodo['Minuto'],
                y=df_periodo[variavel],
                mode='lines+markers',
                name=f'Período {periodo}',
                line=dict(color=COR_PRIMARIA, width=2),
                marker=dict(size=6, color=COR_PRIMARIA),
                showlegend=False
            ),
            row=i, col=1
        )
        
        fig.add_hline(
            y=limiar_75,
            line_dash="dash",
            line_color=COR_ALERTA,
            line_width=1,
            row=i, col=1
        )
        
        media_periodo = df_periodo[variavel].mean()
        fig.add_hline(
            y=media_periodo,
            line_dash="dot",
            line_color=COR_SUCESSO,
            line_width=1,
            row=i, col=1
        )
        
        fig.update_xaxes(
            title_text="Minuto" if i == n_periodos else "",
            gridcolor='#334155',
            tickfont=dict(color='white', size=9),
            tickangle=-45,
            row=i, col=1
        )
        
        fig.update_yaxes(
            title_text=variavel if i == n_periodos//2 + 1 else "",
            gridcolor='#334155',
            tickfont=dict(color='white'),
            row=i, col=1
        )
    
    fig.update_layout(
        title=f"Evolução Temporal por Período - {variavel}",
        plot_bgcolor='rgba(30,41,59,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=11),
        title_font=dict(color=COR_PRIMARIA, size=18),
        height=300 * n_periodos,
        showlegend=False
    )
    
    return fig

def criar_timeline_unica_com_seletor(df, variavel, periodos_selecionados, t):
    opcoes_periodo = ['Todos os períodos'] + list(periodos_selecionados)
    
    indice_atual = 0
    if st.session_state.periodo_timeline in opcoes_periodo:
        indice_atual = opcoes_periodo.index(st.session_state.periodo_timeline)
    
    periodo_escolhido = st.selectbox(
        t['select_period_timeline'],
        options=opcoes_periodo,
        index=indice_atual,
        key="periodo_timeline_select"
    )
    
    if periodo_escolhido != st.session_state.periodo_timeline:
        st.session_state.periodo_timeline = periodo_escolhido
    
    if periodo_escolhido == 'Todos os períodos':
        df_plot = df.copy()
        titulo = f"Evolução Temporal - {variavel} (Todos os períodos)"
    else:
        df_plot = df[df['Período'] == periodo_escolhido].copy()
        titulo = f"Evolução Temporal - {variavel} (Período: {periodo_escolhido})"
    
    df_plot = df_plot.sort_values('Minuto').reset_index(drop=True)
    
    fig = go.Figure()
    
    if periodo_escolhido == 'Todos os períodos':
        periodos_unicos = df_plot['Período'].unique()
        cores = [COR_PRIMARIA, COR_SECUNDARIA, COR_SUCESSO, COR_DESTAQUE, COR_ALERTA]
        
        for i, periodo in enumerate(periodos_unicos):
            df_periodo = df_plot[df_plot['Período'] == periodo]
            cor = cores[i % len(cores)]
            
            fig.add_trace(go.Scatter(
                x=df_periodo['Minuto'],
                y=df_periodo[variavel],
                mode='lines+markers',
                name=f'Período: {periodo}',
                line=dict(color=cor, width=2),
                marker=dict(size=6, color=cor),
                hovertemplate='<b>Período:</b> %{text}<br>' +
                              '<b>Minuto:</b> %{x}<br>' +
                              '<b>Valor:</b> %{y:.2f}<extra></extra>',
                text=[periodo] * len(df_periodo)
            ))
    else:
        valor_maximo = df_plot[variavel].max()
        limiar_75 = valor_maximo * 0.75
        media = df_plot[variavel].mean()
        desvio = df_plot[variavel].std()
        
        fig.add_hrect(
            y0=limiar_75,
            y1=valor_maximo * 1.05,
            fillcolor=f"rgba(213, 94, 0, 0.15)",
            line_width=0,
            layer="below",
            name="Acima do limiar de 75%"
        )
        
        fig.add_hrect(
            y0=0,
            y1=limiar_75,
            fillcolor=f"rgba(0, 114, 178, 0.1)",
            line_width=0,
            layer="below",
            name="Abaixo do limiar de 75%"
        )
        
        fig.add_hline(
            y=limiar_75,
            line_dash="solid",
            line_color=COR_ALERTA,
            line_width=2,
            annotation_text=f"🔴 Limiar 75%: {limiar_75:.2f}",
            annotation_position="top left"
        )
        
        fig.add_hline(
            y=media,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Média: {media:.2f}",
            annotation_position="top left"
        )
        
        fig.add_hrect(
            y0=media-desvio,
            y1=media+desvio,
            fillcolor=COR_PRIMARIA,
            opacity=0.1,
            line_width=0,
            annotation_text="±1 DP"
        )
        
        fig.add_trace(go.Scatter(
            x=df_plot['Minuto'],
            y=df_plot[variavel],
            mode='lines+markers',
            name=variavel,
            line=dict(color=COR_PRIMARIA, width=2),
            marker=dict(size=8, color=COR_PRIMARIA, line=dict(color='white', width=1)),
            hovertemplate='<b>Minuto:</b> %{x}<br>' +
                          '<b>Valor:</b> %{y:.2f}<extra></extra>'
        ))
        
        media_movevel = df_plot[variavel].rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_plot['Minuto'],
            y=media_movevel,
            mode='lines',
            name='Média Móvel (5)',
            line=dict(color=COR_SECUNDARIA, width=2, dash='dot')
        ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Minuto",
        yaxis_title=variavel,
        plot_bgcolor='rgba(30,41,59,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title_font=dict(color=COR_PRIMARIA, size=18),
        hovermode='closest',
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(30,41,59,0.8)',
            bordercolor='#334155'
        )
    )
    
    fig.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'), tickangle=-45)
    fig.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
    
    return fig

def criar_grafico_barras_desvio(df_atleta, df_posicao, df_geral, atleta_nome, posicao, variaveis, titulo="Comparação de Desempenho"):
    valores_atleta = [df_atleta[var].mean() for var in variaveis]
    valores_posicao = [df_posicao[var].mean() for var in variaveis]
    valores_geral = [df_geral[var].mean() for var in variaveis]
    
    desvios_vs_posicao = [((v - valores_posicao[i]) / valores_posicao[i]) * 100 if valores_posicao[i] != 0 else 0 
                          for i, v in enumerate(valores_atleta)]
    desvios_vs_geral = [((v - valores_geral[i]) / valores_geral[i]) * 100 if valores_geral[i] != 0 else 0 
                        for i, v in enumerate(valores_atleta)]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"<b>Valores Absolutos</b>", 
            f"<b>Desvio Percentual do Atleta vs Médias</b>"
        ),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    fig.add_trace(go.Bar(
        x=variaveis,
        y=valores_atleta,
        name=atleta_nome,
        marker_color=COR_PRIMARIA,
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.9,
        text=[f'{v:.1f}' for v in valores_atleta],
        textposition='outside',
        textfont=dict(color='white', size=11, family="Arial Black"),
        hovertemplate='<b>%{x}</b><br>' +
                      f'{atleta_nome}: %{{y:.2f}}<br>' +
                      '<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=variaveis,
        y=valores_posicao,
        name=f'Média {posicao}',
        mode='lines+markers',
        line=dict(color=COR_SECUNDARIA, width=3, dash='dash'),
        marker=dict(size=10, color=COR_SECUNDARIA, symbol='diamond'),
        hovertemplate='<b>%{x}</b><br>' +
                      f'Média {posicao}: %{{y:.2f}}<br>' +
                      '<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=variaveis,
        y=valores_geral,
        name='Média Geral',
        mode='lines+markers',
        line=dict(color=COLORS['gray'], width=3, dash='dot'),
        marker=dict(size=10, color=COLORS['gray'], symbol='circle'),
        hovertemplate='<b>%{x}</b><br>' +
                      'Média Geral: %{y:.2f}<br>' +
                      '<extra></extra>'
    ), row=1, col=1)
    
    cores_desvio = [COR_SUCESSO if d > 0 else COR_ALERTA if d < 0 else COLORS['gray'] for d in desvios_vs_posicao]
    
    fig.add_trace(go.Bar(
        x=variaveis,
        y=desvios_vs_posicao,
        name=f'Desvio vs {posicao}',
        marker_color=cores_desvio,
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.8,
        text=[f'{d:+.1f}%' for d in desvios_vs_posicao],
        textposition='outside',
        textfont=dict(color='white', size=10, family="Arial Black"),
        hovertemplate='<b>%{x}</b><br>' +
                      f'vs {posicao}: %{{y:+.1f}}%<br>' +
                      '<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=variaveis,
        y=desvios_vs_geral,
        name='Desvio vs Geral',
        mode='markers',
        marker=dict(
            size=12,
            color=COR_DESTAQUE,
            symbol='star',
            line=dict(color='white', width=1)
        ),
        text=[f'{d:+.1f}%' for d in desvios_vs_geral],
        textposition='top center',
        textfont=dict(color='white', size=9),
        hovertemplate='<b>%{x}</b><br>' +
                      'vs Geral: %{y:+.1f}%<br>' +
                      '<extra></extra>'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color=COLORS['gray'], line_width=1, opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{titulo}</b>",
            font=dict(size=24, color='white', family="Arial Black"),
            x=0.5
        ),
        plot_bgcolor='rgba(30,41,59,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=800,
        width=900,
        showlegend=True,
        legend=dict(
            font=dict(color='white', size=11),
            bgcolor='rgba(30,41,59,0.9)',
            bordercolor=COR_PRIMARIA,
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            itemclick='toggle',
            itemdoubleclick='toggleothers'
        ),
        hovermode='x unified',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    fig.update_xaxes(
        gridcolor='#334155', 
        tickfont=dict(color='white', size=11),
        title_font=dict(color='white'),
        showline=True,
        linecolor='#334155',
        row=1, col=1
    )
    fig.update_xaxes(
        gridcolor='#334155', 
        tickfont=dict(color='white', size=11),
        title_font=dict(color='white'),
        showline=True,
        linecolor='#334155',
        row=2, col=1
    )
    
    fig.update_yaxes(
        gridcolor='#334155', 
        tickfont=dict(color='white'),
        title="Valor",
        title_font=dict(color='white', size=12),
        showline=True,
        linecolor='#334155',
        row=1, col=1
    )
    
    fig.update_yaxes(
        gridcolor='#334155', 
        tickfont=dict(color='white'),
        title="Desvio (%)",
        title_font=dict(color='white', size=12),
        showline=True,
        linecolor='#334155',
        zeroline=True,
        zerolinecolor='#4a5568',
        zerolinewidth=1,
        row=2, col=1
    )
    
    return fig, valores_atleta, valores_posicao, valores_geral

def criar_tabela_comparativa(atleta_nome, posicao, variaveis, valores_atleta, valores_posicao, valores_geral):
    dados = []
    
    for i, var in enumerate(variaveis):
        val_atleta = valores_atleta[i]
        val_posicao = valores_posicao[i]
        val_geral = valores_geral[i]
        
        diff_vs_posicao = ((val_atleta - val_posicao) / val_posicao) * 100 if val_posicao != 0 else 0
        diff_vs_geral = ((val_atleta - val_geral) / val_geral) * 100 if val_geral != 0 else 0
        
        icone_pos = '▲' if diff_vs_posicao > 0 else '▼' if diff_vs_posicao < 0 else '◆'
        icone_geral = '▲' if diff_vs_geral > 0 else '▼' if diff_vs_geral < 0 else '◆'
        
        dados.append({
            '📊 Métrica': var,
            '🏃 Atleta': f'{val_atleta:.2f}',
            '📊 Média Posição': f'{val_posicao:.2f}',
            '📈 Média Geral': f'{val_geral:.2f}',
            '🎯 vs Posição': f'{icone_pos} {abs(diff_vs_posicao):.1f}%',
            '🌍 vs Geral': f'{icone_geral} {abs(diff_vs_geral):.1f}%'
        })
    
    return dados

def criar_card_resumo(atleta_nome, posicao, dados_comparativos):
    maiores_vantagens = []
    maiores_desvantagens = []
    
    for item in dados_comparativos:
        valor_str = item['🎯 vs Posição']
        if '▲' in valor_str:
            pct = float(valor_str.replace('▲', '').replace('%', '').strip())
            maiores_vantagens.append((item['📊 Métrica'], pct))
        elif '▼' in valor_str:
            pct = float(valor_str.replace('▼', '').replace('%', '').strip())
            maiores_desvantagens.append((item['📊 Métrica'], pct))
    
    maiores_vantagens.sort(key=lambda x: x[1], reverse=True)
    maiores_desvantagens.sort(key=lambda x: x[1], reverse=True)
    
    return maiores_vantagens[:3], maiores_desvantagens[:3]

# ============================================================================
# FUNÇÕES PARA MAGNITUDE-BASED INFERENCE (MBI)
# ============================================================================

def calcular_mbi(valor_atleta, media_grupo, desvio_grupo, n_grupo=30, small_effect=0.2):
    diferenca = valor_atleta - media_grupo
    cohen_d = diferenca / desvio_grupo if desvio_grupo != 0 else 0
    
    limiar_pequeno = small_effect * desvio_grupo
    
    erro_padrao = desvio_grupo / np.sqrt(n_grupo) if n_grupo > 0 else 0
    ic_inf = diferenca - 1.645 * erro_padrao
    ic_sup = diferenca + 1.645 * erro_padrao
    
    if abs(cohen_d) < 0.2:
        magnitude = "trivial"
    elif abs(cohen_d) < 0.6:
        magnitude = "pequena"
    elif abs(cohen_d) < 1.2:
        magnitude = "moderada"
    else:
        magnitude = "grande"
    
    if ic_inf > limiar_pequeno:
        inferencia = "Muito provavelmente benéfico"
        cor = COR_SUCESSO
        prob = 0.95
        icone = "✅"
    elif ic_sup < -limiar_pequeno:
        inferencia = "Muito provavelmente prejudicial"
        cor = COR_ALERTA
        prob = 0.95
        icone = "❌"
    elif ic_inf > -limiar_pequeno and ic_sup < limiar_pequeno:
        inferencia = "Quase certamente trivial"
        cor = COLORS['gray']
        prob = 0.90
        icone = "➖"
    elif ic_inf > -limiar_pequeno and ic_sup > limiar_pequeno and ic_inf < limiar_pequeno:
        if abs(cohen_d) < 0.2:
            inferencia = "Pouco claro (possivelmente trivial/benéfico)"
        else:
            inferencia = "Possivelmente benéfico"
        cor = COR_SECUNDARIA
        prob = 0.75
        icone = "⚠️"
    elif ic_inf < -limiar_pequeno and ic_sup < -limiar_pequeno and ic_sup > -limiar_pequeno:
        inferencia = "Possivelmente prejudicial"
        cor = COR_SECUNDARIA
        prob = 0.75
        icone = "⚠️"
    else:
        inferencia = "Não claro (necessário mais dados)"
        cor = COR_PRIMARIA
        prob = 0.50
        icone = "❓"
    
    return {
        'inferencia': inferencia,
        'cor': cor,
        'cohen_d': cohen_d,
        'ic_90': (ic_inf, ic_sup),
        'probabilidade': prob,
        'magnitude': magnitude,
        'icone': icone
    }

def criar_card_mbi(resultado, atleta_nome, var_nome):
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.8); border-radius: 12px; padding: 20px; 
                border-left: 8px solid {resultado['cor']}; margin: 15px 0;
                backdrop-filter: blur(10px);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="color: white; margin: 0;">{var_nome}</h4>
                <p style="color: white; font-size: 1.2rem; margin: 5px 0;">
                    <strong>{resultado['inferencia']}</strong>
                </p>
                <p style="color: #94a3b8; margin: 5px 0;">
                    Cohen's d = {resultado['cohen_d']:.2f} 
                    <span style="background: {resultado['cor']}; color: white; padding: 2px 8px; border-radius: 12px; margin-left: 10px;">
                        {resultado['magnitude']}
                    </span>
                </p>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 5px 0;">
                    IC 90%: [{resultado['ic_90'][0]:.2f}, {resultado['ic_90'][1]:.2f}]
                </p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">{resultado['icone']}</div>
                <p style="color: white; margin: 0;">{resultado['probabilidade']*100:.0f}%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def criar_heatmap_magnitude(df, posicao_referencia=None):
    metricas = df.select_dtypes(include=[np.number]).columns.tolist()
    metricas = [m for m in metricas if m not in ['Minuto']]
    
    if len(metricas) > 10:
        metricas = metricas[:10]
    
    dados_heatmap = []
    for atleta in df['Nome'].unique():
        df_atleta = df[df['Nome'] == atleta]
        linha = {'Atleta': atleta}
        
        for metrica in metricas:
            if metrica in df.columns:
                valor = df_atleta[metrica].mean()
                media_grupo = df[metrica].mean()
                desvio_grupo = df[metrica].std()
                
                z_score = (valor - media_grupo) / desvio_grupo if desvio_grupo != 0 else 0
                linha[metrica] = z_score
        
        dados_heatmap.append(linha)
    
    df_heat = pd.DataFrame(dados_heatmap).set_index('Atleta')
    
    colorscale = [
        [0, 'rgb(0, 114, 178)'],
        [0.25, 'rgb(86, 180, 233)'],
        [0.5, 'rgb(255, 255, 255)'],
        [0.75, 'rgb(230, 159, 0)'],
        [1, 'rgb(213, 94, 0)']
    ]
    
    fig = px.imshow(
        df_heat.T,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale=colorscale,
        title=f"Perfil de Magnitudes (Z-Scores) - {'Todas Posições' if not posicao_referencia else posicao_referencia}",
        labels=dict(x="Atleta", y="Métrica", color="Z-Score")
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(30,41,59,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family="Arial"),
        title_font=dict(color=COR_PRIMARIA, size=18),
        height=500,
        width=900,
        xaxis=dict(tickangle=-45, tickfont=dict(color='white', size=10)),
        yaxis=dict(tickfont=dict(color='white', size=10)),
        coloraxis_colorbar=dict(
            title="Magnitude",
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=['Muito Baixo (-2σ)', 'Baixo (-1σ)', 'Média (0σ)', 'Alto (+1σ)', 'Muito Alto (+2σ)'],
            title_font=dict(color='white', size=11),
            tickfont=dict(color='white', size=10)
        )
    )
    
    fig.update_traces(textfont=dict(size=9, color='black'))
    fig.update_xaxes(gridcolor='#334155')
    fig.update_yaxes(gridcolor='#334155')
    
    return fig

# ============================================================================
# FUNÇÕES DE EXPORTAÇÃO
# ============================================================================

def export_to_excel(df, stats_dict, t):
    """Exporta dados para Excel"""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Dados', index=False)
            
            # Criar sheet com estatísticas
            stats_df = pd.DataFrame(list(stats_dict.items()), columns=['Métrica', 'Valor'])
            stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)
    except Exception as e:
        st.warning(f"Erro ao exportar Excel: {e}")
        return None
    
    output.seek(0)
    return output

def export_to_pdf(df, stats_dict, t):
    """Exporta relatório para PDF"""
    output = io.BytesIO()
    try:
        doc = SimpleDocTemplate(output, pagesize=letter)
        story = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#0072B2')
        )
        
        # Título
        story.append(Paragraph(t['title'], title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"{t['subtitle']}", styles['Normal']))
        story.append(Spacer(1, 24))
        
        # Estatísticas
        story.append(Paragraph(t['descriptive_stats'], styles['Heading2']))
        story.append(Spacer(1, 12))
        
        stats_data = [[k, str(v)] for k, v in stats_dict.items()]
        stats_table = Table(stats_data, colWidths=[2*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0072B2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
        
        doc.build(story)
    except Exception as e:
        st.warning(f"Erro ao exportar PDF: {e}")
        return None
    
    output.seek(0)
    return output

# ============================================================================
# FUNÇÕES DE ALERTAS E PREVISÕES
# ============================================================================

def detectar_outliers(df, variavel):
    """Detecta outliers usando o método IQR"""
    q1 = df[variavel].quantile(0.25)
    q3 = df[variavel].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    outliers = df[(df[variavel] < limite_inferior) | (df[variavel] > limite_superior)]
    return outliers, limite_inferior, limite_superior

def detectar_tendencias(df, variavel, janela=5):
    """Detecta tendências preocupantes nos dados"""
    if len(df) < janela:
        return None
    
    # Calcula média móvel
    media_movel = df[variavel].rolling(window=janela).mean()
    
    # Verifica tendência de queda
    if len(media_movel.dropna()) >= 2:
        ultimos_valores = media_movel.dropna().tail(3)
        if len(ultimos_valores) >= 2:
            if ultimos_valores.iloc[-1] < ultimos_valores.iloc[-2] * 0.9:
                return "Queda significativa detectada nos últimos períodos"
    
    return None

def forecast_performance(df, variavel, periods_ahead=5):
    """Previsão de desempenho usando Random Forest"""
    if len(df) < 10:
        return None, None, None
    
    # Preparar dados para treino
    df_forecast = df[[variavel]].copy()
    df_forecast['lag_1'] = df_forecast[variavel].shift(1)
    df_forecast['lag_2'] = df_forecast[variavel].shift(2)
    df_forecast['lag_3'] = df_forecast[variavel].shift(3)
    df_forecast['rolling_mean_3'] = df_forecast[variavel].rolling(window=3).mean()
    df_forecast = df_forecast.dropna()
    
    if len(df_forecast) < 5:
        return None, None, None
    
    X = df_forecast[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']].values
    y = df_forecast[variavel].values
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Fazer previsões
    ultimos_valores = df[variavel].tail(3).values
    previsoes = []
    intervalos = []
    
    for i in range(periods_ahead):
        if i == 0:
            features = np.array([[
                ultimos_valores[-1] if len(ultimos_valores) > 0 else 0,
                ultimos_valores[-2] if len(ultimos_valores) > 1 else 0,
                ultimos_valores[-3] if len(ultimos_valores) > 2 else 0,
                np.mean(ultimos_valores[-3:]) if len(ultimos_valores) >= 3 else 0
            ]])
        else:
            features = np.array([[
                previsoes[-1],
                previsoes[-2] if len(previsoes) > 1 else 0,
                previsoes[-3] if len(previsoes) > 2 else 0,
                np.mean(previsoes[-3:]) if len(previsoes) >= 3 else 0
            ]])
        
        pred = model.predict(features)[0]
        previsoes.append(pred)
        
        # Calcular intervalo de confiança aproximado
        intervalo = np.std(y) * 1.96
        intervalos.append((pred - intervalo, pred + intervalo))
    
    return previsoes, intervalos, model.score(X, y)

def criar_grafico_forecast(df, variavel, previsoes, intervalos, t):
    """Cria gráfico de previsão de desempenho"""
    fig = go.Figure()
    
    # Dados históricos
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[variavel].values,
        mode='lines+markers',
        name='Histórico',
        line=dict(color=COR_PRIMARIA, width=2),
        marker=dict(size=8, color=COR_PRIMARIA)
    ))
    
    # Previsões
    indices_futuros = list(range(len(df), len(df) + len(previsoes)))
    fig.add_trace(go.Scatter(
        x=indices_futuros,
        y=previsoes,
        mode='lines+markers',
        name=t['forecast'],
        line=dict(color=COR_ALERTA, width=2, dash='dash'),
        marker=dict(size=10, color=COR_ALERTA, symbol='diamond')
    ))
    
    # Intervalo de confiança
    y_upper = [intervalo[1] for intervalo in intervalos]
    y_lower = [intervalo[0] for intervalo in intervalos]
    
    fig.add_trace(go.Scatter(
        x=indices_futuros + indices_futuros[::-1],
        y=y_upper + y_lower[::-1],
        fill='toself',
        fillcolor=f'rgba(213, 94, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=t['confidence_interval_forecast']
    ))
    
    fig.update_layout(
        title=f'Previsão de Desempenho - {variavel}',
        xaxis_title='Observações',
        yaxis_title=variavel,
        plot_bgcolor='rgba(30,41,59,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title_font=dict(color=COR_PRIMARIA, size=18),
        hovermode='closest'
    )
    
    return fig

# ============================================================================
# FUNÇÕES DE CSS DINÂMICO PARA TEMAS
# ============================================================================

def get_theme_css(tema):
    """Retorna o CSS baseado no tema selecionado"""
    theme = THEMES[tema]
    return f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%);
        }}
        
        .executive-card, .metric-card, .time-metric-card, .zone-card, .note-card, .metric-container {{
            background: {theme['bg_card']};
            border-color: {theme['border']};
        }}
        
        .executive-card .label, .metric-card h3, .time-metric-card .label, .metric-container h4 {{
            color: {theme['text_secondary']};
        }}
        
        .executive-card .value, .metric-card h2, .time-metric-card .value, .metric-container p {{
            color: {theme['text_primary']};
        }}
        
        h1, h2, h3, h4, p, li, .stMarkdown, .stTabs [data-baseweb="tab"] {{
            color: {theme['text_primary']} !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            background: rgba(30, 41, 59, 0.6);
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COR_PRIMARIA} 0%, {COR_DESTAQUE} 100%) !important;
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, {COR_PRIMARIA} 0%, {COLORS['darkblue']} 100%);
        }}
        
        .dataframe th {{
            background: {theme['bg_secondary']} !important;
        }}
        
        .dataframe td {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
        }}
    </style>
    """

def toggle_presentation_mode():
    """Alterna o modo de apresentação"""
    if st.session_state.modo_apresentacao:
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            .main > div {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: block;
            }
        </style>
        """, unsafe_allow_html=True)

# ============================================================================
# FUNÇÃO PRINCIPAL PARA PROCESSAR UPLOAD
# ============================================================================

def processar_upload(upload_files):
    """Processa os arquivos enviados e atualiza o session state"""
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
            st.error("❌ Arquivos com estruturas diferentes")
            return False
        
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
                    st.session_state.atletas_selecionados = sorted(df_completo['Nome'].unique().tolist())
                    st.session_state.todos_posicoes = posicoes_unicas
                    st.session_state.posicoes_selecionadas = posicoes_unicas.copy()
                    st.session_state.todos_periodos = periodos_unicos
                    st.session_state.periodos_selecionados = periodos_unicos.copy()
                    st.session_state.ordem_personalizada = periodos_unicos.copy()
                    st.session_state.upload_files_names = arquivos_validos
                    st.session_state.upload_concluido = True
                    
                    if variaveis_quant and st.session_state.variavel_selecionada is None:
                        st.session_state.variavel_selecionada = variaveis_quant[0]
                    
                    return True
    
    return False

# ============================================================================
# CALLBACKS
# ============================================================================

def atualizar_metodo_zona():
    if hasattr(st.session_state, 'metodo_zona_radio'):
        valor_radio = st.session_state.metodo_zona_radio
        if valor_radio in ["Percentis", "Percentiles"]:
            st.session_state.metodo_zona = 'percentis'
        else:
            st.session_state.metodo_zona = 'based_on_max'

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Seletor de idioma
    st.markdown("<h2 class='sidebar-title'>🌐 Idioma / Language</h2>", unsafe_allow_html=True)
    
    idioma_opcoes = {
        'pt_br': 'Português (Brasil)',
        'es_mx': 'Español (México)',
        'es_la': 'Español (Latinoamérica)',
        'en_us': 'English (US)',
        'en_uk': 'English (UK)',
        'fr': 'Français',
        'zh': '中文'
    }
    
    idioma_selecionado = st.selectbox(
        "",
        options=list(idioma_opcoes.keys()),
        format_func=lambda x: idioma_opcoes[x],
        index=list(idioma_opcoes.keys()).index(st.session_state.idioma) if st.session_state.idioma in idioma_opcoes else 0,
        key="idioma_selector"
    )
    
    if idioma_selecionado != st.session_state.idioma:
        st.session_state.idioma = idioma_selecionado
        st.rerun()
    
    t = translations[st.session_state.idioma]
    
    # Configurações de tema e modo apresentação
    st.markdown("---")
    st.markdown(f"<h2 class='sidebar-title'>⚙️ {t['config']}</h2>", unsafe_allow_html=True)
    
    col_theme1, col_theme2 = st.columns(2)
    with col_theme1:
        tema_atual = st.selectbox(
            t['theme'],
            options=['dark', 'light'],
            format_func=lambda x: t['dark_theme'] if x == 'dark' else t['light_theme'],
            index=0 if st.session_state.tema == 'dark' else 1,
            key="tema_selector"
        )
        if tema_atual != st.session_state.tema:
            st.session_state.tema = tema_atual
            st.rerun()
    
    with col_theme2:
        modo_apresentacao = st.checkbox(
            t['presentation_mode'],
            value=st.session_state.modo_apresentacao,
            key="modo_apresentacao_check"
        )
        if modo_apresentacao != st.session_state.modo_apresentacao:
            st.session_state.modo_apresentacao = modo_apresentacao
            st.rerun()
    
    # Upload de arquivos
    st.markdown("---")
    st.markdown(f"<h2 class='sidebar-title'>📂 {t['upload']}</h2>", unsafe_allow_html=True)
    
    upload_files = st.file_uploader(
        "",
        type=['csv'],
        accept_multiple_files=True,
        help=t['tip_text'],
        key="file_uploader"
    )
    
    # Verificar se novos arquivos foram carregados
    if upload_files and len(upload_files) > 0:
        if st.session_state.upload_concluido and len(upload_files) != len(st.session_state.upload_files_names):
            st.session_state.upload_concluido = False
        
        if not st.session_state.upload_concluido:
            with st.spinner('🔄 Processando...'):
                time.sleep(0.5)
                sucesso = processar_upload(upload_files)
                if sucesso:
                    sucesso_msg = "arquivo(s) carregado(s)"
                    st.success(f"✅ {len(upload_files)} {sucesso_msg}")
                    st.rerun()
    
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
            
            df_temp = st.session_state.df_completo[variavel_selecionada].dropna()
            if not df_temp.empty:
                st.caption(f"📊 {len(df_temp)} obs | {t['mean']}: {df_temp.mean():.2f}")
        
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown(f"<h2 class='sidebar-title'>📍 {t['position']}</h2>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox(
                f"Selecionar todas as {t['position'].lower()}s",
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
                f"Selecionar todos os {t['period'].lower()}s",
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
        
        st.markdown("---")
        st.markdown(f"<h2 class='sidebar-title'>👤 {t['athlete']}</h2>", unsafe_allow_html=True)
        
        df_temp = st.session_state.df_completo.copy()
        if st.session_state.posicoes_selecionadas:
            df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
        if st.session_state.periodos_selecionados:
            df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
        
        atletas_disponiveis = sorted(df_temp['Nome'].unique().tolist()) if not df_temp.empty else []
        
        if not atletas_disponiveis:
            st.warning("⚠️ Nenhum atleta disponível com os filtros atuais")
            st.session_state.atletas_selecionados = []
        else:
            atletas_selecionados_validos = [a for a in st.session_state.atletas_selecionados if a in atletas_disponiveis]
            
            if not atletas_selecionados_validos:
                atletas_selecionados_validos = [atletas_disponiveis[0]]
                st.session_state.atletas_selecionados = atletas_selecionados_validos
            
            selecionar_todos = st.checkbox(
                f"Selecionar todos os {t['athlete'].lower()}s",
                value=len(atletas_selecionados_validos) == len(atletas_disponiveis) and len(atletas_disponiveis) > 0,
                key="todos_atletas_check"
            )
            
            if selecionar_todos:
                if atletas_selecionados_validos != atletas_disponiveis:
                    st.session_state.atletas_selecionados = atletas_disponiveis
                    st.session_state.dados_processados = False
                    st.rerun()
            else:
                atletas_sel = st.multiselect(
                    "",
                    options=atletas_disponiveis,
                    default=atletas_selecionados_validos,
                    label_visibility="collapsed",
                    key="atletas_selector"
                )
                
                if atletas_sel != st.session_state.atletas_selecionados:
                    if len(atletas_sel) > 0:
                        st.session_state.atletas_selecionados = atletas_sel
                    else:
                        st.session_state.atletas_selecionados = [atletas_disponiveis[0]]
                    st.session_state.dados_processados = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown(f"<h2 class='sidebar-title'>⚙️ {t['config']}</h2>", unsafe_allow_html=True)
        
        n_classes = st.slider(f"{t['config']}:", 3, 20, st.session_state.n_classes, key="classes_slider")
        if n_classes != st.session_state.n_classes:
            st.session_state.n_classes = n_classes
            st.rerun()
        
        st.markdown("---")
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button(t['process'], use_container_width=True, disabled=not pode_processar, key="process_button"):
            st.session_state.processar_click = True
            st.session_state.kmeans_ativo = False
            st.rerun()

# ============================================================================
# HEADER
# ============================================================================

if not st.session_state.modo_apresentacao:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h1>🏃 Sports Science Analytics Pro</h1>
            <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 10px;">
                {t['subtitle']}
            </p>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
                <span style="background: {COR_PRIMARIA}; color: white; padding: 5px 15px; border-radius: 50px; font-size: 0.9rem;">⚡ Real-time</span>
                <span style="background: {COR_DESTAQUE}; color: white; padding: 5px 15px; border-radius: 50px; font-size: 0.9rem;">📊 Statistical</span>
                <span style="background: {COR_SUCESSO}; color: white; padding: 5px 15px; border-radius: 50px; font-size: 0.9rem;">🎯 Precision</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Aplicar tema
st.markdown(get_theme_css(st.session_state.tema), unsafe_allow_html=True)

# Modo apresentação
if st.session_state.modo_apresentacao:
    toggle_presentation_mode()

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.df_completo is not None:
    
    if st.session_state.processar_click:
        with st.spinner('🔄 ' + ("Gerando análises...")):
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
                st.warning("⚠️ " + ("Nenhum dado encontrado"))
                st.session_state.dados_processados = False
                st.session_state.df_filtrado = None
            else:
                # Detectar alertas
                alertas = []
                
                # Detectar outliers
                outliers, _, _ = detectar_outliers(df_filtrado, variavel_analise)
                if len(outliers) > 0:
                    alertas.append(f"⚠️ {t['outlier_alert']}: {len(outliers)} valores anômalos detectados em {variavel_analise}")
                
                # Detectar tendências
                tendencia = detectar_tendencias(df_filtrado, variavel_analise)
                if tendencia:
                    alertas.append(f"📉 {t['trend_alert']}: {tendencia}")
                
                st.session_state.alertas = alertas
                st.session_state.dados_processados = True
                st.session_state.df_filtrado = df_filtrado
                st.session_state.processar_click = False
                st.rerun()
    
    elif st.session_state.dados_processados and st.session_state.df_filtrado is not None:
        df_filtrado = st.session_state.df_filtrado
        atletas_selecionados = st.session_state.atletas_selecionados
        posicoes_selecionadas = st.session_state.posicoes_selecionadas
        periodos_selecionados = st.session_state.periodos_selecionados
        variavel_analise = st.session_state.variavel_selecionada
        n_classes = st.session_state.n_classes
        
        # Exibir alertas
        if st.session_state.alertas:
            with st.expander(f"⚠️ {t['alerts']} ({len(st.session_state.alertas)})", expanded=True):
                for alerta in st.session_state.alertas:
                    st.warning(alerta)
        
        # Botões de exportação
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        # Preparar estatísticas para exportação
        stats_export = {
            t['mean']: f"{df_filtrado[variavel_analise].mean():.2f}",
            t['median']: f"{df_filtrado[variavel_analise].median():.2f}",
            t['std']: f"{df_filtrado[variavel_analise].std():.2f}",
            t['cv']: f"{calcular_cv(df_filtrado[variavel_analise].mean(), df_filtrado[variavel_analise].std()):.1f}%",
            t['min']: f"{df_filtrado[variavel_analise].min():.2f}",
            t['max']: f"{df_filtrado[variavel_analise].max():.2f}",
            t['observations']: str(len(df_filtrado))
        }
        
        with col_exp1:
            excel_data = export_to_excel(df_filtrado, stats_export, t)
            if excel_data:
                st.download_button(
                    label=t['export_excel'],
                    data=excel_data,
                    file_name=f"analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col_exp2:
            pdf_data = export_to_pdf(df_filtrado, stats_export, t)
            if pdf_data:
                st.download_button(
                    label=t['export_pdf'],
                    data=pdf_data,
                    file_name=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        with col_exp3:
            st.markdown(f"<div style='text-align: center; padding: 10px; background: rgba(30,41,59,0.5); border-radius: 8px;'>{t['observations']}: {len(df_filtrado)}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Cards executivos
        st.markdown(f"<h2>📊 Visão Geral</h2>", unsafe_allow_html=True)
        
        media_global = df_filtrado[variavel_analise].mean()
        desvio_global = df_filtrado[variavel_analise].std()
        cv_global = calcular_cv(media_global, desvio_global)
        amplitude_global = df_filtrado[variavel_analise].max() - df_filtrado[variavel_analise].min()
        
        # Função para executive card
        def executive_card(titulo, valor, icone, cor_status=COR_PRIMARIA):
            st.markdown(f"""
            <div class="executive-card" style="border-left-color: {cor_status};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p class="label">{titulo}</p>
                        <p class="value">{valor}</p>
                    </div>
                    <div class="icon">{icone}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        cols_exec = st.columns(4)
        with cols_exec[0]:
            executive_card(t['mean'], f"{media_global:.2f}", "📊")
        with cols_exec[1]:
            executive_card(t['cv'], f"{cv_global:.1f}%", "📈")
        with cols_exec[2]:
            executive_card(t['amplitude'], f"{amplitude_global:.2f}", "📏")
        with cols_exec[3]:
            executive_card(t['observations'], len(df_filtrado), "👥")
        
        st.markdown("---")
        
        data_inicio, data_fim = time_range_selector(t)
        
        st.markdown("---")
        
        # Tabs principais
        tab_titles = [
            t['tab_distribution'], 
            t['tab_temporal'], 
            t['tab_boxplots'], 
            t['tab_correlation'],
            t['tab_kmeans'],
            t['tab_comparador'],
            t['tab_mbi'],
            t['tab_individual'],
            t['tab_trends'],
            t['tab_executive']
        ]
        
        tabs = st.tabs(tab_titles)
        
        # ================================================================
        # TAB 0: DISTRIBUIÇÃO
        # ================================================================
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
                    marker_color=COR_PRIMARIA,
                    opacity=0.8
                ))
                
                media_hist = dados_hist.mean()
                fig_hist.add_vline(
                    x=media_hist,
                    line_dash="dash",
                    line_color=COR_ALERTA,
                    line_width=2,
                    annotation_text=f"{t['mean']}: {media_hist:.2f}",
                    annotation_position="top",
                    annotation_font_color="white"
                )
                
                mediana_hist = dados_hist.median()
                fig_hist.add_vline(
                    x=mediana_hist,
                    line_dash="dot",
                    line_color=COR_SECUNDARIA,
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
                    title_font=dict(color=COR_PRIMARIA, size=16),
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
                    marker=dict(color=COR_PRIMARIA, size=8, opacity=0.7)
                ))
                
                fig_qq.add_trace(go.Scatter(
                    x=quantis_teoricos,
                    y=linha_ref(quantis_teoricos),
                    mode='lines',
                    name=f'Referência (R² = {r2:.3f})',
                    line=dict(color=COR_ALERTA, width=2)
                ))
                
                fig_qq.update_layout(
                    title=f"QQ Plot - {variavel_analise}",
                    plot_bgcolor='rgba(30, 41, 59, 0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color=COR_PRIMARIA, size=16),
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
            
            percentuais_valores = []
            for i in range(n_classes):
                limite_inferior = limites[i]
                limite_superior = limites[i+1]
                
                percentual_inferior = ((limite_inferior - minimo) / amplitude) * 100 if amplitude > 0 else 0
                percentual_superior = ((limite_superior - minimo) / amplitude) * 100 if amplitude > 0 else 100
                
                percentuais_valores.append(f"{percentual_inferior:.1f}% - {percentual_superior:.1f}%")
            
            freq_table.insert(1, 'Faixa Percentual (baseada nos valores)', percentuais_valores)
            
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
        
        # ================================================================
        # TAB 1: ESTATÍSTICAS & TEMPORAL
        # ================================================================
        with tabs[1]:
            st.markdown(f"<h3>{t['tab_temporal']}</h3>", unsafe_allow_html=True)
            
            df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
            
            valor_maximo = df_tempo[variavel_analise].max()
            valor_minimo = df_tempo[variavel_analise].min()
            minuto_maximo = extrair_minuto_do_extremo(df_tempo, variavel_analise, 'Minuto', 'max')
            minuto_minimo = extrair_minuto_do_extremo(df_tempo, variavel_analise, 'Minuto', 'min')
            media_tempo = df_tempo[variavel_analise].mean()
            limiar_75 = valor_maximo * 0.75
            limiar_50 = valor_maximo * 0.50
            
            eventos_acima_75 = (df_tempo[variavel_analise] >= limiar_75).sum()
            percentual_acima_75 = (eventos_acima_75 / len(df_tempo)) * 100 if len(df_tempo) > 0 else 0
            
            eventos_entre_50_75 = ((df_tempo[variavel_analise] >= limiar_50) & (df_tempo[variavel_analise] < limiar_75)).sum()
            percentual_entre_50_75 = (eventos_entre_50_75 / len(df_tempo)) * 100 if len(df_tempo) > 0 else 0
            
            def time_metric_card(label, valor, sub_label="", cor=COR_PRIMARIA):
                st.markdown(f"""
                <div class="time-metric-card" style="border-left-color: {cor};">
                    <div class="label">{label}</div>
                    <div class="value">{valor}</div>
                    <div class="sub-value">{sub_label}</div>
                </div>
                """, unsafe_allow_html=True)
            
            cols_t = st.columns(5)
            with cols_t[0]:
                time_metric_card(t['min_value'], f"{valor_minimo:.2f}", f"{t['minute_of_min']}: {minuto_minimo}", COR_SUCESSO)
            with cols_t[1]:
                time_metric_card(t['max_value'], f"{valor_maximo:.2f}", f"{t['minute_of_max']}: {minuto_maximo}", COR_ALERTA)
            with cols_t[2]:
                time_metric_card(t['mean'], f"{media_tempo:.2f}", t['mean'], COR_PRIMARIA)
            with cols_t[3]:
                time_metric_card(t['threshold_50'], f"{limiar_50:.2f}", f"50% do máx ({valor_maximo:.2f})", COR_SECUNDARIA)
            with cols_t[4]:
                time_metric_card(t['threshold_75'], f"{limiar_75:.2f}", f"75% do máx ({valor_maximo:.2f})", COR_ALERTA)
            
            st.markdown("---")
            
            col_intensity1, col_intensity2 = st.columns(2)
            with col_intensity1:
                def medium_card(titulo, valor, subtitulo, icone="📊"):
                    st.markdown(f"""
                    <div class="medium-card" style="background: linear-gradient(135deg, {COR_SECUNDARIA} 0%, #c97e00 100%);">
                        <div class="label">{icone} {titulo}</div>
                        <div class="value">{valor}</div>
                        <div class="sub-label">{subtitulo}</div>
                    </div>
                    """, unsafe_allow_html=True)
                medium_card(t['medium_high_intensity_events'], f"{eventos_entre_50_75}", f"{percentual_entre_50_75:.1f}% {t['between_50_75']}", "📊")
            with col_intensity2:
                def warning_card(titulo, valor, subtitulo, icone="⚠️"):
                    st.markdown(f"""
                    <div class="warning-card" style="background: linear-gradient(135deg, {COR_ALERTA} 0%, #b94c1c 100%);">
                        <div class="label">{icone} {titulo}</div>
                        <div class="value">{valor}</div>
                        <div class="sub-label">{subtitulo}</div>
                    </div>
                    """, unsafe_allow_html=True)
                warning_card(t['high_intensity_events'], f"{eventos_acima_75}", f"{percentual_acima_75:.1f}% {t['above_threshold_75']}", "⚡")
            
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
                cores_zonas = [COR_PRIMARIA, COR_DESTAQUE, COR_SECUNDARIA, COR_ALERTA, COR_SUCESSO]
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
            
            col_op1, col_op2 = st.columns([1, 3])
            
            with col_op1:
                opcoes_timeline = ["Gráfico único", "Comparar períodos"]
                idx_timeline = 0 if st.session_state.modo_timeline == 'unico' else 1
                
                modo_timeline = st.radio(
                    "Modo de visualização",
                    opcoes_timeline,
                    index=idx_timeline,
                    key="modo_timeline_radio",
                    on_change=atualizar_modo_timeline,
                    label_visibility="collapsed"
                )
            
            with col_op2:
                if st.session_state.modo_timeline == 'unico':
                    if len(periodos_selecionados) > 0:
                        fig_tempo = criar_timeline_unica_com_seletor(df_filtrado, variavel_analise, periodos_selecionados, t)
                        st.plotly_chart(fig_tempo, use_container_width=True)
                    else:
                        st.warning("Nenhum período selecionado")
                else:
                    if len(periodos_selecionados) > 1:
                        fig_multipla = criar_timeline_multipla(df_filtrado, variavel_analise, periodos_selecionados, t)
                        st.plotly_chart(fig_multipla, use_container_width=True)
                    elif len(periodos_selecionados) == 1:
                        st.info("Selecione mais de um período para comparação")
                        fig_tempo = criar_timeline_unica_com_seletor(df_filtrado, variavel_analise, periodos_selecionados, t)
                        st.plotly_chart(fig_tempo, use_container_width=True)
                    else:
                        st.warning("Nenhum período selecionado")
            
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
            
            col_ic1, col_ic2 = st.columns([1, 2])
            
            with col_ic1:
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
                    marker=dict(color=COR_PRIMARIA, size=20),
                    error_y=dict(type='constant', value=(ic_sup - media), color=COR_ALERTA, thickness=3, width=15),
                    name=t['mean']
                ))
                
                fig_ic.update_layout(
                    title=t['confidence_interval'],
                    plot_bgcolor='rgba(30, 41, 59, 0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=11),
                    title_font=dict(color=COR_PRIMARIA, size=14),
                    showlegend=False,
                    yaxis_title=variavel_analise
                )
                fig_ic.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                fig_ic.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                
                st.plotly_chart(fig_ic, use_container_width=True)
            
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
                
                styled_df = criar_tabela_destaque(df_resumo, ['Média', f'Máx {variavel_analise}', f'Mín {variavel_analise}'])
                
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
        
        # ================================================================
        # TAB 2: BOXPLOTS
        # ================================================================
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
                        marker_color=COR_PRIMARIA,
                        line_color='white',
                        fillcolor=f'rgba(0, 114, 178, 0.7)',
                        jitter=0.3,
                        pointpos=-1.8,
                        opacity=0.8
                    ))
            
            fig_box_pos.update_layout(
                title=f"{t['position']} - {variavel_analise}",
                plot_bgcolor='rgba(30, 41, 59, 0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=11),
                title_font=dict(color=COR_PRIMARIA, size=16),
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
                        marker_color=COR_DESTAQUE,
                        line_color='white',
                        fillcolor=f'rgba(204, 121, 167, 0.7)',
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
                title_font=dict(color=COR_PRIMARIA, size=16),
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
                    <h5 style="color: {COR_PRIMARIA};">{t['iqr_title']}</h5>
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
        
        # ================================================================
        # TAB 3: CORRELAÇÕES
        # ================================================================
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
                        color_continuous_scale=CORRELATION_COLORS,
                        title=f"{t['tab_correlation']}",
                        zmin=-1, zmax=1
                    )
                    
                    fig_corr.update_layout(
                        plot_bgcolor='rgba(30, 41, 59, 0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=11),
                        title_font=dict(color=COR_PRIMARIA, size=16),
                        height=500
                    )
                    fig_corr.update_xaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    fig_corr.update_yaxes(gridcolor='#334155', tickfont=dict(color='white'))
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.markdown(f"<h4>📊 {t['tab_correlation']}</h4>", unsafe_allow_html=True)
                    
                    def style_correlation(val):
                        if pd.isna(val):
                            return 'color: #94a3b8;'
                        if val == 1.0:
                            return 'color: #2d3748; font-weight: bold; background-color: #d3d3d3;'
                        if val < 0:
                            color = COR_ALERTA if abs(val) > 0.7 else COR_SECUNDARIA if abs(val) > 0.5 else COR_PRIMARIA
                        else:
                            color = COR_SUCESSO if val > 0.7 else COR_PRIMARIA if val > 0.5 else COR_PRIMARIA
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
                            color_discrete_sequence=[COR_PRIMARIA, COR_SECUNDARIA, COR_SUCESSO, COR_DESTAQUE, COR_ALERTA]
                        )
                        fig_scatter.update_layout(
                            plot_bgcolor='rgba(30, 41, 59, 0.8)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', size=11),
                            title_font=dict(color=COR_PRIMARIA, size=16),
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
        
        # ================================================================
        # TAB 4: K-MEANS CLUSTERS
        # ================================================================
        with tabs[4]:
            st.markdown(f"<h3>{t['tab_kmeans']}</h3>", unsafe_allow_html=True)
            
            st.session_state.kmeans_ativo = True
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        padding: 15px; border-radius: 12px; margin-bottom: 20px;
                        border-left: 4px solid {COR_DESTAQUE};">
                <p style="color: #94a3b8; margin: 0;">
                    <span style="color: {COR_DESTAQUE};">🎯 Segmentação de Atletas:</span> 
                    Selecione as variáveis e clique em PROCESSAR para gerar os clusters.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if len(st.session_state.variaveis_quantitativas) >= 2:
                
                if st.session_state.kmeans_var_x is None or st.session_state.kmeans_var_x not in st.session_state.variaveis_quantitativas:
                    st.session_state.kmeans_var_x = st.session_state.variaveis_quantitativas[0]
                
                vars_disponiveis = [v for v in st.session_state.variaveis_quantitativas 
                                   if v != st.session_state.kmeans_var_x]
                if st.session_state.kmeans_var_y is None or st.session_state.kmeans_var_y not in vars_disponiveis:
                    st.session_state.kmeans_var_y = vars_disponiveis[0] if vars_disponiveis else None
                
                col_k1, col_k2, col_k3, col_k4 = st.columns([2, 2, 1, 1])
                
                with col_k1:
                    var_x = st.selectbox(
                        "Variável do eixo X",
                        options=st.session_state.variaveis_quantitativas,
                        index=st.session_state.variaveis_quantitativas.index(st.session_state.kmeans_var_x),
                        key="kmeans_var_x_select"
                    )
                    if var_x != st.session_state.kmeans_var_x:
                        st.session_state.kmeans_var_x = var_x
                        st.session_state.kmeans_resultados = None
                        if st.session_state.kmeans_var_y == var_x:
                            novas_opcoes = [v for v in st.session_state.variaveis_quantitativas if v != var_x]
                            st.session_state.kmeans_var_y = novas_opcoes[0] if novas_opcoes else None
                
                with col_k2:
                    opcoes_y = [v for v in st.session_state.variaveis_quantitativas if v != var_x]
                    
                    current_y = st.session_state.kmeans_var_y
                    if current_y not in opcoes_y and opcoes_y:
                        current_y = opcoes_y[0]
                        st.session_state.kmeans_var_y = current_y
                    
                    var_y = st.selectbox(
                        "Variável do eixo Y",
                        options=opcoes_y,
                        index=opcoes_y.index(current_y) if current_y in opcoes_y else 0,
                        key="kmeans_var_y_select"
                    )
                    
                    if var_y != st.session_state.kmeans_var_y:
                        st.session_state.kmeans_var_y = var_y
                        st.session_state.kmeans_resultados = None
                
                with col_k3:
                    n_clusters = st.number_input(
                        "Nº Clusters",
                        min_value=2,
                        max_value=6,
                        value=st.session_state.kmeans_n_clusters,
                        step=1,
                        key="kmeans_n_clusters_input"
                    )
                    if n_clusters != st.session_state.kmeans_n_clusters:
                        st.session_state.kmeans_n_clusters = n_clusters
                        st.session_state.kmeans_resultados = None
                
                with col_k4:
                    st.markdown("<br>", unsafe_allow_html=True)
                    processar_click = st.button(
                        "🚀 PROCESSAR", 
                        type="primary", 
                        use_container_width=True,
                        key="kmeans_processar_btn"
                    )
                    
                    if processar_click:
                        with st.spinner('🔄 Calculando clusters...'):
                            try:
                                dados_cluster = df_filtrado[[var_x, var_y]].dropna()
                                
                                if len(dados_cluster) < n_clusters * 2:
                                    st.warning(f"⚠️ Dados insuficientes. Necessário pelo menos {n_clusters*2} observações.")
                                else:
                                    scaler = StandardScaler()
                                    dados_padronizados = scaler.fit_transform(dados_cluster)
                                    
                                    kmeans = KMeans(
                                        n_clusters=n_clusters, 
                                        random_state=42, 
                                        n_init=10
                                    )
                                    clusters = kmeans.fit_predict(dados_padronizados)
                                    
                                    df_cluster = dados_cluster.copy()
                                    df_cluster['Cluster'] = clusters
                                    df_cluster['Atleta'] = df_filtrado.loc[dados_cluster.index, 'Nome'].values
                                    df_cluster['Posição'] = df_filtrado.loc[dados_cluster.index, 'Posição'].values
                                    
                                    centroides = scaler.inverse_transform(kmeans.cluster_centers_)
                                    
                                    st.session_state.kmeans_resultados = {
                                        'df_cluster': df_cluster,
                                        'centroides': centroides,
                                        'var_x': var_x,
                                        'var_y': var_y,
                                        'n_clusters': n_clusters
                                    }
                                    
                            except Exception as e:
                                st.error(f"Erro na análise: {str(e)}")
                
                if st.session_state.kmeans_resultados is not None:
                    resultados = st.session_state.kmeans_resultados
                    df_cluster = resultados['df_cluster']
                    centroides = resultados['centroides']
                    var_x = resultados['var_x']
                    var_y = resultados['var_y']
                    n_clusters = resultados['n_clusters']
                    
                    fig_kmeans = go.Figure()
                    cores = [COR_PRIMARIA, COR_SECUNDARIA, COR_SUCESSO, COR_DESTAQUE, COR_ALERTA, COLORS['yellow']]
                    
                    for i in range(n_clusters):
                        dados_i = df_cluster[df_cluster['Cluster'] == i]
                        
                        fig_kmeans.add_trace(go.Scatter(
                            x=dados_i[var_x],
                            y=dados_i[var_y],
                            mode='markers',
                            name=f'Cluster {i+1}',
                            marker=dict(
                                size=12,
                                color=cores[i % len(cores)],
                                opacity=0.7,
                                line=dict(color='white', width=1)
                            ),
                            text=dados_i['Atleta'],
                            hovertemplate='<b>%{text}</b><br>' +
                                          f'{var_x}: %{{x:.2f}}<br>' +
                                          f'{var_y}: %{{y:.2f}}<br>' +
                                          f'Posição: %{{customdata}}<br>' +
                                          '<extra></extra>',
                            customdata=dados_i['Posição']
                        ))
                    
                    fig_kmeans.add_trace(go.Scatter(
                        x=centroides[:, 0],
                        y=centroides[:, 1],
                        mode='markers',
                        name='Centroides',
                        marker=dict(
                            size=20,
                            color='black',
                            symbol='x',
                            line=dict(color='white', width=3)
                        )
                    ))
                    
                    fig_kmeans.update_layout(
                        title=f'<b>Clusters: {var_x} vs {var_y}</b>',
                        xaxis_title=var_x,
                        yaxis_title=var_y,
                        plot_bgcolor='rgba(30,41,59,0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=12),
                        title_font=dict(color=COR_DESTAQUE, size=20),
                        height=600,
                        hovermode='closest',
                        legend=dict(
                            font=dict(color='white'),
                            bgcolor='rgba(30,41,59,0.8)',
                            bordercolor='#334155',
                            borderwidth=1
                        )
                    )
                    
                    fig_kmeans.update_xaxes(
                        gridcolor='#334155', 
                        tickfont=dict(color='white'),
                        title_font=dict(color='#94a3b8')
                    )
                    fig_kmeans.update_yaxes(
                        gridcolor='#334155', 
                        tickfont=dict(color='white'),
                        title_font=dict(color='#94a3b8')
                    )
                    
                    st.plotly_chart(fig_kmeans, use_container_width=True)
                    
                    st.markdown("### 📊 Perfil dos Clusters")
                    
                    stats_data = []
                    for i in range(n_clusters):
                        dados_i = df_cluster[df_cluster['Cluster'] == i]
                        
                        stats_data.append({
                            'Cluster': f'Cluster {i+1}',
                            '🎯 Atletas': len(dados_i['Atleta'].unique()),
                            '📊 Obs': len(dados_i),
                            f'📈 Média {var_x}': f'{dados_i[var_x].mean():.2f}',
                            f'📉 Média {var_y}': f'{dados_i[var_y].mean():.2f}',
                            '📍 Posições': ', '.join(dados_i['Posição'].unique()[:3]) + ('...' if len(dados_i['Posição'].unique()) > 3 else '')
                        })
                    
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True, hide_index=True)
                    
                    with st.expander("📋 Ver detalhamento dos atletas por cluster"):
                        for i in range(n_clusters):
                            dados_i = df_cluster[df_cluster['Cluster'] == i]
                            st.markdown(f"**Cluster {i+1}** ({len(dados_i)} observações)")
                            
                            atletas_cluster = dados_i.groupby('Atleta').agg({
                                var_x: 'mean',
                                var_y: 'mean',
                                'Posição': 'first'
                            }).reset_index()
                            
                            atletas_cluster.columns = ['Atleta', f'Média {var_x}', f'Média {var_y}', 'Posição']
                            st.dataframe(
                                atletas_cluster.sort_values(f'Média {var_x}', ascending=False),
                                use_container_width=True,
                                hide_index=True
                            )
                
                elif var_x and var_y:
                    st.info("👆 Clique em **PROCESSAR** para gerar a análise de clusters")
            
            else:
                st.info("ℹ️ São necessárias pelo menos 2 variáveis para análise de clusters")
        
        # ================================================================
        # TAB 5: COMPARADOR DE ATLETAS
        # ================================================================
        with tabs[5]:
            st.markdown(f"<h3>{t['tab_comparador']}</h3>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        padding: 20px; border-radius: 16px; margin-bottom: 25px;
                        border-left: 4px solid {COR_DESTAQUE};">
                <p style="color: #94a3b8; margin: 0;">
                    <span style="color: {COR_DESTAQUE}; font-size: 1.2rem;">🆚 Comparação Individual</span><br>
                    Compare o desempenho de um atleta com a média da sua posição e a média geral de todos os atletas.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if len(st.session_state.variaveis_quantitativas) >= 3 and len(atletas_selecionados) >= 1:
                
                col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
                
                with col_s1:
                    atleta_comp = st.selectbox(
                        "Selecione o Atleta",
                        options=atletas_selecionados,
                        key="comp_atleta"
                    )
                
                if not df_filtrado[df_filtrado['Nome'] == atleta_comp].empty:
                    posicao_atleta = df_filtrado[df_filtrado['Nome'] == atleta_comp]['Posição'].iloc[0]
                else:
                    posicao_atleta = None
                
                with col_s2:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.8); padding: 10px; border-radius: 8px; margin-top: 25px;">
                        <p style="color: #94a3b8; margin: 0;">Posição do Atleta</p>
                        <p style="color: white; font-size: 1.2rem; margin: 0;">{posicao_atleta}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_s3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    processar_comp = st.button(
                        "🔍 COMPARAR",
                        type="primary",
                        use_container_width=True,
                        key="comp_processar"
                    )
                
                vars_comp = st.multiselect(
                    "Selecione as variáveis para comparação (mínimo 3, máximo 8)",
                    options=st.session_state.variaveis_quantitativas,
                    default=st.session_state.variaveis_quantitativas[:min(5, len(st.session_state.variaveis_quantitativas))],
                    max_selections=8,
                    key="comp_vars"
                )
                
                if processar_comp and len(vars_comp) >= 3:
                    with st.spinner('🔄 Gerando comparação...'):
                        time.sleep(0.5)
                        
                        df_atleta = df_filtrado[df_filtrado['Nome'] == atleta_comp].copy()
                        df_posicao = df_filtrado[df_filtrado['Posição'] == posicao_atleta].copy()
                        df_geral = df_filtrado.copy()
                        
                        if not df_atleta.empty and not df_posicao.empty:
                            
                            fig_radar, valores_atleta, valores_posicao, valores_geral = criar_grafico_barras_desvio(
                                df_atleta, df_posicao, df_geral,
                                atleta_comp, posicao_atleta, vars_comp,
                                f"Comparação: {atleta_comp} vs Média da Posição vs Média Geral"
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
                            
                            st.markdown("### 📊 Tabela Comparativa Detalhada")
                            
                            dados_tabela = criar_tabela_comparativa(
                                atleta_comp, posicao_atleta, vars_comp,
                                valores_atleta, valores_posicao, valores_geral
                            )
                            
                            df_comp = pd.DataFrame(dados_tabela)
                            
                            def color_diff(val):
                                if '▲' in str(val):
                                    return f'color: {COR_SUCESSO}; font-weight: bold;'
                                elif '▼' in str(val):
                                    return f'color: {COR_ALERTA}; font-weight: bold;'
                                return 'color: #94a3b8;'
                            
                            styled_df = df_comp[['📊 Métrica', '🏃 Atleta', '📊 Média Posição', '📈 Média Geral', '🎯 vs Posição', '🌍 vs Geral']].style.applymap(
                                color_diff, subset=['🎯 vs Posição', '🌍 vs Geral']
                            )
                            
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            st.markdown("### 🏆 Destaques do Atleta")
                            
                            vantagens, desvantagens = criar_card_resumo(atleta_comp, posicao_atleta, dados_tabela)
                            
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                st.markdown(f"""
                                <div style="background: rgba(30, 41, 59, 0.8); padding: 20px; 
                                            border-radius: 16px; border-left: 4px solid {COR_SUCESSO};">
                                    <h4 style="color: {COR_SUCESSO}; margin-top: 0;">✅ Pontos Fortes</h4>
                                """, unsafe_allow_html=True)
                                
                                if vantagens:
                                    for metrica, pct in vantagens:
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                            <span style="color: white;">{metrica}</span>
                                            <span style="color: {COR_SUCESSO}; font-weight: bold;">+{pct:.1f}%</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown("<p style='color: #94a3b8;'>Nenhum destaque significativo</p>", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with col_d2:
                                st.markdown(f"""
                                <div style="background: rgba(30, 41, 59, 0.8); padding: 20px; 
                                            border-radius: 16px; border-left: 4px solid {COR_ALERTA};">
                                    <h4 style="color: {COR_ALERTA}; margin-top: 0;">⚠️ Pontos a Desenvolver</h4>
                                """, unsafe_allow_html=True)
                                
                                if desvantagens:
                                    for metrica, pct in desvantagens:
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                            <span style="color: white;">{metrica}</span>
                                            <span style="color: {COR_ALERTA}; font-weight: bold;">-{pct:.1f}%</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown("<p style='color: #94a3b8;'>Acima da média em todas as métricas!</p>", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("---")
                            st.markdown("### 💡 Insights Automáticos")
                            
                            performance_media = 0
                            count = 0
                            for i, v in enumerate(valores_atleta):
                                if valores_posicao[i] != 0:
                                    performance_media += (v - valores_posicao[i]) / valores_posicao[i]
                                    count += 1
                            
                            if count > 0:
                                performance_media = (performance_media / count) * 100
                                
                                if performance_media > 10:
                                    status = "EXCELENTE"
                                    cor = COR_SUCESSO
                                    icone = "🏆"
                                elif performance_media > 0:
                                    status = "ACIMA DA MÉDIA"
                                    cor = COR_PRIMARIA
                                    icone = "📈"
                                elif performance_media > -10:
                                    status = "NA MÉDIA"
                                    cor = COLORS['gray']
                                    icone = "📊"
                                else:
                                    status = "ABAIXO DA MÉDIA"
                                    cor = COR_ALERTA
                                    icone = "📉"
                                
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                                            padding: 20px; border-radius: 16px; margin-top: 20px;
                                            border-left: 6px solid {cor};">
                                    <div style="display: flex; align-items: center; gap: 20px;">
                                        <div style="font-size: 3rem;">{icone}</div>
                                        <div>
                                            <h4 style="color: white; margin: 0;">Desempenho Geral: <span style="color: {cor};">{status}</span></h4>
                                            <p style="color: #94a3b8; margin: 5px 0;">
                                                {atleta_comp} está {performance_media:+.1f}% acima da média da posição
                                            </p>
                                            <p style="color: #64748b; margin: 5px 0; font-size: 0.9rem;">
                                                { 'Destaque absoluto da posição!' if performance_media > 20 else 
                                                 'Bom desempenho, pode ser referência.' if performance_media > 10 else
                                                 'Desempenho sólido e consistente.' if performance_media > 0 else
                                                 'Necessita atenção em algumas áreas.' }
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("Não foi possível calcular a performance geral.")
                            
                        else:
                            st.warning("⚠️ Dados insuficientes para o atleta selecionado")
                
                elif processar_comp and len(vars_comp) < 3:
                    st.warning("⚠️ Selecione pelo menos 3 variáveis para uma comparação significativa")
            
            else:
                st.info("ℹ️ São necessários pelo menos 3 atletas e 3 variáveis para comparação")
        
        # ================================================================
        # TAB 6: ANÁLISE MBI
        # ================================================================
        with tabs[6]:
            st.markdown(f"<h3>🔬 {t['tab_mbi']}</h3>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        padding: 20px; border-radius: 16px; margin-bottom: 25px;
                        border-left: 4px solid {COR_DESTAQUE};">
                <p style="color: #94a3b8; margin: 0;">
                    <span style="color: {COR_DESTAQUE}; font-size: 1.2rem;">📊 Magnitude-Based Inference</span><br>
                    Baseado em Hopkins & Batterham (2006). Inferência sobre a importância clínica/prática das diferenças.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔥 Heatmap de Magnitudes (Z-Scores)")
            
            with st.expander("ℹ️ Sobre o Heatmap de Magnitudes"):
                st.markdown("""
                <p style="color: #94a3b8;">
                    O heatmap mostra o perfil de magnitudes de cada atleta em relação à média geral.
                    Valores positivos (laranja/vermelho) indicam desempenho acima da média, 
                    negativos (azul) abaixo da média. Esta visualização é acessível para daltônicos.
                </p>
                """, unsafe_allow_html=True)
            
            posicao_heatmap = st.selectbox(
                "Filtrar por posição (opcional)",
                options=["Todas as posições"] + list(posicoes_selecionadas),
                key="heatmap_posicao"
            )
            
            if posicao_heatmap == "Todas as posições":
                df_heatmap = df_filtrado
                pos_ref = None
            else:
                df_heatmap = df_filtrado[df_filtrado['Posição'] == posicao_heatmap]
                pos_ref = posicao_heatmap
            
            if len(df_heatmap) > 0:
                fig_heatmap = criar_heatmap_magnitude(df_heatmap, pos_ref)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown("### 📊 Análise Individual por Magnitude")
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                atleta_mbi = st.selectbox(
                    "Selecione o Atleta para análise MBI",
                    options=atletas_selecionados,
                    key="mbi_atleta"
                )
            
            with col_m2:
                referencia_mbi = st.selectbox(
                    "Referência de comparação",
                    options=["Média da Posição", "Média Geral"],
                    key="mbi_referencia"
                )
            
            df_atleta_mbi = df_filtrado[df_filtrado['Nome'] == atleta_mbi]
            
            if len(df_atleta_mbi) > 0:
                
                if referencia_mbi == "Média da Posição":
                    posicao_atleta = df_atleta_mbi['Posição'].iloc[0]
                    df_ref = df_filtrado[df_filtrado['Posição'] == posicao_atleta]
                    nome_ref = f"Média {posicao_atleta}"
                else:
                    df_ref = df_filtrado
                    nome_ref = "Média Geral"
                
                metricas_mbi = st.multiselect(
                    "Selecione as métricas para análise MBI",
                    options=st.session_state.variaveis_quantitativas,
                    default=st.session_state.variaveis_quantitativas[:3],
                    key="mbi_metricas"
                )
                
                if metricas_mbi and st.button("🔬 Calcular MBI", key="btn_mbi", use_container_width=True):
                    
                    resultados_mbi = []
                    
                    for metrica in metricas_mbi:
                        valor_atleta = df_atleta_mbi[metrica].mean()
                        valores_ref = df_ref[metrica].dropna()
                        
                        if len(valores_ref) > 1:
                            media_ref = valores_ref.mean()
                            desvio_ref = valores_ref.std()
                            n_ref = len(valores_ref)
                            
                            resultado = calcular_mbi(valor_atleta, media_ref, desvio_ref, n_ref)
                            
                            st.markdown(f"### 📈 {metrica}")
                            criar_card_mbi(resultado, atleta_mbi, metrica)
                            
                            resultados_mbi.append({
                                'Métrica': metrica,
                                'Atleta': f'{valor_atleta:.2f}',
                                'Referência': f'{media_ref:.2f}',
                                'Cohen\'s d': f'{resultado["cohen_d"]:.2f}',
                                'Magnitude': resultado['magnitude'],
                                'Inferência': resultado['inferencia'],
                                'IC 90%': f'[{resultado["ic_90"][0]:.2f}, {resultado["ic_90"][1]:.2f}]'
                            })
                    
                    if resultados_mbi:
                        st.markdown("### 📋 Resumo das Análises MBI")
                        df_resultados = pd.DataFrame(resultados_mbi)
                        st.dataframe(df_resultados, use_container_width=True, hide_index=True)
        
        # ================================================================
        # TAB 7: INDIVIDUAL
        # ================================================================
        with tabs[7]:
            st.markdown(f"<h3>{t['tab_individual']}</h3>", unsafe_allow_html=True)
            
            atleta_individual = st.selectbox(
                t['athlete'],
                options=atletas_selecionados,
                key="individual_atleta"
            )
            
            if atleta_individual:
                df_atleta = df_filtrado[df_filtrado['Nome'] == atleta_individual].copy()
                
                if not df_atleta.empty:
                    posicao_atleta = df_atleta['Posição'].iloc[0] if 'Posição' in df_atleta.columns else "N/A"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px; border-radius: 16px; margin-bottom: 20px;">
                        <h2 style="margin: 0;">{atleta_individual}</h2>
                        <p style="color: #94a3b8; margin: 5px 0;">{t['position']}: {posicao_atleta}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Métricas do atleta
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    
                    with col_met1:
                        media_atleta = df_atleta[variavel_analise].mean()
                        st.metric(t['mean'], f"{media_atleta:.2f}")
                    
                    with col_met2:
                        max_atleta = df_atleta[variavel_analise].max()
                        st.metric(t['max'], f"{max_atleta:.2f}")
                    
                    with col_met3:
                        min_atleta = df_atleta[variavel_analise].min()
                        st.metric(t['min'], f"{min_atleta:.2f}")
                    
                    with col_met4:
                        cv_atleta = calcular_cv(media_atleta, df_atleta[variavel_analise].std())
                        st.metric(t['cv'], f"{cv_atleta:.1f}%")
                    
                    st.markdown("---")
                    st.markdown(f"### {t['performance_history']}")
                    
                    # Gráfico de evolução do atleta
                    fig_atleta = go.Figure()
                    
                    fig_atleta.add_trace(go.Scatter(
                        x=df_atleta['Minuto'],
                        y=df_atleta[variavel_analise],
                        mode='lines+markers',
                        name=variavel_analise,
                        line=dict(color=COR_PRIMARIA, width=2),
                        marker=dict(size=8, color=COR_PRIMARIA)
                    ))
                    
                    fig_atleta.update_layout(
                        title=f"{variavel_analise} - Evolução Temporal",
                        xaxis_title="Minuto",
                        yaxis_title=variavel_analise,
                        plot_bgcolor='rgba(30,41,59,0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=12),
                        height=400
                    )
                    
                    st.plotly_chart(fig_atleta, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown(f"### {t['weekly_summary']}")
                    
                    # Resumo por período
                    if 'Período' in df_atleta.columns:
                        resumo_periodos = df_atleta.groupby('Período')[variavel_analise].agg(['mean', 'std', 'min', 'max']).round(2)
                        resumo_periodos.columns = [t['mean'], t['std'], t['min'], t['max']]
                        st.dataframe(resumo_periodos, use_container_width=True)
        
        # ================================================================
        # TAB 8: TENDÊNCIAS ML
        # ================================================================
        with tabs[8]:
            st.markdown(f"<h3>{t['tab_trends']}</h3>", unsafe_allow_html=True)
            
            st.markdown("### 📈 Previsão de Desempenho com Machine Learning")
            
            atleta_forecast = st.selectbox(
                t['athlete'],
                options=atletas_selecionados,
                key="forecast_atleta"
            )
            
            if atleta_forecast:
                df_atleta = df_filtrado[df_filtrado['Nome'] == atleta_forecast].copy()
                
                if len(df_atleta) >= 10:
                    periods_ahead = st.slider("Períodos a prever", 1, 10, 5)
                    
                    if st.button("🔮 Gerar Previsão", key="forecast_btn"):
                        with st.spinner('🔄 Calculando previsões...'):
                            previsoes, intervalos, r2 = forecast_performance(df_atleta, variavel_analise, periods_ahead)
                            
                            if previsoes:
                                fig_forecast = criar_grafico_forecast(df_atleta, variavel_analise, previsoes, intervalos, t)
                                st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                st.markdown(f"""
                                <div class="metric-container">
                                    <h4>📊 Qualidade do Modelo</h4>
                                    <p><strong>R² Score:</strong> {r2:.3f}</p>
                                    <p><strong>Interpretação:</strong> {'Boa capacidade preditiva' if r2 > 0.7 else 'Capacidade preditiva moderada' if r2 > 0.5 else 'Baixa capacidade preditiva'}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Tabela de previsões
                                df_previsoes = pd.DataFrame({
                                    'Período': [f"+{i+1}" for i in range(len(previsoes))],
                                    'Previsão': [f"{p:.2f}" for p in previsoes],
                                    'IC Inferior (95%)': [f"{i[0]:.2f}" for i in intervalos],
                                    'IC Superior (95%)': [f"{i[1]:.2f}" for i in intervalos]
                                })
                                
                                st.markdown("### 📋 Previsões Detalhadas")
                                st.dataframe(df_previsoes, use_container_width=True, hide_index=True)
                            else:
                                st.warning("Dados insuficientes para gerar previsões confiáveis. Necessário pelo menos 10 observações.")
                else:
                    st.info(f"Dados insuficientes para previsão. Necessário pelo menos 10 observações. Atualmente: {len(df_atleta)} observações.")
        
        # ================================================================
        # TAB 9: EXECUTIVO
        # ================================================================
        with tabs[9]:
            st.markdown(f"<h3>{t['tab_executive']}</h3>", unsafe_allow_html=True)
            
            st.markdown("### 🆚 Comparação de Atletas")
            if len(atletas_selecionados) >= 2:
                col_atl1, col_atl2 = st.columns(2)
                with col_atl1:
                    atleta1_comp = st.selectbox(
                        "Atleta 1", 
                        atletas_selecionados, 
                        index=0, 
                        key="exec_atleta1"
                    )
                with col_atl2:
                    atleta2_comp = st.selectbox(
                        "Atleta 2", 
                        atletas_selecionados, 
                        index=min(1, len(atletas_selecionados)-1), 
                        key="exec_atleta2"
                    )
                
                if atleta1_comp != atleta2_comp:
                    vars_comp = st.multiselect(
                        "Variáveis para comparar",
                        st.session_state.variaveis_quantitativas,
                        default=st.session_state.variaveis_quantitativas[:3],
                        key="exec_vars"
                    )
                    
                    if len(vars_comp) >= 1:
                        comparar_atletas(df_filtrado, atleta1_comp, atleta2_comp, vars_comp, t)
                else:
                    st.info("Selecione atletas diferentes para comparação")
            else:
                st.info("Selecione pelo menos 2 atletas para comparação")
            
            st.markdown("---")
            
            sistema_anotacoes(t)
        
        # Dados brutos
        with st.expander("📋 Visualizar dados brutos filtrados"):
            st.dataframe(df_filtrado, use_container_width=True)
    
    else:
        st.info("👈 Selecione os filtros e clique em Processar Análise na barra lateral")

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

# ============================================================================
# RODAPÉ COM REFERÊNCIAS ACADÊMICAS
# ============================================================================

st.markdown(f"""
<div class="references-footer">
    <h4>📚 Referências Acadêmicas</h4>
    <p>
        <strong>Batterham, A. M., & Hopkins, W. G. (2006).</strong> Making meaningful inferences about magnitudes. 
        <em>International Journal of Sports Physiology and Performance</em>, 1(1), 50-57.
    </p>
    <p>
        <strong>Hopkins, W. G., Marshall, S. W., Batterham, A. M., & Hanin, J. (2009).</strong> 
        Progressive statistics for studies in sports medicine and exercise science. 
        <em>Medicine & Science in Sports & Exercise</em>, 41(1), 3-12.
    </p>
    <p>
        <strong>Cohen, J. (1988).</strong> <em>Statistical power analysis for the behavioral sciences</em> (2nd ed.). 
        Lawrence Erlbaum Associates.
    </p>
    <p>
        <strong>Okabe, M., & Ito, K. (2008).</strong> Color universal design (CUD) - How to make figures and presentations 
        that are friendly to colorblind people. <em>J*Fly</em>
    </p>
</div>
""", unsafe_allow_html=True)