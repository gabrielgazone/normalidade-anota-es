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

# Instalar scikit-learn se não estiver disponível
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Sports Science Analytics Pro - Academic Edition", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ============================================================================
# CSS PERSONALIZADO - ESTILO ACADÊMICO (Tópicos 7 e 8)
# ============================================================================

st.markdown("""
<style>
    /* Tema científico - fundo branco, texto escuro */
    .stApp {
        background: #ffffff !important;
    }
    
    /* Sidebar elegante - tons de cinza */
    .css-1d391kg, .css-1wrcr25 {
        background: #f8f9fa !important;
        border-right: 1px solid #dee2e6;
    }
    
    /* Cards para resultados científicos */
    .scientific-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    
    .scientific-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .scientific-card .label {
        color: #6c757d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .scientific-card .value {
        color: #212529;
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    
    .scientific-card .inference {
        font-size: 1rem;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px 0;
    }
    
    /* Títulos no formato acadêmico */
    h1 {
        color: #212529 !important;
        font-family: 'Times New Roman', serif;
        font-weight: 700;
        border-bottom: 2px solid #212529;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #343a40 !important;
        font-family: 'Times New Roman', serif;
        font-weight: 600;
    }
    
    h3 {
        color: #495057 !important;
        font-family: 'Times New Roman', serif;
        font-weight: 500;
    }
    
    /* Abas no estilo acadêmico */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #f8f9fa;
        padding: 5px;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
        color: #495057 !important;
        font-family: 'Times New Roman', serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: #212529 !important;
        color: white !important;
    }
    
    /* Tabelas no formato APA */
    .dataframe {
        font-family: 'Times New Roman', serif;
        border: 1px solid #dee2e6;
    }
    
    .dataframe th {
        background: #f8f9fa;
        color: #212529;
        font-weight: 600;
        border-bottom: 2px solid #212529;
    }
    
    .dataframe td {
        color: #212529;
        border: 1px solid #dee2e6;
    }
    
    /* Botões estilo acadêmico */
    .stButton > button {
        background: white;
        color: #212529;
        border: 1px solid #212529;
        border-radius: 4px;
        font-family: 'Times New Roman', serif;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #212529;
        color: white;
        border: 1px solid #212529;
    }
    
    /* Header científico */
    .scientific-header {
        background: white;
        padding: 25px;
        border-bottom: 3px solid #212529;
        margin-bottom: 30px;
    }
    
    /* Rodapé acadêmico */
    .academic-footer {
        background: #f8f9fa;
        padding: 20px;
        border-top: 1px solid #dee2e6;
        margin-top: 50px;
        font-family: 'Times New Roman', serif;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INTERNACIONALIZAÇÃO
# ============================================================================

translations = {
    'pt': {
        'title': 'Sports Science Analytics Pro - Edição Acadêmica',
        'subtitle': 'Dashboard Científico para Análise de Desempenho Esportivo',
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
        'tab_export': '📋 Exportação APA',
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
        'below_threshold': 'abaixo do limiar de 80%',
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
        ''',
        'select_period_timeline': 'Selecione o período para visualização temporal',
        'all_periods': 'Todos os períodos (gráfico único)',
        'compare_periods': 'Comparar períodos (múltiplos gráficos)'
    },
    'en': {
        'title': 'Sports Science Analytics Pro - Academic Edition',
        'subtitle': 'Scientific Dashboard for Sports Performance Analysis',
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
        'tab_export': '📋 APA Export',
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
        'below_threshold': 'below 80% threshold',
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
        ''',
        'select_period_timeline': 'Select period for temporal visualization',
        'all_periods': 'All periods (single chart)',
        'compare_periods': 'Compare periods (multiple charts)'
    },
    'es': {
        'title': 'Sports Science Analytics Pro - Edición Académica',
        'subtitle': 'Dashboard Científico para Análisis de Rendimiento Deportivo',
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
        'tab_export': '📋 Exportación APA',
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
        'below_threshold': 'por debajo del umbral 80%',
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
        ''',
        'select_period_timeline': 'Seleccione el período para visualización temporal',
        'all_periods': 'Todos los períodos (gráfico único)',
        'compare_periods': 'Comparar períodos (múltiples gráficos)'
    }
}

# ============================================================================
# FUNÇÕES PARA MAGNITUDE-BASED INFERENCE (MBI) - Tópico 1
# ============================================================================

def calcular_mbi(valor_atleta, media_grupo, desvio_grupo, n_grupo=30, small_effect=0.2):
    """
    Implementa Magnitude-Based Inference segundo Hopkins & Batterham (2006)
    Retorna: inferência qualitativa, intervalo de confiança 90% e probabilidades
    """
    from scipy import stats
    import numpy as np
    
    # Calcular diferença e tamanho do efeito (Cohen's d)
    diferenca = valor_atleta - media_grupo
    cohen_d = diferenca / desvio_grupo if desvio_grupo != 0 else 0
    
    # Limiares para magnitudes (Hopkins, 2006)
    limiar_pequeno = small_effect * desvio_grupo
    limiar_moderado = 0.6 * desvio_grupo
    limiar_grande = 1.2 * desvio_grupo
    
    # Classificar magnitude
    if abs(cohen_d) < 0.2:
        magnitude = "trivial"
    elif abs(cohen_d) < 0.6:
        magnitude = "pequena"
    elif abs(cohen_d) < 1.2:
        magnitude = "moderada"
    else:
        magnitude = "grande"
    
    # Intervalo de confiança 90% (padrão MBI)
    erro_padrao = desvio_grupo / np.sqrt(n_grupo) if n_grupo > 0 else 0
    
    ic_inf = diferenca - 1.645 * erro_padrao  # 90% CI
    ic_sup = diferenca + 1.645 * erro_padrao
    
    # Classificação MBI - Regras de decisão (Hopkins & Batterham, 2006)
    if ic_inf > limiar_pequeno:
        inferencia = "Muito provavelmente benéfico"
        cor = "#2ca02c"  # Verde
        prob = 0.95
        icone = "✅"
    elif ic_sup < -limiar_pequeno:
        inferencia = "Muito provavelmente prejudicial"
        cor = "#d62728"  # Vermelho
        prob = 0.95
        icone = "❌"
    elif ic_inf > -limiar_pequeno and ic_sup < limiar_pequeno:
        inferencia = "Quase certamente trivial"
        cor = "#7f7f7f"  # Cinza
        prob = 0.90
        icone = "➖"
    elif ic_inf > -limiar_pequeno and ic_sup > limiar_pequeno and ic_inf < limiar_pequeno:
        # Intervalo cruza o limiar positivo
        if abs(cohen_d) < 0.2:
            inferencia = "Pouco claro (possivelmente trivial/benéfico)"
        else:
            inferencia = "Possivelmente benéfico"
        cor = "#ff7f0e"  # Laranja
        prob = 0.75
        icone = "⚠️"
    elif ic_inf < -limiar_pequeno and ic_sup < -limiar_pequeno and ic_sup > -limiar_pequeno:
        inferencia = "Possivelmente prejudicial"
        cor = "#ff7f0e"
        prob = 0.75
        icone = "⚠️"
    else:
        inferencia = "Não claro (necessário mais dados)"
        cor = "#1f77b4"  # Azul
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
    """Card estilizado para apresentar resultados MBI"""
    
    st.markdown(f"""
    <div class="scientific-card" style="border-left: 8px solid {resultado['cor']};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="color: #212529; margin: 0;">{var_nome}</h4>
                <p style="color: #212529; font-size: 1.2rem; margin: 5px 0;">
                    <strong>{resultado['inferencia']}</strong>
                </p>
                <p style="color: #6c757d; margin: 5px 0;">
                    Cohen's d = {resultado['cohen_d']:.2f} 
                    <span style="background: {resultado['cor']}; color: white; padding: 2px 8px; border-radius: 12px; margin-left: 10px;">
                        {resultado['magnitude']}
                    </span>
                </p>
                <p style="color: #6c757d; font-size: 0.9rem; margin: 5px 0;">
                    IC 90%: [{resultado['ic_90'][0]:.2f}, {resultado['ic_90'][1]:.2f}]
                </p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">{resultado['icone']}</div>
                <p style="color: #212529; margin: 0;">{resultado['probabilidade']*100:.0f}%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def criar_heatmap_magnitude(df, posicao_referencia=None):
    """
    Heatmap mostrando magnitudes dos atletas vs referência - Tópico 2
    Inspirado no NFL Combine IQ
    """
    
    # Selecionar métricas numéricas
    metricas = df.select_dtypes(include=[np.number]).columns.tolist()
    metricas = [m for m in metricas if m not in ['Minuto']]  # Excluir colunas não-métricas
    
    # Limitar para não poluir (máx 10 métricas)
    if len(metricas) > 10:
        metricas = metricas[:10]
    
    # Calcular z-scores (magnitudes) para cada atleta
    dados_heatmap = []
    for atleta in df['Nome'].unique():
        df_atleta = df[df['Nome'] == atleta]
        linha = {'Atleta': atleta}
        
        for metrica in metricas:
            if metrica in df.columns:
                valor = df_atleta[metrica].mean()
                media_grupo = df[metrica].mean()
                desvio_grupo = df[metrica].std()
                
                # Z-score (magnitude normalizada)
                z_score = (valor - media_grupo) / desvio_grupo if desvio_grupo != 0 else 0
                linha[metrica] = z_score
        
        dados_heatmap.append(linha)
    
    df_heat = pd.DataFrame(dados_heatmap).set_index('Atleta')
    
    # Criar heatmap com escala de magnitude
    fig = px.imshow(
        df_heat.T,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title=f"Perfil de Magnitudes (Z-Scores) - {'Todas Posições' if not posicao_referencia else posicao_referencia}",
        labels=dict(x="Atleta", y="Métrica", color="Z-Score")
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529', family="Times New Roman"),
        height=500,
        width=900,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        coloraxis_colorbar=dict(
            title="Magnitude",
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=['Muito Baixo (-2σ)', 'Baixo (-1σ)', 'Média (0σ)', 'Alto (+1σ)', 'Muito Alto (+2σ)'],
            title_font=dict(size=11),
            tickfont=dict(size=10)
        )
    )
    
    # Adicionar anotações para os valores
    fig.update_traces(textfont=dict(size=9, color='black'))
    
    return fig

# ============================================================================
# FUNÇÕES AUXILIARES EXISTENTES (mantidas do código anterior)
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

def executive_card(titulo, valor, delta, icone, cor_status="#1f77b4"):
    delta_icon = "▲" if delta > 0 else "▼"
    delta_color = "#2ca02c" if delta > 0 else "#d62728"
    
    st.markdown(f"""
    <div class="scientific-card" style="border-left-color: {cor_status};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p class="label">{titulo}</p>
                <p class="value">{valor}</p>
                <p class="delta" style="color: {delta_color};">
                    {delta_icon} {abs(delta):.1f}% vs. média
                </p>
            </div>
            <div style="font-size: 2.5rem;">{icone}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def time_metric_card(label, valor, sub_label="", cor="#1f77b4"):
    st.markdown(f"""
    <div class="scientific-card" style="border-left-color: {cor}; padding: 15px;">
        <p class="label">{label}</p>
        <p class="value" style="font-size: 1.6rem;">{valor}</p>
        <p style="color: #6c757d; margin: 0;">{sub_label}</p>
    </div>
    """, unsafe_allow_html=True)

def warning_card(titulo, valor, subtitulo, icone="⚠️"):
    st.markdown(f"""
    <div class="scientific-card" style="border-left-color: #d62728;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="font-size: 2rem;">{icone}</div>
            <div>
                <p class="label">{titulo}</p>
                <p class="value" style="font-size: 1.8rem;">{valor}</p>
                <p style="color: #6c757d;">{subtitulo}</p>
            </div>
        </div>
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

def criar_timeline_profissional(df, variavel, t):
    fig = go.Figure()
    
    media_movevel = df[variavel].rolling(window=5, min_periods=1).mean()
    valor_maximo = df[variavel].max()
    limiar_80 = valor_maximo * 0.8
    
    acima_limiar = df[variavel] > limiar_80
    abaixo_limiar = df[variavel] <= limiar_80
    
    fig.add_hrect(
        y0=limiar_80,
        y1=valor_maximo * 1.05,
        fillcolor="rgba(214, 39, 40, 0.15)",
        line_width=0,
        layer="below",
        name=f"{t['above_threshold']}"
    )
    
    fig.add_hrect(
        y0=0,
        y1=limiar_80,
        fillcolor="rgba(31, 119, 180, 0.1)",
        line_width=0,
        layer="below",
        name=f"{t['below_threshold']}"
    )
    
    fig.add_hline(
        y=limiar_80,
        line_dash="solid",
        line_color="#d62728",
        line_width=2,
        annotation_text=f"🔴 {t['threshold_80']}: {limiar_80:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="#212529", size=11)
    )
    
    df_acima = df[acima_limiar].copy()
    df_abaixo = df[abaixo_limiar].copy()
    
    if not df_acima.empty:
        fig.add_trace(go.Scatter(
            x=df_acima['Minuto'],
            y=df_acima[variavel],
            mode='markers',
            name=t['above_threshold'],
            marker=dict(
                size=10,
                color='#d62728',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>Minuto:</b> %{x}<br>' +
                          '<b>Valor:</b> %{y:.2f} (ACIMA DO LIMIAR)<extra></extra>'
        ))
    
    if not df_abaixo.empty:
        fig.add_trace(go.Scatter(
            x=df_abaixo['Minuto'],
            y=df_abaixo[variavel],
            mode='markers',
            name=t['below_threshold'],
            marker=dict(
                size=8,
                color='#1f77b4',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>Minuto:</b> %{x}<br>' +
                          '<b>Valor:</b> %{y:.2f}<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=df['Minuto'],
        y=media_movevel,
        mode='lines',
        name='Média Móvel (5)',
        line=dict(color='#ff7f0e', width=2, dash='dot')
    ))
    
    media = df[variavel].mean()
    desvio = df[variavel].std()
    
    fig.add_hline(
        y=media, 
        line_dash="dash", 
        line_color="#7f7f7f",
        annotation_text=f"Média: {media:.2f}", 
        annotation_position="top left",
        annotation_font=dict(color="#212529")
    )
    
    fig.add_hrect(
        y0=media-desvio, 
        y1=media+desvio,
        fillcolor="#1f77b4", 
        opacity=0.1, 
        line_width=0,
        annotation_text="±1 DP",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Evolução de {variavel} - Áreas: Azul (abaixo do limiar) | Vermelho (acima do limiar)",
        xaxis_title="Minuto",
        yaxis_title=variavel,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529', size=12, family="Times New Roman"),
        title_font=dict(color='#212529', size=18),
        showlegend=True,
        legend=dict(
            font=dict(color='#212529'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#dee2e6',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
    fig.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
    
    return fig

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
                return ['background-color: rgba(44, 160, 44, 0.2)'] * len(row)
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
            cor = "#2ca02c" if delta > 0 else "#d62728"
            st.markdown(f"""
            <div class="scientific-card" style="padding: 10px; border-left: 3px solid {cor};">
                <span style="color: #6c757d;">{var}:</span>
                <span style="color: #212529; font-weight: bold; float: right;">{dados1[var]:.2f}</span>
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
            cor = "#2ca02c" if delta > 0 else "#d62728"
            st.markdown(f"""
            <div class="scientific-card" style="padding: 10px; border-left: 3px solid {cor};">
                <span style="color: #6c757d;">{var}:</span>
                <span style="color: #212529; font-weight: bold; float: right;">{dados2[var]:.2f}</span>
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
            <div class="scientific-card" style="padding: 10px;">
                <p style="color: #6c757d; margin: 0;">{anotacao['data']}</p>
                <p style="color: #212529; margin: 5px 0;">{anotacao['texto']}</p>
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
        limiar_80 = valor_maximo * 0.8
        
        fig.add_trace(
            go.Scatter(
                x=df_periodo['Minuto'],
                y=df_periodo[variavel],
                mode='lines+markers',
                name=f'Período {periodo}',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6, color='#1f77b4'),
                showlegend=False
            ),
            row=i, col=1
        )
        
        fig.add_hline(
            y=limiar_80,
            line_dash="dash",
            line_color="#d62728",
            line_width=1,
            row=i, col=1
        )
        
        media_periodo = df_periodo[variavel].mean()
        fig.add_hline(
            y=media_periodo,
            line_dash="dot",
            line_color="#2ca02c",
            line_width=1,
            row=i, col=1
        )
        
        fig.update_xaxes(
            title_text="Minuto" if i == n_periodos else "",
            gridcolor='#e1e5e9',
            tickfont=dict(color='#212529', size=9),
            tickangle=-45,
            row=i, col=1
        )
        
        fig.update_yaxes(
            title_text=variavel if i == n_periodos//2 + 1 else "",
            gridcolor='#e1e5e9',
            tickfont=dict(color='#212529'),
            row=i, col=1
        )
    
    fig.update_layout(
        title=f"Evolução Temporal por Período - {variavel}",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529', size=11, family="Times New Roman"),
        title_font=dict(color='#212529', size=18),
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
        cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
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
        limiar_80 = valor_maximo * 0.8
        media = df_plot[variavel].mean()
        desvio = df_plot[variavel].std()
        
        fig.add_hrect(
            y0=limiar_80,
            y1=valor_maximo * 1.05,
            fillcolor="rgba(214, 39, 40, 0.15)",
            line_width=0,
            layer="below",
            name="Acima do limiar"
        )
        
        fig.add_hrect(
            y0=0,
            y1=limiar_80,
            fillcolor="rgba(31, 119, 180, 0.1)",
            line_width=0,
            layer="below",
            name="Abaixo do limiar"
        )
        
        fig.add_hline(
            y=limiar_80,
            line_dash="solid",
            line_color="#d62728",
            line_width=2,
            annotation_text=f"🔴 Limiar 80%: {limiar_80:.2f}",
            annotation_position="top left"
        )
        
        fig.add_hline(
            y=media,
            line_dash="dash",
            line_color="#7f7f7f",
            annotation_text=f"Média: {media:.2f}",
            annotation_position="top left"
        )
        
        fig.add_hrect(
            y0=media-desvio,
            y1=media+desvio,
            fillcolor="#1f77b4",
            opacity=0.1,
            line_width=0,
            annotation_text="±1 DP"
        )
        
        fig.add_trace(go.Scatter(
            x=df_plot['Minuto'],
            y=df_plot[variavel],
            mode='lines+markers',
            name=variavel,
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8, color='#1f77b4', line=dict(color='white', width=1)),
            hovertemplate='<b>Minuto:</b> %{x}<br>' +
                          '<b>Valor:</b> %{y:.2f}<extra></extra>'
        ))
        
        media_movevel = df_plot[variavel].rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_plot['Minuto'],
            y=media_movevel,
            mode='lines',
            name='Média Móvel (5)',
            line=dict(color='#ff7f0e', width=2, dash='dot')
        ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Minuto",
        yaxis_title=variavel,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529', size=12, family="Times New Roman"),
        title_font=dict(color='#212529', size=18),
        hovermode='closest',
        legend=dict(
            font=dict(color='#212529'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#dee2e6'
        )
    )
    
    fig.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'), tickangle=-45)
    fig.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
    
    return fig

def criar_grafico_barras_desvio(df_atleta, df_posicao, df_geral, atleta_nome, posicao, variaveis, titulo="Comparação de Desempenho"):
    """
    Gráfico de barras mostrando o desvio percentual do atleta em relação às médias
    """
    
    # Calcular valores
    valores_atleta = [df_atleta[var].mean() for var in variaveis]
    valores_posicao = [df_posicao[var].mean() for var in variaveis]
    valores_geral = [df_geral[var].mean() for var in variaveis]
    
    # Calcular desvios percentuais
    desvios_vs_posicao = [((v - valores_posicao[i]) / valores_posicao[i]) * 100 if valores_posicao[i] != 0 else 0 
                          for i, v in enumerate(valores_atleta)]
    desvios_vs_geral = [((v - valores_geral[i]) / valores_geral[i]) * 100 if valores_geral[i] != 0 else 0 
                        for i, v in enumerate(valores_atleta)]
    
    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"<b>Valores Absolutos</b>", 
            f"<b>Desvio Percentual do Atleta vs Médias</b>"
        ),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Gráfico superior - Valores absolutos
    fig.add_trace(go.Bar(
        x=variaveis,
        y=valores_atleta,
        name=atleta_nome,
        marker_color='#1f77b4',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.9,
        text=[f'{v:.1f}' for v in valores_atleta],
        textposition='outside',
        textfont=dict(color='#212529', size=11, family="Times New Roman"),
        hovertemplate='<b>%{x}</b><br>' +
                      f'{atleta_nome}: %{{y:.2f}}<br>' +
                      '<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=variaveis,
        y=valores_posicao,
        name=f'Média {posicao}',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=10, color='#ff7f0e', symbol='diamond'),
        hovertemplate='<b>%{x}</b><br>' +
                      f'Média {posicao}: %{{y:.2f}}<br>' +
                      '<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=variaveis,
        y=valores_geral,
        name='Média Geral',
        mode='lines+markers',
        line=dict(color='#7f7f7f', width=3, dash='dot'),
        marker=dict(size=10, color='#7f7f7f', symbol='circle'),
        hovertemplate='<b>%{x}</b><br>' +
                      'Média Geral: %{y:.2f}<br>' +
                      '<extra></extra>'
    ), row=1, col=1)
    
    # Gráfico inferior - Desvios percentuais
    cores_desvio = ['#2ca02c' if d > 0 else '#d62728' if d < 0 else '#7f7f7f' for d in desvios_vs_posicao]
    
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
        textfont=dict(color='#212529', size=10, family="Times New Roman"),
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
            color='#9467bd',
            symbol='star',
            line=dict(color='white', width=1)
        ),
        text=[f'{d:+.1f}%' for d in desvios_vs_geral],
        textposition='top center',
        textfont=dict(color='#212529', size=9),
        hovertemplate='<b>%{x}</b><br>' +
                      'vs Geral: %{y:+.1f}%<br>' +
                      '<extra></extra>'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color="#7f7f7f", line_width=1, opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{titulo}</b>",
            font=dict(size=24, color='#212529', family="Times New Roman"),
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529', family="Times New Roman"),
        height=800,
        width=900,
        showlegend=True,
        legend=dict(
            font=dict(color='#212529', size=11),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#dee2e6',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    fig.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529', size=11), row=1, col=1)
    fig.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529', size=11), row=2, col=1)
    
    fig.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'), title="Valor", row=1, col=1)
    fig.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'), title="Desvio (%)", row=2, col=1)
    
    return fig, valores_atleta, valores_posicao, valores_geral

def criar_tabela_comparativa(atleta_nome, posicao, variaveis, valores_atleta, valores_posicao, valores_geral):
    """
    Cria uma tabela comparativa com diferenças percentuais
    """
    dados = []
    
    for i, var in enumerate(variaveis):
        val_atleta = valores_atleta[i]
        val_posicao = valores_posicao[i]
        val_geral = valores_geral[i]
        
        # Calcular diferenças percentuais
        diff_vs_posicao = ((val_atleta - val_posicao) / val_posicao) * 100 if val_posicao != 0 else 0
        diff_vs_geral = ((val_atleta - val_geral) / val_geral) * 100 if val_geral != 0 else 0
        
        # Determinar ícones e cores
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
    """
    Cria cards de resumo com os destaques do atleta
    """
    # Encontrar maiores diferenças
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

def calcular_ic_95(dados):
    """Calcula intervalo de confiança de 95% para a média"""
    from scipy import stats
    import numpy as np
    
    if len(dados) < 2:
        return np.mean(dados) if len(dados) > 0 else 0, 0, 0
    
    media = np.mean(dados)
    erro_padrao = stats.sem(dados)
    ic = stats.t.interval(0.95, len(dados)-1, loc=media, scale=erro_padrao)
    
    return media, ic[0], ic[1]

def criar_tabela_com_ic(df_atleta, df_posicao, df_geral, variaveis):
    """
    Tabela com médias e intervalos de confiança (formato científico)
    """
    dados = []
    
    for var in variaveis:
        # Atleta (não tem IC porque é um indivíduo)
        val_atleta = df_atleta[var].mean() if len(df_atleta) > 0 else 0
        
        # Posição com IC
        dados_pos = df_posicao[var].dropna()
        if len(dados_pos) > 1:
            media_pos, ic_pos_inf, ic_pos_sup = calcular_ic_95(dados_pos)
        else:
            media_pos = dados_pos.mean() if len(dados_pos) > 0 else 0
            ic_pos_inf = ic_pos_sup = media_pos
        
        # Geral com IC
        dados_geral = df_geral[var].dropna()
        if len(dados_geral) > 1:
            media_geral, ic_geral_inf, ic_geral_sup = calcular_ic_95(dados_geral)
        else:
            media_geral = dados_geral.mean() if len(dados_geral) > 0 else 0
            ic_geral_inf = ic_geral_sup = media_geral
        
        dados.append({
            'Variável': var,
            'Atleta': f'{val_atleta:.2f}',
            'Média Posição': f'{media_pos:.2f}',
            'IC 95% Posição': f'[{ic_pos_inf:.2f}, {ic_pos_sup:.2f}]',
            'Média Geral': f'{media_geral:.2f}',
            'IC 95% Geral': f'[{ic_geral_inf:.2f}, {ic_geral_sup:.2f}]'
        })
    
    return pd.DataFrame(dados)

def gerar_relatorio_apa(df_atleta, df_grupo, atleta_nome, metricas, data_coleta, n_atletas):
    """
    Gera relatório no formato APA (American Psychological Association) - Tópico 6
    """
    from datetime import datetime
    
    resultado = f"""
    <div style="background: white; padding: 30px; border-radius: 5px; font-family: 'Times New Roman', serif; max-width: 800px; margin: 0 auto;">
        <h2 style="text-align: center; border-bottom: 2px solid #000; padding-bottom: 10px;">Relatório de Desempenho Esportivo</h2>
        <p style="text-align: center; font-style: italic;">Formato APA (7ª edição)</p>
        
        <hr style="border: 1px solid #000; margin: 20px 0;">
        
        <h3>Método</h3>
        <p><strong>Participante:</strong> {atleta_nome}</p>
        <p><strong>Data da coleta:</strong> {data_coleta}</p>
        <p><strong>Amostra de referência:</strong> {n_atletas} atletas</p>
        
        <h3>Resultados</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <thead>
                <tr style="background: #f0f0f0;">
                    <th style="border: 1px solid #000; padding: 8px;">Variável</th>
                    <th style="border: 1px solid #000; padding: 8px;">Atleta (M)</th>
                    <th style="border: 1px solid #000; padding: 8px;">Grupo (M ± DP)</th>
                    <th style="border: 1px solid #000; padding: 8px;">IC 95%</th>
                    <th style="border: 1px solid #000; padding: 8px;">d de Cohen</th>
                    <th style="border: 1px solid #000; padding: 8px;">Inferência MBI</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for metrica in metricas:
        if metrica not in df_atleta.columns or metrica not in df_grupo.columns:
            continue
            
        valor_atleta = df_atleta[metrica].mean()
        valores_grupo = df_grupo[metrica].dropna()
        
        if len(valores_grupo) > 0:
            media_grupo = valores_grupo.mean()
            dp_grupo = valores_grupo.std()
            ic_inf, ic_sup = stats.t.interval(0.95, len(valores_grupo)-1, loc=media_grupo, scale=stats.sem(valores_grupo)) if len(valores_grupo) > 1 else (media_grupo, media_grupo)
            
            # Calcular MBI
            resultado_mbi = calcular_mbi(valor_atleta, media_grupo, dp_grupo, len(valores_grupo))
            
            resultado += f"""
                <tr>
                    <td style="border: 1px solid #000; padding: 8px;">{metrica}</td>
                    <td style="border: 1px solid #000; padding: 8px;">{valor_atleta:.2f}</td>
                    <td style="border: 1px solid #000; padding: 8px;">{media_grupo:.2f} ± {dp_grupo:.2f}</td>
                    <td style="border: 1px solid #000; padding: 8px;">[{ic_inf:.2f}, {ic_sup:.2f}]</td>
                    <td style="border: 1px solid #000; padding: 8px;">{resultado_mbi['cohen_d']:.2f} ({resultado_mbi['magnitude']})</td>
                    <td style="border: 1px solid #000; padding: 8px; color: {resultado_mbi['cor']}; font-weight: bold;">{resultado_mbi['inferencia']}</td>
                </tr>
            """
    
    resultado += """
            </tbody>
        </table>
        
        <h3>Referências</h3>
        <p style="font-size: 0.9rem; margin-left: 20px; text-indent: -20px;">
            Batterham, A. M., & Hopkins, W. G. (2006). Making meaningful inferences about magnitudes. 
            <em>International Journal of Sports Physiology and Performance</em>, 1(1), 50-57.
        </p>
        <p style="font-size: 0.9rem; margin-left: 20px; text-indent: -20px;">
            Hopkins, W. G., Marshall, S. W., Batterham, A. M., & Hanin, J. (2009). 
            Progressive statistics for studies in sports medicine and exercise science. 
            <em>Medicine & Science in Sports & Exercise</em>, 41(1), 3-12.
        </p>
        <p style="font-size: 0.9rem; margin-left: 20px; text-indent: -20px;">
            Cohen, J. (1988). <em>Statistical power analysis for the behavioral sciences</em> (2nd ed.). 
            Lawrence Erlbaum Associates.
        </p>
        <p style="font-size: 0.9rem; margin-left: 20px; text-indent: -20px;">
            Gabbett, T. J. (2016). The training—injury prevention paradox: should athletes be training smarter 
            and harder?. <em>British Journal of Sports Medicine</em>, 50(5), 273-280.
        </p>
        
        <hr style="border: 1px solid #000; margin: 20px 0;">
        <p style="font-size: 0.8rem; text-align: center;">
            Relatório gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} pelo Sports Science Analytics Pro - Academic Edition
        </p>
    </div>
    """
    
    return resultado

# ============================================================================
# CALLBACKS
# ============================================================================

def atualizar_metodo_zona():
    valor_radio = st.session_state.metodo_zona_radio
    if valor_radio in ["Percentis", "Percentiles"]:
        st.session_state.metodo_zona = 'percentis'
    else:
        st.session_state.metodo_zona = 'based_on_max'

def atualizar_grupos():
    pass

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

init_session_state()

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
    
    if upload_files and len(upload_files) > 0 and not st.session_state.upload_concluido:
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
                    except Exception as e:
                        arquivos_invalidos.append(f"{uploaded_file.name}")
                
                if dataframes:
                    estruturas_ok, estrutura_referencia = verificar_estruturas_arquivos(dataframes)
                    
                    if not estruturas_ok:
                        st.error("❌ " + ("Arquivos com estruturas diferentes" if st.session_state.idioma == 'pt' else 
                                        "Files with different structures" if st.session_state.idioma == 'en' else
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
                f"Selecionar todos os {t['athlete'].lower()}s" if st.session_state.idioma == 'pt' else
                f"Select all {t['athlete'].lower()}s" if st.session_state.idioma == 'en' else
                f"Seleccionar todos los {t['athlete'].lower()}s",
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
# HEADER CIENTÍFICO (Tópico 8)
# ============================================================================

st.markdown("""
<div class="scientific-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">📊 Sports Science Analytics Pro</h1>
            <p style="color: #6c757d; margin: 5px 0 0 0; font-size: 1rem;">
                <em>Análise de Desempenho baseada em Magnitudes (Hopkins & Batterham, 2006)</em>
            </p>
        </div>
        <div style="text-align: right;" id="header_info">
            <p style="color: #212529; margin: 0; font-weight: bold;">Protocolo de Avaliação</p>
            <p style="color: #6c757d; margin: 0;" id="n_atletas">n = 0 atletas</p>
            <p style="color: #6c757d; margin: 0;" id="n_obs">0 observações</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if st.session_state.df_completo is not None:
    
    if st.session_state.processar_click:
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
                st.session_state.dados_processados = False
                st.session_state.df_filtrado = None
            else:
                st.session_state.dados_processados = True
                st.session_state.df_filtrado = df_filtrado
                st.session_state.processar_click = False
                
                # Atualizar header com informações
                st.markdown(f"""
                <script>
                    document.getElementById('n_atletas').innerHTML = 'n = {len(df_filtrado['Nome'].unique())} atletas';
                    document.getElementById('n_obs').innerHTML = '{len(df_filtrado)} observações';
                </script>
                """, unsafe_allow_html=True)
                
                st.rerun()
    
    elif st.session_state.dados_processados and st.session_state.df_filtrado is not None:
        df_filtrado = st.session_state.df_filtrado
        atletas_selecionados = st.session_state.atletas_selecionados
        posicoes_selecionadas = st.session_state.posicoes_selecionadas
        periodos_selecionados = st.session_state.periodos_selecionados
        variavel_analise = st.session_state.variavel_selecionada
        n_classes = st.session_state.n_classes
        t = translations[st.session_state.idioma]
        
        # Atualizar header
        st.markdown(f"""
        <script>
            document.getElementById('n_atletas').innerHTML = 'n = {len(df_filtrado['Nome'].unique())} atletas';
            document.getElementById('n_obs').innerHTML = '{len(df_filtrado)} observações';
        </script>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<h2>📊 {t['title'].split(' -')[0] if ' -' in t['title'] else 'Visão Geral'}</h2>", unsafe_allow_html=True)
        
        media_global = df_filtrado[variavel_analise].mean()
        media_posicoes = df_filtrado.groupby('Posição')[variavel_analise].mean()
        melhor_posicao = media_posicoes.idxmax() if not media_posicoes.empty else "N/A"
        pior_posicao = media_posicoes.idxmin() if not media_posicoes.empty else "N/A"
        
        if n_colunas == 1:
            executive_card(t['mean'], f"{media_global:.2f}", 5.2, "📊")
            executive_card("Melhor Posição", melhor_posicao, 8.1, "🏆", "#2ca02c")
            executive_card("Pior Posição", pior_posicao, -3.4, "📉", "#d62728")
            executive_card(t['observations'], len(df_filtrado), 0, "👥")
        else:
            cols_exec = st.columns(4)
            with cols_exec[0]:
                executive_card(t['mean'], f"{media_global:.2f}", 5.2, "📊")
            with cols_exec[1]:
                executive_card("Melhor Posição", melhor_posicao, 8.1, "🏆", "#2ca02c")
            with cols_exec[2]:
                executive_card("Pior Posição", pior_posicao, -3.4, "📉", "#d62728")
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
            t['tab_kmeans'],
            t['tab_comparador'],
            t['tab_mbi'],
            t['tab_export'],
            t['tab_executive']
        ]
        
        tabs = st.tabs(tab_titles)
        
        with tabs[0]:  # Distribuição
            st.markdown(f"<h3>{t['tab_distribution']}</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                dados_hist = df_filtrado[variavel_analise].dropna()
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=dados_hist,
                    nbinsx=n_classes,
                    name='Frequência',
                    marker_color='#1f77b4',
                    opacity=0.8
                ))
                
                media_hist = dados_hist.mean()
                fig_hist.add_vline(
                    x=media_hist,
                    line_dash="dash",
                    line_color="#d62728",
                    line_width=2,
                    annotation_text=f"{t['mean']}: {media_hist:.2f}",
                    annotation_position="top",
                    annotation_font_color="#212529"
                )
                
                mediana_hist = dados_hist.median()
                fig_hist.add_vline(
                    x=mediana_hist,
                    line_dash="dot",
                    line_color="#ff7f0e",
                    line_width=2,
                    annotation_text=f"{t['median']}: {mediana_hist:.2f}",
                    annotation_position="bottom",
                    annotation_font_color="#212529"
                )
                
                fig_hist.update_layout(
                    title=f"Histograma - {variavel_analise} ({n_classes} classes)",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#212529', size=11, family="Times New Roman"),
                    title_font=dict(color='#212529', size=16),
                    xaxis_title=variavel_analise,
                    yaxis_title="Frequência",
                    showlegend=False,
                    bargap=0.1
                )
                fig_hist.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                fig_hist.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                
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
                    marker=dict(color='#1f77b4', size=8, opacity=0.7)
                ))
                
                fig_qq.add_trace(go.Scatter(
                    x=quantis_teoricos,
                    y=linha_ref(quantis_teoricos),
                    mode='lines',
                    name=f'Referência (R² = {r2:.3f})',
                    line=dict(color='#d62728', width=2)
                ))
                
                fig_qq.update_layout(
                    title=f"QQ Plot - {variavel_analise}",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#212529', size=11, family="Times New Roman"),
                    title_font=dict(color='#212529', size=16),
                    xaxis_title="Quantis Teóricos",
                    yaxis_title="Quantis Observados"
                )
                fig_qq.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                fig_qq.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                
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
        
        with tabs[1]:  # Temporal
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
                time_metric_card(t['max_value'], f"{valor_maximo:.2f}", f"{t['minute_of_max']}: {minuto_maximo}", "#d62728")
            with cols_t[1]:
                time_metric_card(t['min_value'], f"{valor_minimo:.2f}", f"{t['minute_of_min']}: {minuto_minimo}", "#2ca02c")
            with cols_t[2]:
                time_metric_card(t['mean'], f"{media_tempo:.2f}", t['mean'], "#1f77b4")
            with cols_t[3]:
                time_metric_card(t['threshold_80'], f"{limiar_80:.2f}", f"80% do máx ({valor_maximo:.2f})", "#ff7f0e")
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
                cores_zonas = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728', '#2ca02c']
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
                        <div class="scientific-card" style="border-left-color: {cores_zonas[i]}; padding: 10px;">
                            <div style="font-size: 0.9rem; color: #6c757d;">{zona}</div>
                            <div style="font-size: 1.2rem; color: #212529; font-weight: 600;">{limite:.1f}</div>
                            <div style="font-size: 0.8rem; color: #6c757d;">{n_obs} obs ({n_obs/len(df_filtrado)*100:.0f}%)</div>
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
                <div class="scientific-card">
                    <h4 style="color: #1f77b4; margin-top: 0;">{t['mean']}</h4>
                    <p><strong>{t['mean']}:</strong> {media:.3f}</p>
                    <p><strong>{t['median']}:</strong> {mediana:.3f}</p>
                    <p><strong>{t['mode']}:</strong> {moda}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_e2:
                st.markdown(f"""
                <div class="scientific-card">
                    <h4 style="color: #1f77b4; margin-top: 0;">{t['std']}</h4>
                    <p><strong>{t['std']}:</strong> {desvio:.3f}</p>
                    <p><strong>{t['variance']}:</strong> {variancia:.3f}</p>
                    <p><strong>{t['cv']}:</strong> {cv:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_e3:
                st.markdown(f"""
                <div class="scientific-card">
                    <h4 style="color: #1f77b4; margin-top: 0;">{t['iqr']}</h4>
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
                <div class="scientific-card">
                    <h4 style="color: #1f77b4; margin-top: 0;">{t['skewness']}</h4>
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
                <div class="scientific-card">
                    <h4 style="color: #1f77b4; margin-top: 0;">{t['kurtosis']}</h4>
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
                <div class="scientific-card">
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
                    marker=dict(color='#1f77b4', size=20),
                    error_y=dict(type='constant', value=(ic_sup - media), color='#d62728', thickness=3, width=15),
                    name=t['mean']
                ))
                
                fig_ic.update_layout(
                    title=t['confidence_interval'],
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#212529', size=11, family="Times New Roman"),
                    title_font=dict(color='#212529', size=14),
                    showlegend=False,
                    yaxis_title=variavel_analise
                )
                fig_ic.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                fig_ic.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                
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
        
        with tabs[2]:  # Boxplots
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
                        marker_color='#1f77b4',
                        line_color='#212529',
                        fillcolor='rgba(31, 119, 180, 0.7)',
                        jitter=0.3,
                        pointpos=-1.8,
                        opacity=0.8
                    ))
            
            fig_box_pos.update_layout(
                title=f"{t['position']} - {variavel_analise}",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#212529', size=11, family="Times New Roman"),
                title_font=dict(color='#212529', size=16),
                yaxis_title=variavel_analise,
                showlegend=False
            )
            fig_box_pos.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
            fig_box_pos.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
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
                        marker_color='#9467bd',
                        line_color='#212529',
                        fillcolor='rgba(148, 103, 189, 0.7)',
                        jitter=0.3,
                        pointpos=-1.8,
                        opacity=0.8
                    ))
            
            altura_boxplot = max(400, len(atletas_selecionados) * 25)
            
            fig_box_atl.update_layout(
                title=f"{t['athlete']} - {variavel_analise}",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#212529', size=11, family="Times New Roman"),
                title_font=dict(color='#212529', size=16),
                yaxis_title=variavel_analise,
                showlegend=False,
                height=altura_boxplot
            )
            fig_box_atl.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'), tickangle=-45)
            fig_box_atl.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
            st.plotly_chart(fig_box_atl, use_container_width=True)
            
            with st.expander(f"📊 {t['descriptive_stats']} {t['athlete'].lower()}"):
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h5 style="color: #1f77b4;">{t['iqr_title']}</h5>
                    <p style="color: #6c757d;">{t['iqr_explanation']}</p>
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
        
        with tabs[3]:  # Correlações
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
                    
                    colorscale = [
                        [0, 'rgba(31, 119, 180, 0.9)'],
                        [0.25, 'rgba(44, 160, 44, 0.9)'],
                        [0.5, 'rgba(255, 255, 255, 0.9)'],
                        [0.75, 'rgba(255, 127, 14, 0.9)'],
                        [1, 'rgba(214, 39, 40, 0.9)']
                    ]
                    
                    df_corr_display = df_corr.copy()
                    
                    fig_corr = px.imshow(
                        df_corr_display,
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale=colorscale,
                        title=f"{t['tab_correlation']}",
                        zmin=-1, zmax=1
                    )
                    
                    for i in range(len(df_corr)):
                        fig_corr.add_annotation(
                            x=i,
                            y=i,
                            text=f"{df_corr.iloc[i, i]:.2f}",
                            showarrow=False,
                            font=dict(color='white', size=12, weight='bold'),
                            bgcolor='#4a5568',
                            bordercolor='#718096',
                            borderwidth=1,
                            opacity=0.9
                        )
                        
                        fig_corr.add_shape(
                            type="rect",
                            x0=i - 0.5,
                            y0=i - 0.5,
                            x1=i + 0.5,
                            y1=i + 0.5,
                            line=dict(width=0),
                            fillcolor='rgba(74, 85, 104, 0.5)',
                            layer="below"
                        )
                    
                    fig_corr.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#212529', size=11, family="Times New Roman"),
                        title_font=dict(color='#212529', size=16),
                        height=500
                    )
                    fig_corr.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                    fig_corr.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.markdown(f"<h4>📊 {t['tab_correlation']}</h4>", unsafe_allow_html=True)
                    
                    def style_correlation(val):
                        if pd.isna(val):
                            return 'color: #6c757d;'
                        if val == 1.0:
                            return 'color: #2d3748; font-weight: bold; background-color: #e1e5e9;'
                        color = '#d62728' if abs(val) > 0.7 else '#ff7f0e' if abs(val) > 0.5 else '#1f77b4'
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
                            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                        )
                        fig_scatter.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#212529', size=11, family="Times New Roman"),
                            title_font=dict(color='#212529', size=16),
                            height=500,
                            legend=dict(font=dict(color='#212529'), bgcolor='rgba(255,255,255,0.8)', bordercolor='#dee2e6')
                        )
                        fig_scatter.update_xaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
                        fig_scatter.update_yaxes(gridcolor='#e1e5e9', tickfont=dict(color='#212529'))
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
                        <div class="scientific-card">
                            <h4 style="color: #1f77b4;">📊 {t['tab_correlation']}</h4>
                            <hr style="border-color: #dee2e6;">
                            <p><strong>Pearson:</strong> {corr_valor:.3f}</p>
                            <p><strong>{t['tab_correlation']}:</strong> {interp_corr}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ " + ("Selecione pelo menos 2 variáveis" if st.session_state.idioma == 'pt' else 
                                   "Select at least 2 variables"))
            else:
                st.info("ℹ️ " + ("São necessárias pelo menos 2 variáveis" if st.session_state.idioma == 'pt' else 
                               "At least 2 variables are needed"))
        
        with tabs[4]:  # K-means
            st.markdown(f"<h3>{t['tab_kmeans']}</h3>", unsafe_allow_html=True)
            
            st.session_state.kmeans_ativo = True
            
            st.markdown("""
            <div class="scientific-card" style="border-left: 4px solid #9467bd;">
                <p style="color: #6c757d; margin: 0;">
                    <span style="color: #9467bd; font-size: 1.2rem;">🎯 Segmentação de Atletas</span><br>
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
                    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    
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
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#212529', size=12, family="Times New Roman"),
                        title_font=dict(color='#9467bd', size=20),
                        height=600,
                        hovermode='closest',
                        legend=dict(
                            font=dict(color='#212529'),
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='#dee2e6',
                            borderwidth=1
                        )
                    )
                    
                    fig_kmeans.update_xaxes(
                        gridcolor='#e1e5e9', 
                        tickfont=dict(color='#212529'),
                        title_font=dict(color='#212529')
                    )
                    fig_kmeans.update_yaxes(
                        gridcolor='#e1e5e9', 
                        tickfont=dict(color='#212529'),
                        title_font=dict(color='#212529')
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
        
        with tabs[5]:  # Comparador de Atletas
            st.markdown(f"<h3>{t['tab_comparador']}</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="scientific-card" style="border-left: 4px solid #9467bd;">
                <p style="color: #6c757d; margin: 0;">
                    <span style="color: #9467bd; font-size: 1.2rem;">🆚 Comparação Individual</span><br>
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
                    <div class="scientific-card" style="padding: 10px; margin-top: 25px;">
                        <p style="color: #6c757d; margin: 0;">Posição do Atleta</p>
                        <p style="color: #212529; font-size: 1.2rem; margin: 0;">{posicao_atleta}</p>
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
                                    return 'color: #2ca02c; font-weight: bold;'
                                elif '▼' in str(val):
                                    return 'color: #d62728; font-weight: bold;'
                                return 'color: #6c757d;'
                            
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
                                st.markdown("""
                                <div class="scientific-card" style="border-left: 4px solid #2ca02c;">
                                    <h4 style="color: #2ca02c; margin-top: 0;">✅ Pontos Fortes</h4>
                                """, unsafe_allow_html=True)
                                
                                if vantagens:
                                    for metrica, pct in vantagens:
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                            <span style="color: #212529;">{metrica}</span>
                                            <span style="color: #2ca02c; font-weight: bold;">+{pct:.1f}%</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown("<p style='color: #6c757d;'>Nenhum destaque significativo</p>", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with col_d2:
                                st.markdown("""
                                <div class="scientific-card" style="border-left: 4px solid #d62728;">
                                    <h4 style="color: #d62728; margin-top: 0;">⚠️ Pontos a Desenvolver</h4>
                                """, unsafe_allow_html=True)
                                
                                if desvantagens:
                                    for metrica, pct in desvantagens:
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                            <span style="color: #212529;">{metrica}</span>
                                            <span style="color: #d62728; font-weight: bold;">-{pct:.1f}%</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown("<p style='color: #6c757d;'>Acima da média em todas as métricas!</p>", unsafe_allow_html=True)
                                
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
                                    cor = "#2ca02c"
                                    icone = "🏆"
                                elif performance_media > 0:
                                    status = "ACIMA DA MÉDIA"
                                    cor = "#1f77b4"
                                    icone = "📈"
                                elif performance_media > -10:
                                    status = "NA MÉDIA"
                                    cor = "#6c757d"
                                    icone = "📊"
                                else:
                                    status = "ABAIXO DA MÉDIA"
                                    cor = "#d62728"
                                    icone = "📉"
                                
                                st.markdown(f"""
                                <div class="scientific-card" style="border-left: 6px solid {cor};">
                                    <div style="display: flex; align-items: center; gap: 20px;">
                                        <div style="font-size: 3rem;">{icone}</div>
                                        <div>
                                            <h4 style="color: #212529; margin: 0;">Desempenho Geral: <span style="color: {cor};">{status}</span></h4>
                                            <p style="color: #6c757d; margin: 5px 0;">
                                                {atleta_comp} está {performance_media:+.1f}% acima da média da posição
                                            </p>
                                            <p style="color: #6c757d; margin: 5px 0; font-size: 0.9rem;">
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
        
        with tabs[6]:  # NOVA ABA: Análise MBI (Tópicos 1 e 2)
            st.markdown(f"<h3>🔬 Análise de Magnitude (MBI)</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="scientific-card" style="border-left: 4px solid #9467bd;">
                <p style="color: #6c757d; margin: 0;">
                    <span style="color: #9467bd; font-size: 1.2rem;">📊 Magnitude-Based Inference</span><br>
                    Baseado em Hopkins & Batterham (2006). Inferência sobre a importância clínica/prática das diferenças.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                atleta_mbi = st.selectbox(
                    "Selecione o Atleta",
                    options=atletas_selecionados,
                    key="mbi_atleta"
                )
            
            with col_m2:
                referencia_mbi = st.selectbox(
                    "Referência de comparação",
                    options=["Média da Posição", "Média Geral"],
                    key="mbi_referencia"
                )
            
            # Heatmap de Magnitude (Tópico 2)
            st.markdown("### 🔥 Heatmap de Magnitudes (Z-Scores)")
            
            with st.expander("ℹ️ Sobre o Heatmap de Magnitudes"):
                st.markdown("""
                <p style="color: #6c757d;">
                    O heatmap mostra o perfil de magnitudes de cada atleta em relação à média geral.
                    Valores positivos (vermelho) indicam desempenho acima da média, negativos (azul) abaixo da média.
                </p>
                """, unsafe_allow_html=True)
            
            fig_heatmap = criar_heatmap_magnitude(df_filtrado, posicao_referencia=None)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Análise MBI individual
            st.markdown("### 📊 Análise Individual por Magnitude")
            
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
                    default=st.session_state.variaveis_quantitativas[:5],
                    key="mbi_metricas"
                )
                
                if metricas_mbi and st.button("🔬 Calcular MBI", key="btn_mbi"):
                    
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
            
            # Referências científicas
            with st.expander("📚 Referências sobre MBI"):
                st.markdown("""
                <p style="color: #212529;">
                    <strong>Batterham, A. M., & Hopkins, W. G. (2006).</strong> Making meaningful inferences about magnitudes. 
                    <em>International Journal of Sports Physiology and Performance</em>, 1(1), 50-57.
                </p>
                <p style="color: #212529;">
                    <strong>Hopkins, W. G., Marshall, S. W., Batterham, A. M., & Hanin, J. (2009).</strong> 
                    Progressive statistics for studies in sports medicine and exercise science. 
                    <em>Medicine & Science in Sports & Exercise</em>, 41(1), 3-12.
                </p>
                """, unsafe_allow_html=True)
        
        with tabs[7]:  # NOVA ABA: Exportação APA (Tópico 6)
            st.markdown(f"<h3>📋 Exportação no Formato APA</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="scientific-card" style="border-left: 4px solid #9467bd;">
                <p style="color: #6c757d; margin: 0;">
                    <span style="color: #9467bd; font-size: 1.2rem;">📄 Relatório Científico</span><br>
                    Gere relatórios no formato APA (American Psychological Association) para publicação.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                atleta_export = st.selectbox(
                    "Atleta para relatório",
                    options=atletas_selecionados,
                    key="export_atleta"
                )
            
            with col_e2:
                data_coleta = st.date_input(
                    "Data da coleta",
                    value=datetime.now(),
                    key="export_data"
                )
            
            metricas_export = st.multiselect(
                "Métricas a incluir no relatório",
                options=st.session_state.variaveis_quantitativas,
                default=st.session_state.variaveis_quantitativas,
                key="export_metricas"
            )
            
            if st.button("📄 Gerar Relatório APA", key="btn_export"):
                with st.spinner('Gerando relatório no formato APA...'):
                    time.sleep(1)
                    
                    df_atleta_exp = df_filtrado[df_filtrado['Nome'] == atleta_export]
                    df_grupo_exp = df_filtrado
                    
                    relatorio_html = gerar_relatorio_apa(
                        df_atleta_exp, df_grupo_exp, atleta_export, 
                        metricas_export, data_coleta.strftime('%d/%m/%Y'),
                        len(df_filtrado['Nome'].unique())
                    )
                    
                    st.markdown("### 📄 Visualização do Relatório")
                    st.markdown(relatorio_html, unsafe_allow_html=True)
                    
                    # Download do relatório
                    st.download_button(
                        label="📥 Download Relatório (HTML)",
                        data=relatorio_html,
                        file_name=f"relatorio_apa_{atleta_export}_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )
        
        with tabs[8]:  # Executivo
            st.markdown(f"<h3>{t['tab_executive']}</h3>", unsafe_allow_html=True)
            
            st.markdown("### 🆚 Comparação de Atletas")
            if len(atletas_selecionados) >= 2:
                col_atl1, col_atl2 = st.columns(2)
                with col_atl1:
                    atleta1_comp = st.selectbox(
                        "Atleta 1", 
                        atletas_selecionados, 
                        index=0, 
                        key="atleta1_comp"
                    )
                with col_atl2:
                    atleta2_comp = st.selectbox(
                        "Atleta 2", 
                        atletas_selecionados, 
                        index=min(1, len(atletas_selecionados)-1), 
                        key="atleta2_comp"
                    )
                
                if atleta1_comp != atleta2_comp:
                    vars_comp = st.multiselect(
                        "Variáveis para comparar",
                        st.session_state.variaveis_quantitativas,
                        default=st.session_state.variaveis_quantitativas[:3],
                        key="vars_comp_exec"
                    )
                    
                    if len(vars_comp) >= 1:
                        comparar_atletas(df_filtrado, atleta1_comp, atleta2_comp, vars_comp, t)
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
    
    else:
        t = translations[st.session_state.idioma]
        st.info("👈 Selecione os filtros e clique em **Processar Análise** na barra lateral")

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
        <div class="scientific-card">
            <h4 style="color: #1f77b4;">📌 Componentes</h4>
            <hr style="border-color: #dee2e6;">
            <p><strong>Nome:</strong> Mariano, Maria, Joao...</p>
            <p><strong>Período:</strong> 1 TEMPO, SEGUNDO TEMPO...</p>
            <p><strong>Minuto:</strong> 00:00-01:00, 05:00-06:00...</p>
            <p><strong>Posição:</strong> Atacante, Meio-campo...</p>
        </div>
        
        <div class="scientific-card" style="margin-top: 20px;">
            <h4 style="color: #1f77b4;">💡 Dica</h4>
            <hr style="border-color: #dee2e6;">
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
        st.caption(f"**{t['variable']}s:** {', '.join(st.session_state.variaveis_quantitativas)}")
        if st.session_state.todos_posicoes:
            st.caption(f"**{t['positions']}:** {', '.join(st.session_state.todos_posicoes)}")
        if st.session_state.todos_periodos:
            st.caption(f"**{t['periods']}:** {', '.join(st.session_state.todos_periodos)}")

# ============================================================================
# RODAPÉ ACADÊMICO (Tópico 9)
# ============================================================================

st.markdown("""
<div class="academic-footer">
    <p style="color: #6c757d; font-size: 0.85rem; margin: 0; text-align: center;">
        <strong>Referências Metodológicas:</strong><br>
        Batterham, A. M., & Hopkins, W. G. (2006). Making meaningful inferences about magnitudes. 
        <em>International Journal of Sports Physiology and Performance</em>, 1(1), 50-57.<br>
        Hopkins, W. G., Marshall, S. W., Batterham, A. M., & Hanin, J. (2009). Progressive statistics for studies 
        in sports medicine and exercise science. <em>Medicine & Science in Sports & Exercise</em>, 41(1), 3-12.<br>
        Cohen, J. (1988). <em>Statistical power analysis for the behavioral sciences</em> (2nd ed.). 
        Lawrence Erlbaum Associates.<br>
        Gabbett, T. J. (2016). The training—injury prevention paradox: should athletes be training smarter 
        and harder?. <em>British Journal of Sports Medicine</em>, 50(5), 273-280.
    </p>
    <p style="color: #adb5bd; font-size: 0.8rem; margin-top: 10px; text-align: center;">
        © 2024 Sports Science Analytics Pro | Versão Acadêmica 2.0 | Baseado em princípios de inferência por magnitude
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# GARANTIR QUE NÃO PERCA O ESTADO AO RECARREGAR
# ============================================================================
if st.session_state.dados_processados and not st.session_state.processar_click:
    pass