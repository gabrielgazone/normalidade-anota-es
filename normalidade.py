import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import time

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide", initial_sidebar_state="expanded")

# Tema dark moderno com CSS
st.markdown("""
<style>
    /* Tema dark moderno */
    .stApp {
        background: #0a0a0a;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }
    
    /* Sidebar - fundo preto com texto branco */
    .css-1d391kg, .css-1wrcr25 {
        background: #000000 !important;
        border-right: 1px solid #333333;
    }
    
    /* Títulos na sidebar */
    .sidebar-title {
        color: white !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #3498db;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Texto na sidebar */
    .css-1d391kg p, .css-1d391kg label, .css-1d391kg .stMarkdown, 
    .css-1d391kg .stSelectbox label, .css-1d391kg .stMultiselect label {
        color: #ffffff !important;
    }
    
    /* Inputs na sidebar */
    .css-1d391kg .stSelectbox, .css-1d391kg .stMultiselect {
        background: #1a1a1a;
        border-radius: 6px;
        border: 1px solid #333333;
        color: white;
    }
    
    .css-1d391kg .stSelectbox div, .css-1d391kg .stMultiselect div {
        color: white !important;
    }
    
    /* Slider na sidebar */
    .css-1d391kg .stSlider label {
        color: white !important;
    }
    
    /* Botão na sidebar */
    .css-1d391kg .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Cards com gradiente */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        text-align: center;
        color: white !important;
        transition: all 0.3s ease;
        border: 1px solid #333;
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
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(52, 152, 219, 0.3);
        border-color: #3498db;
    }
    
    .metric-card h3 {
        color: #3498db !important;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    
    .metric-card h2 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    /* Cards para métricas temporais */
    .time-metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    
    .time-metric-card .label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .time-metric-card .value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .time-metric-card .sub-value {
        color: #888;
        font-size: 0.9rem;
    }
    
    /* Card especial para eventos acima do limiar */
    .warning-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(231, 76, 60, 0.3);
        text-align: center;
        color: white;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .warning-card .label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    
    .warning-card .value {
        font-size: 3rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .warning-card .sub-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Títulos principais */
    h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(52, 152, 219, 0.5);
        background: linear-gradient(135deg, #3498db, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    }
    
    h2 {
        color: white !important;
        font-size: 2rem;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #3498db !important;
        font-size: 1.5rem;
        font-weight: 500;
    }
    
    h4 {
        color: #9b59b6 !important;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* Abas personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #1a1a1a;
        padding: 10px;
        border-radius: 50px;
        border: 1px solid #333;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 600;
        color: #888 !important;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #9b59b6 100%) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a1a !important;
        border-radius: 10px !important;
        border: 1px solid #333 !important;
        color: white !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: #1a1a1a;
        border-radius: 0 0 10px 10px;
        border: 1px solid #333;
        border-top: none;
    }
    
    /* Containers de métricas */
    .metric-container {
        background: #1a1a1a;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #333;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        border-color: #3498db;
        box-shadow: 0 10px 25px rgba(52, 152, 219, 0.2);
    }
    
    .metric-container h4 {
        color: #3498db !important;
        margin-bottom: 15px;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-container p {
        color: white !important;
        margin: 10px 0;
        font-size: 1rem;
    }
    
    .metric-container strong {
        color: #9b59b6;
    }
    
    /* Dataframe */
    .dataframe {
        background: #1a1a1a !important;
        border-radius: 10px !important;
        border: 1px solid #333 !important;
        color: white !important;
    }
    
    .dataframe th {
        background: #2d2d2d !important;
        color: #3498db !important;
        font-weight: 600;
    }
    
    .dataframe td {
        background: #1a1a1a !important;
        color: white !important;
        border-color: #333 !important;
    }
    
    /* Texto geral */
    p, li, .caption, .stMarkdown {
        color: #cccccc !important;
    }
    
    /* Info boxes */
    .stInfo, .stWarning, .stError {
        background: #1a1a1a !important;
        border-left-color: #3498db !important;
        color: white !important;
        border-radius: 10px;
    }
    
    /* Ícones */
    .icon-container {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    
    /* Linhas divisórias */
    hr {
        border-color: #333 !important;
        margin: 20px 0;
    }
    
    /* Animações de entrada */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Header com ícone
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>📊 Análise de Normalidade</h1>
        <p style="color: #888; font-size: 1.2rem;">Dashboard Profissional para Análise Estatística</p>
    </div>
    """, unsafe_allow_html=True)

# Inicializar session state
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

# --- FUNÇÕES AUXILIARES ---
def interpretar_teste(p_valor, nome_teste):
    """Função auxiliar para interpretar resultados do teste de normalidade"""
    if p_valor < 0.0001:
        p_text = f"{p_valor:.2e}"
    else:
        p_text = f"{p_valor:.5f}"
    
    if p_valor > 0.05:
        status = "✅ Dados seguem distribuição normal"
        cor = "#27ae60"
    else:
        status = "⚠️ Dados não seguem distribuição normal"
        cor = "#e74c3c"
    
    st.markdown(f"""
    <div style="background: #1a1a1a; border-radius: 10px; padding: 20px; border-left: 5px solid {cor}; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
        <h4 style="color: white; margin: 0 0 10px 0;">{status}</h4>
        <p style="color: #ccc; margin: 5px 0;"><strong>Teste:</strong> {nome_teste}</p>
        <p style="color: #ccc; margin: 5px 0;"><strong>Valor de p:</strong> <span style="color: {cor};">{p_text}</span></p>
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
    """Verifica se todos os dataframes têm a mesma estrutura de colunas"""
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
        <div class="icon-container">{icone}</div>
        <h3>{titulo}</h3>
        <h2>{valor}</h2>
    </div>
    """, unsafe_allow_html=True)

def time_metric_card(label, valor, sub_label="", cor="#3498db"):
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
    if media != 0:
        return (desvio / media) * 100
    return 0

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>📂 Upload dos Dados</h2>", unsafe_allow_html=True)
    
    upload_files = st.file_uploader(
        "",
        type=['csv'],
        accept_multiple_files=True,
        help="Selecione um ou mais arquivos CSV com a mesma estrutura"
    )
    
    if upload_files:
        with st.spinner('Processando arquivos...'):
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
                        st.error("❌ Arquivos com estruturas diferentes")
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
                                
                                st.success(f"✅ {len(arquivos_validos)} arquivo(s) carregado(s)")
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
    
    if st.session_state.df_completo is not None:
        st.markdown("---")
        
        # Seleção de Variável
        if st.session_state.variaveis_quantitativas:
            st.markdown("<h2 class='sidebar-title'>📈 Variável</h2>", unsafe_allow_html=True)
            
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
                st.caption(f"📊 {len(df_temp)} obs | Média: {df_temp.mean():.2f}")
        
        # Filtro por Posição
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown("<h2 class='sidebar-title'>📍 Posição</h2>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Selecionar todas as posições", value=True)
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
            st.markdown("<h2 class='sidebar-title'>📅 Período</h2>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Selecionar todos os períodos", value=True)
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
            st.markdown("<h2 class='sidebar-title'>👤 Atleta</h2>", unsafe_allow_html=True)
            
            df_temp = st.session_state.df_completo.copy()
            if st.session_state.posicoes_selecionadas:
                df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
            if st.session_state.periodos_selecionados:
                df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
            
            atletas_disponiveis = sorted(df_temp['Nome'].unique())
            
            selecionar_todos = st.checkbox("Selecionar todos os atletas", value=True)
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
        st.markdown("<h2 class='sidebar-title'>⚙️ Configurações</h2>", unsafe_allow_html=True)
        
        n_classes = st.slider("Número de classes:", 3, 20, 5, help="Define o número de classes/barras no histograma e na tabela de frequência")
        
        # Botão Processar
        st.markdown("---")
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button("🚀 Processar Análise", use_container_width=True, disabled=not pode_processar):
            st.session_state.process_button = True
            st.rerun()

# --- ÁREA PRINCIPAL ---
if st.session_state.get('process_button', False) and st.session_state.df_completo is not None:
    
    with st.spinner('🔄 Gerando análises...'):
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
            st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados")
        else:
            # Métricas principais
            st.markdown("<h2>📊 Visão Geral</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Posições", len(posicoes_selecionadas), "📍", "linear-gradient(135deg, #3498db, #2980b9)")
            with col2:
                metric_card("Períodos", len(periodos_selecionados), "📅", "linear-gradient(135deg, #9b59b6, #8e44ad)")
            with col3:
                metric_card("Atletas", len(atletas_selecionados), "👥", "linear-gradient(135deg, #e74c3c, #c0392b)")
            with col4:
                metric_card("Observações", len(df_filtrado), "📊", "linear-gradient(135deg, #f39c12, #d35400)")
            
            st.markdown("---")
            
            # Organizar em abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Distribuição", 
                "📈 Estatísticas & Temporal", 
                "📦 Boxplots",
                "🔥 Correlações"
            ])
            
            with tab1:
                st.markdown("<h3>📊 Análise de Distribuição</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma com número de classes dinâmico
                    dados_hist = df_filtrado[variavel_analise].dropna()
                    
                    # Calcular os bins corretamente
                    valor_min = dados_hist.min()
                    valor_max = dados_hist.max()
                    bin_edges = np.linspace(valor_min, valor_max, n_classes + 1)
                    
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=dados_hist,
                        xbins=dict(
                            start=valor_min,
                            end=valor_max,
                            size=(valor_max - valor_min) / n_classes
                        ),
                        autobinx=False,
                        nbinsx=n_classes,
                        name='Frequência',
                        marker_color='#3498db',
                        opacity=0.8,
                        hovertemplate='Faixa: %{x}<br>Frequência: %{y}<extra></extra>'
                    ))
                    
                    media_hist = dados_hist.mean()
                    fig_hist.add_vline(
                        x=media_hist,
                        line_dash="dash",
                        line_color="#e74c3c",
                        line_width=2,
                        annotation_text=f"Média: {media_hist:.2f}",
                        annotation_position="top",
                        annotation_font_color="white"
                    )
                    
                    mediana_hist = dados_hist.median()
                    fig_hist.add_vline(
                        x=mediana_hist,
                        line_dash="dot",
                        line_color="#f39c12",
                        line_width=2,
                        annotation_text=f"Mediana: {mediana_hist:.2f}",
                        annotation_position="bottom",
                        annotation_font_color="white"
                    )
                    
                    fig_hist.update_layout(
                        title=f"Histograma - {variavel_analise} ({n_classes} classes)",
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3498db', size=16),
                        xaxis_title=variavel_analise,
                        yaxis_title="Frequência",
                        showlegend=False,
                        bargap=0.1
                    )
                    fig_hist.update_xaxes(gridcolor='#333', tickfont=dict(color='white'))
                    fig_hist.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                    
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
                    r2 = 1 - (ss_res / ss_tot)
                    
                    fig_qq = go.Figure()
                    
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=quantis_observados,
                        mode='markers',
                        name='Dados',
                        marker=dict(color='#3498db', size=8, opacity=0.7),
                        hovertemplate='Teórico: %{x:.2f}<br>Observado: %{y:.2f}<extra></extra>'
                    ))
                    
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=linha_ref(quantis_teoricos),
                        mode='lines',
                        name=f'Referência (R² = {r2:.3f})',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    
                    fig_qq.update_layout(
                        title=f"QQ Plot - {variavel_analise}",
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3498db', size=16),
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis Observados"
                    )
                    fig_qq.update_xaxes(gridcolor='#333', tickfont=dict(color='white'))
                    fig_qq.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # Tabela de Frequência com número de classes dinâmico
                st.markdown("---")
                st.markdown(f"<h4>📋 Tabela de Frequência ({n_classes} classes)</h4>", unsafe_allow_html=True)
                
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
            
            with tab2:
                st.markdown("<h3>📈 Estatísticas e Evolução Temporal</h3>", unsafe_allow_html=True)
                
                # Calcular métricas temporais globais
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                valor_maximo = df_tempo[variavel_analise].max()
                minuto_maximo = df_tempo.loc[df_tempo[variavel_analise].idxmax(), 'Minuto']
                media_tempo = df_tempo[variavel_analise].mean()
                limiar_80 = valor_maximo * 0.8
                
                # Contar eventos acima do limiar
                eventos_acima_80 = (df_tempo[variavel_analise] > limiar_80).sum()
                percentual_acima_80 = (eventos_acima_80 / len(df_tempo)) * 100
                
                # Cards com métricas temporais (agora com 4 colunas)
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                
                with col_t1:
                    time_metric_card(
                        "📈 VALOR MÁXIMO",
                        f"{valor_maximo:.2f}",
                        f"Minuto: {minuto_maximo}",
                        "#e74c3c"
                    )
                
                with col_t2:
                    time_metric_card(
                        "📊 MÉDIA",
                        f"{media_tempo:.2f}",
                        "Valor médio",
                        "#3498db"
                    )
                
                with col_t3:
                    time_metric_card(
                        "🎯 LIMIAR 80%",
                        f"{limiar_80:.2f}",
                        f"80% do máximo ({valor_maximo:.2f})",
                        "#f39c12"
                    )
                
                with col_t4:
                    warning_card(
                        "EVENTOS CRÍTICOS",
                        f"{eventos_acima_80}",
                        f"{percentual_acima_80:.1f}% das observações acima do limiar de 80%",
                        "⚠️"
                    )
                
                # Gráfico temporal com cores condicionais
                fig_tempo = go.Figure()
                
                # Criar lista de cores baseada no limiar de 80%
                cores_pontos = ['#e74c3c' if v > limiar_80 else '#3498db' for v in df_tempo[variavel_analise]]
                
                # Adicionar linha conectando os pontos
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'],
                    y=df_tempo[variavel_analise],
                    mode='lines',
                    line=dict(color='#888', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Adicionar pontos coloridos
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'],
                    y=df_tempo[variavel_analise],
                    mode='markers',
                    marker=dict(
                        color=cores_pontos,
                        size=10,
                        line=dict(color='white', width=1)
                    ),
                    showlegend=False,
                    hovertemplate='Minuto: %{x}<br>Valor: %{y:.2f}<br>' + 
                                 'Status: %{text}<extra></extra>',
                    text=['⚠️ ACIMA DO LIMIAR' if v > limiar_80 else '✅ ABAIXO DO LIMIAR' 
                          for v in df_tempo[variavel_analise]]
                ))
                
                # Linhas de referência
                fig_tempo.add_hline(
                    y=media_tempo,
                    line_dash="dash",
                    line_color="#3498db",
                    line_width=2,
                    annotation_text=f"Média: {media_tempo:.2f}",
                    annotation_position="left",
                    annotation_font_color="white"
                )
                
                fig_tempo.add_hline(
                    y=limiar_80,
                    line_dash="dot",
                    line_color="#f39c12",
                    line_width=2,
                    annotation_text=f"80% do Máx: {limiar_80:.2f}",
                    annotation_position="right",
                    annotation_font_color="white"
                )
                
                fig_tempo.add_hline(
                    y=valor_maximo,
                    line_dash="solid",
                    line_color="#e74c3c",
                    line_width=2,
                    annotation_text=f"Máx: {valor_maximo:.2f}",
                    annotation_position="right",
                    annotation_font_color="white"
                )
                
                fig_tempo.update_layout(
                    title=f"Evolução Temporal - {variavel_analise}",
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3498db', size=16),
                    xaxis_title="Minuto",
                    yaxis_title=variavel_analise,
                    hovermode='closest',
                    hoverlabel=dict(bgcolor="#1a1a1a", font_size=12)
                )
                fig_tempo.update_xaxes(gridcolor='#333', tickfont=dict(color='white'), tickangle=-45)
                fig_tempo.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                
                st.plotly_chart(fig_tempo, use_container_width=True)
                
                # Legenda explicativa
                st.markdown("""
                <div style="background: #1a1a1a; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <p style="color: #ccc; margin: 0;">
                        <span style="color: #e74c3c;">🔴 Pontos vermelhos:</span> Valores acima de 80% do máximo 
                        | <span style="color: #3498db;">🔵 Pontos azuis:</span> Valores abaixo de 80% do máximo
                        | <span style="color: #f39c12;">🟡 Linha amarela:</span> Limiar de 80%
                        | <span style="color: #e74c3c;">🔴 Linha vermelha:</span> Valor máximo
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Estatísticas descritivas
                st.markdown("---")
                st.markdown("<h4>📊 Estatísticas Descritivas</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    media = df_filtrado[variavel_analise].mean()
                    mediana = df_filtrado[variavel_analise].median()
                    moda = df_filtrado[variavel_analise].mode().iloc[0] if not df_filtrado[variavel_analise].mode().empty else 'N/A'
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📊 Medidas de Tendência</h4>
                        <hr style="border-color: #333;">
                        <p><strong>Média:</strong> {media:.3f}</p>
                        <p><strong>Mediana:</strong> {mediana:.3f}</p>
                        <p><strong>Moda:</strong> {moda}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    desvio = df_filtrado[variavel_analise].std()
                    variancia = df_filtrado[variavel_analise].var()
                    cv = calcular_cv(media, desvio)
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📈 Medidas de Dispersão</h4>
                        <hr style="border-color: #333;">
                        <p><strong>Desvio Padrão:</strong> {desvio:.3f}</p>
                        <p><strong>Variância:</strong> {variancia:.3f}</p>
                        <p><strong>Coeficiente de Variação:</strong> {cv:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    q1 = df_filtrado[variavel_analise].quantile(0.25)
                    q3 = df_filtrado[variavel_analise].quantile(0.75)
                    iqr = q3 - q1
                    amplitude_total = df_filtrado[variavel_analise].max() - df_filtrado[variavel_analise].min()
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📐 Quartis e Amplitude</h4>
                        <hr style="border-color: #333;">
                        <p><strong>Q1 (25%):</strong> {q1:.3f}</p>
                        <p><strong>Q3 (75%):</strong> {q3:.3f}</p>
                        <p><strong>IQR (Intervalo Interquartil):</strong> {iqr:.3f}</p>
                        <p><small>O IQR representa a amplitude dos 50% centrais dos dados, sendo uma medida robusta de dispersão menos sensível a outliers.</small></p>
                        <p><strong>Amplitude Total:</strong> {amplitude_total:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Assimetria e Curtose
                col_a1, col_a2 = st.columns(2)
                
                with col_a1:
                    assimetria = df_filtrado[variavel_analise].skew()
                    interpretacao_ass = ""
                    if abs(assimetria) < 0.5:
                        interpretacao_ass = "Aproximadamente simétrica"
                    elif abs(assimetria) < 1:
                        interpretacao_ass = "Moderadamente assimétrica"
                    else:
                        interpretacao_ass = "Fortemente assimétrica"
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📏 Assimetria</h4>
                        <hr style="border-color: #333;">
                        <p><strong>Valor:</strong> {assimetria:.3f}</p>
                        <p><strong>Interpretação:</strong> {interpretacao_ass}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_a2:
                    curtose = df_filtrado[variavel_analise].kurtosis()
                    interpretacao_curt = ""
                    if curtose > 0:
                        interpretacao_curt = "Leptocúrtica (caudas pesadas)"
                    elif curtose < 0:
                        interpretacao_curt = "Platicúrtica (caudas leves)"
                    else:
                        interpretacao_curt = "Mesocúrtica (normal)"
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📐 Curtose</h4>
                        <hr style="border-color: #333;">
                        <p><strong>Valor:</strong> {curtose:.3f}</p>
                        <p><strong>Interpretação:</strong> {interpretacao_curt}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Intervalo de Confiança
                st.markdown("---")
                st.markdown("<h4>🎯 Intervalo de Confiança (95%) para a Média</h4>", unsafe_allow_html=True)
                
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
                        t = stats.t.ppf(0.975, n-1)
                        ic_inf = media - t * erro_padrao
                        ic_sup = media + t * erro_padrao
                        dist = "t-Student"
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <p><strong>Média:</strong> {media:.3f}</p>
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
                        marker=dict(color='#3498db', size=20),
                        error_y=dict(
                            type='constant',
                            value=(ic_sup - media),
                            color='#e74c3c',
                            thickness=3,
                            width=15
                        ),
                        name='Média'
                    ))
                    
                    fig_ic.update_layout(
                        title="Intervalo de Confiança (95%)",
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        font=dict(color='white', size=11),
                        title_font=dict(color='#3498db', size=14),
                        showlegend=False,
                        yaxis_title=variavel_analise
                    )
                    fig_ic.update_xaxes(gridcolor='#333', tickfont=dict(color='white'))
                    fig_ic.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                    
                    st.plotly_chart(fig_ic, use_container_width=True)
                
                # Teste de Normalidade
                st.markdown("---")
                st.markdown("<h4>🧪 Teste de Normalidade</h4>", unsafe_allow_html=True)
                
                dados_teste = df_filtrado[variavel_analise].dropna()
                n_teste = len(dados_teste)
                
                if n_teste < 3:
                    st.error("❌ Amostra muito pequena (n < 3). Teste não aplicável.")
                elif n_teste > 5000:
                    st.info("ℹ️ Amostra grande demais para Shapiro-Wilk. Usando teste D'Agostino-Pearson.")
                    try:
                        k2, p = stats.normaltest(dados_teste)
                        interpretar_teste(p, "D'Agostino-Pearson")
                    except:
                        st.warning("⚠️ Teste D'Agostino-Pearson não pôde ser calculado. Usando Kolmogorov-Smirnov.")
                        try:
                            _, p = stats.kstest(dados_teste, 'norm', args=(dados_teste.mean(), dados_teste.std()))
                            interpretar_teste(p, "Kolmogorov-Smirnov")
                        except:
                            st.error("❌ Não foi possível realizar nenhum teste de normalidade.")
                else:
                    try:
                        shapiro = stats.shapiro(dados_teste)
                        interpretar_teste(shapiro.pvalue, "Shapiro-Wilk")
                    except Exception as e:
                        st.error(f"❌ Erro no teste Shapiro-Wilk: {str(e)}")
                
                # Resumo por Atleta, Posição e Período com minuto do máximo
                st.markdown("---")
                st.markdown("<h4>🏃 Resumo por Atleta, Posição e Período</h4>", unsafe_allow_html=True)
                
                resumo = []
                for nome in atletas_selecionados[:10]:
                    for posicao in posicoes_selecionadas[:5]:
                        for periodo in periodos_selecionados[:5]:
                            dados = df_filtrado[
                                (df_filtrado['Nome'] == nome) & 
                                (df_filtrado['Posição'] == posicao) &
                                (df_filtrado['Período'] == periodo)
                            ]
                            if not dados.empty:
                                media_grupo = dados[variavel_analise].mean()
                                desvio_grupo = dados[variavel_analise].std()
                                cv_grupo = calcular_cv(media_grupo, desvio_grupo)
                                valor_max_grupo = dados[variavel_analise].max()
                                minuto_max_grupo = dados.loc[dados[variavel_analise].idxmax(), 'Minuto']
                                
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
            
            with tab3:
                st.markdown("<h3>📦 Análise por Boxplot</h3>", unsafe_allow_html=True)
                
                # Boxplot por posição
                st.markdown("<h4>📍 Distribuição por Posição</h4>", unsafe_allow_html=True)
                
                fig_box_pos = go.Figure()
                for posicao in posicoes_selecionadas:
                    dados_pos = df_filtrado[df_filtrado['Posição'] == posicao][variavel_analise]
                    if len(dados_pos) > 0:
                        fig_box_pos.add_trace(go.Box(
                            y=dados_pos,
                            name=posicao,
                            boxmean='sd',
                            marker_color='#3498db',
                            line_color='white',
                            fillcolor='rgba(52, 152, 219, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8,
                            hovertemplate='Posição: %{x}<br>Valor: %{y:.2f}<br>Mediana: %{median:.2f}<br>IQR: %{q3:%{q1:.2f}<extra></extra>'
                        ))
                
                fig_box_pos.update_layout(
                    title=f"Distribuição de {variavel_analise} por Posição",
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3498db', size=16),
                    yaxis_title=variavel_analise,
                    showlegend=False
                )
                fig_box_pos.update_xaxes(gridcolor='#333', tickfont=dict(color='white'))
                fig_box_pos.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                # Boxplot por atleta
                st.markdown("<h4>👥 Distribuição por Atleta</h4>", unsafe_allow_html=True)
                
                atletas_plot = atletas_selecionados[:10]
                if len(atletas_selecionados) > 10:
                    st.info(f"ℹ️ Mostrando 10 de {len(atletas_selecionados)} atletas")
                
                fig_box_atl = go.Figure()
                for atleta in atletas_plot:
                    dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                    if len(dados_atl) > 0:
                        fig_box_atl.add_trace(go.Box(
                            y=dados_atl,
                            name=atleta[:15] + "..." if len(atleta) > 15 else atleta,
                            boxmean='sd',
                            marker_color='#9b59b6',
                            line_color='white',
                            fillcolor='rgba(155, 89, 182, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_atl.update_layout(
                    title=f"Distribuição de {variavel_analise} por Atleta",
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='white', size=11),
                    title_font=dict(color='#3498db', size=16),
                    yaxis_title=variavel_analise,
                    showlegend=False,
                    height=400
                )
                fig_box_atl.update_xaxes(gridcolor='#333', tickfont=dict(color='white'), tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                st.plotly_chart(fig_box_atl, use_container_width=True)
                
                # Estatísticas por atleta com explicação do IQR
                with st.expander("📊 Estatísticas detalhadas por atleta"):
                    st.markdown("""
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <h5 style="color: #3498db;">📌 O que é IQR?</h5>
                        <p style="color: #ccc;">
                            <strong>IQR (Intervalo Interquartil)</strong> é a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1). 
                            Ele representa a amplitude dos 50% centrais dos dados, sendo uma medida robusta de dispersão menos sensível a outliers.
                            Um IQR maior indica maior variabilidade nos dados.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_atletas = []
                    for atleta in atletas_selecionados[:20]:
                        dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                        if len(dados_atl) > 0:
                            q1 = dados_atl.quantile(0.25)
                            q3 = dados_atl.quantile(0.75)
                            iqr = q3 - q1
                            media_atl = dados_atl.mean()
                            desvio_atl = dados_atl.std()
                            cv_atl = calcular_cv(media_atl, desvio_atl)
                            valor_max_atl = dados_atl.max()
                            minuto_max_atl = dados_atl.loc[dados_atl.idxmax(), 'Minuto']
                            
                            stats_atletas.append({
                                'Atleta': atleta,
                                'Média': media_atl,
                                'Mediana': dados_atl.median(),
                                'Desvio Padrão': desvio_atl,
                                'CV (%)': cv_atl,
                                'Mínimo': dados_atl.min(),
                                'Q1': q1,
                                'Q3': q3,
                                'Máximo': valor_max_atl,
                                'Minuto do Máx': minuto_max_atl,
                                'IQR': iqr,
                                'Outliers': len(dados_atl[(dados_atl < q1 - 1.5*iqr) | (dados_atl > q3 + 1.5*iqr)]),
                                'N': len(dados_atl)
                            })
                    
                    df_stats = pd.DataFrame(stats_atletas)
                    st.dataframe(
                        df_stats.style.format({
                            'Média': '{:.2f}',
                            'Mediana': '{:.2f}',
                            'Desvio Padrão': '{:.2f}',
                            'CV (%)': '{:.1f}',
                            'Mínimo': '{:.2f}',
                            'Q1': '{:.2f}',
                            'Q3': '{:.2f}',
                            'Máximo': '{:.2f}',
                            'IQR': '{:.2f}',
                            'Outliers': '{:.0f}',
                            'N': '{:.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            
            with tab4:
                st.markdown("<h3>🔥 Matriz de Correlação</h3>", unsafe_allow_html=True)
                
                if len(st.session_state.variaveis_quantitativas) > 1:
                    # Selecionar variáveis para correlação
                    vars_corr = st.multiselect(
                        "Selecione as variáveis para análise de correlação:",
                        options=st.session_state.variaveis_quantitativas,
                        default=st.session_state.variaveis_quantitativas[:min(5, len(st.session_state.variaveis_quantitativas))]
                    )
                    
                    if len(vars_corr) >= 2:
                        df_corr = df_filtrado[vars_corr].corr()
                        
                        # Heatmap de correlação
                        fig_corr = px.imshow(
                            df_corr,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Matriz de Correlação",
                            zmin=-1, zmax=1
                        )
                        fig_corr.update_layout(
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            font=dict(color='white', size=11),
                            title_font=dict(color='#3498db', size=16)
                        )
                        fig_corr.update_xaxes(gridcolor='#333', tickfont=dict(color='white'))
                        fig_corr.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Tabela de correlação
                        st.markdown("<h4>📊 Tabela de Correlação</h4>", unsafe_allow_html=True)
                        
                        # Estilizar a tabela com cores
                        def style_correlation(val):
                            color = '#e74c3c' if abs(val) > 0.7 else '#f39c12' if abs(val) > 0.5 else '#3498db'
                            return f'color: {color}; font-weight: bold;'
                        
                        st.dataframe(
                            df_corr.style.format('{:.3f}').applymap(style_correlation),
                            use_container_width=True
                        )
                        
                        # Gráfico de dispersão para pares
                        if len(vars_corr) == 2:
                            st.markdown("<h4>📈 Relação entre as variáveis</h4>", unsafe_allow_html=True)
                            
                            fig_scatter = px.scatter(
                                df_filtrado,
                                x=vars_corr[0],
                                y=vars_corr[1],
                                color='Posição',
                                title=f"Relação entre {vars_corr[0]} e {vars_corr[1]}",
                                opacity=0.7,
                                trendline="ols",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig_scatter.update_layout(
                                plot_bgcolor='#1a1a1a',
                                paper_bgcolor='#1a1a1a',
                                font=dict(color='white', size=11),
                                title_font=dict(color='#3498db', size=16)
                            )
                            fig_scatter.update_xaxes(gridcolor='#333', tickfont=dict(color='white'))
                            fig_scatter.update_yaxes(gridcolor='#333', tickfont=dict(color='white'))
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            # Estatísticas de correlação
                            corr_valor = df_corr.iloc[0, 1]
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4>📊 Estatísticas de Correlação</h4>
                                <hr style="border-color: #333;">
                                <p><strong>Correlação de Pearson:</strong> {corr_valor:.3f}</p>
                                <p><strong>Interpretação:</strong> {
                                    'Correlação forte positiva' if corr_valor > 0.7 else
                                    'Correlação moderada positiva' if corr_valor > 0.5 else
                                    'Correlação fraca positiva' if corr_valor > 0.3 else
                                    'Correlação muito fraca positiva' if corr_valor > 0 else
                                    'Correlação muito fraca negativa' if corr_valor > -0.3 else
                                    'Correlação fraca negativa' if corr_valor > -0.5 else
                                    'Correlação moderada negativa' if corr_valor > -0.7 else
                                    'Correlação forte negativa'
                                }</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ Selecione pelo menos 2 variáveis para análise de correlação")
                else:
                    st.info("ℹ️ São necessárias pelo menos 2 variáveis para análise de correlação")
            
            # Dados brutos
            with st.expander("📋 Visualizar dados brutos filtrados"):
                st.dataframe(df_filtrado, use_container_width=True)
    
    # Reset do botão
    st.session_state.process_button = False

else:
    # Tela inicial com exemplo de dados
    if st.session_state.df_completo is None:
        st.info("👈 **Passo 1:** Faça upload de um ou mais arquivos CSV para começar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📋 Formato esperado do arquivo:
            
            **Primeira coluna:** Identificação no formato `Nome-Período-Minuto`  
            **Segunda coluna:** Posição do atleta  
            **Demais colunas (3+):** Variáveis numéricas para análise
            """)
            
            # Exemplo de dados
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
            st.markdown("""
            <div class="metric-container">
                <h4>📌 Componentes</h4>
                <hr style="border-color: #333;">
                <p><strong>Nome:</strong> Mariano, Maria, Joao...</p>
                <p><strong>Período:</strong> 1 TEMPO, SEGUNDO TEMPO...</p>
                <p><strong>Minuto:</strong> 00:00-01:00, 05:00-06:00...</p>
                <p><strong>Posição:</strong> Atacante, Meio-campo...</p>
            </div>
            
            <div class="metric-container" style="margin-top: 20px;">
                <h4>💡 Dica</h4>
                <hr style="border-color: #333;">
                <p>Você pode selecionar múltiplos arquivos CSV com a mesma estrutura. O sistema verificará automaticamente a compatibilidade.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar exemplo de múltiplos arquivos
        with st.expander("📁 Exemplo de uso com múltiplos arquivos"):
            st.markdown("""
            ### Carregando múltiplos arquivos:
            
            1. Prepare seus arquivos CSV com a **mesma estrutura** de colunas
            2. Selecione todos os arquivos desejados no seletor de arquivos (use Ctrl ou Shift para selecionar múltiplos)
            3. O sistema irá:
               - Verificar se todos têm a mesma estrutura de colunas
               - Concatenar os dados automaticamente
               - Manter a informação de quais arquivos foram carregados
               - Mostrar estatísticas consolidadas
            
            **Importante:** Todos os arquivos devem ter exatamente as mesmas colunas (mesmo nome e ordem).
            """)
    else:
        st.info("👈 **Passo 2:** Selecione a variável, posições, períodos, atletas e clique em 'Processar Análise'")
        
        with st.expander("📋 Preview dos dados carregados"):
            if st.session_state.upload_files_names:
                st.caption(f"**Arquivos carregados ({len(st.session_state.upload_files_names)}):**")
                for arquivo in st.session_state.upload_files_names:
                    st.write(f"- {arquivo}")
                st.markdown("---")
            
            st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
            st.caption(f"**Total de observações:** {len(st.session_state.df_completo)}")
            st.caption(f"**Variáveis disponíveis:** {', '.join(st.session_state.variaveis_quantitativas)}")
            if st.session_state.todos_posicoes:
                st.caption(f"**Posições disponíveis:** {', '.join(st.session_state.todos_posicoes)}")
            if st.session_state.todos_periodos:
                st.caption(f"**Períodos disponíveis:** {', '.join(st.session_state.todos_periodos)}")