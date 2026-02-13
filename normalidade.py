import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import time

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide", initial_sidebar_state="expanded")

# Tema corporativo com CSS
st.markdown("""
<style>
    /* Tema corporativo - cores legíveis */
    .stApp {
        background: #ffffff;
    }
    
    /* Cards com gradiente corporativo */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white !important;
        transition: transform 0.2s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card h3, .metric-card h2 {
        color: white !important;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(30, 60, 114, 0.2);
    }
    
    /* Títulos - cor escura para contraste */
    h1 {
        color: #1e3c72 !important;
        font-family: 'Arial', sans-serif;
        font-weight: 600;
    }
    h2 {
        color: #1e3c72 !important;
        font-family: 'Arial', sans-serif;
        font-weight: 500;
    }
    h3 {
        color: #1e3c72 !important;
        font-family: 'Arial', sans-serif;
        font-weight: 500;
    }
    h4 {
        color: #1e3c72 !important;
    }
    
    /* Abas personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        padding: 8px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        color: #1e3c72 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3c72 !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        color: #1e3c72 !important;
    }
    
    /* Botões */
    .stButton > button {
        background: #1e3c72;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #2a5298;
        box-shadow: 0 4px 8px rgba(42, 82, 152, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Métricas container */
    .metric-container {
        background: white;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-container h4, .metric-container p {
        color: #1e3c72 !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: white;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    /* Texto geral */
    p, li, .caption, .stMarkdown {
        color: #1e3c72 !important;
    }
    
    /* Links e texto informativo */
    .stInfo, .stWarning, .stError {
        color: #1e3c72 !important;
    }
    
    /* Input labels */
    .stSelectbox label, .stMultiselect label, .stSlider label {
        color: #1e3c72 !important;
    }
    
    /* Métricas do Plotly */
    .js-plotly-plot .plotly .gtitle {
        fill: #1e3c72 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 **Teste de Normalidade dos Dados - Múltiplas Variáveis**")
st.markdown("<p style='color: #1e3c72; font-size: 1.1rem;'>Análise estatística profissional com visualizações interativas</p>", unsafe_allow_html=True)

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
        status = "✅ Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados"
        cor = "#27ae60"
    else:
        status = "⚠️ Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados"
        cor = "#e74c3c"
    
    st.markdown(f"""
    <div style="background: white; border-radius: 8px; padding: 20px; border-left: 5px solid {cor}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h4 style="color: #1e3c72; margin: 0 0 10px 0;">Resultado do Teste</h4>
        <p style="color: #1e3c72; margin: 5px 0;"><strong>Teste utilizado:</strong> {nome_teste}</p>
        <p style="color: #1e3c72; margin: 5px 0;"><strong>Valor de p:</strong> <span style="color: {cor};">{p_text}</span></p>
        <p style="color: #1e3c72; margin: 5px 0;">{status}</p>
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
    <div class="metric-card" style="background: {cor_gradiente};">
        <h3 style="margin: 0; font-size: 1rem; font-weight: normal; opacity: 0.9; color: white !important;">{icone} {titulo}</h3>
        <h2 style="margin: 10px 0; font-size: 2rem; font-weight: bold; color: white !important;">{valor}</h2>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color: #1e3c72; text-align: center;'>📂 Upload dos Dados</h2>", unsafe_allow_html=True)
    
    upload_files = st.file_uploader(
        "Escolha os arquivos CSV:", 
        type=['csv'],
        accept_multiple_files=True,
        help="Selecione um ou mais arquivos CSV com a mesma estrutura. Formato: Primeira coluna = Identificação (Nome-Período-Minuto), Segunda coluna = Posição, Demais colunas = Variáveis numéricas"
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
            st.markdown("<h3 style='color: #1e3c72;'>📈 Seleção da Variável</h3>", unsafe_allow_html=True)
            
            current_index = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                current_index = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            variavel_selecionada = st.selectbox(
                "Escolha a variável para análise:",
                options=st.session_state.variaveis_quantitativas,
                index=current_index
            )
            st.session_state.variavel_selecionada = variavel_selecionada
            
            df_temp = st.session_state.df_completo[variavel_selecionada].dropna()
            if not df_temp.empty:
                st.caption(f"📊 {len(df_temp)} obs | Média: {df_temp.mean():.2f} | DP: {df_temp.std():.2f}")
        
        # Filtro por Posição
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown("<h3 style='color: #1e3c72;'>📍 Filtro por Posição</h3>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Selecionar todas as posições", value=True)
            if selecionar_todos:
                st.session_state.posicoes_selecionadas = st.session_state.todos_posicoes.copy()
            else:
                st.session_state.posicoes_selecionadas = st.multiselect(
                    "Selecione as posições:",
                    options=st.session_state.todos_posicoes,
                    default=st.session_state.posicoes_selecionadas
                )
        
        # Filtro por Período
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown("<h3 style='color: #1e3c72;'>📅 Filtro por Período</h3>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Selecionar todos os períodos", value=True)
            if selecionar_todos:
                st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                st.session_state.ordem_personalizada = st.session_state.todos_periodos.copy()
            else:
                st.session_state.periodos_selecionados = st.multiselect(
                    "Selecione os períodos:",
                    options=st.session_state.todos_periodos,
                    default=st.session_state.periodos_selecionados
                )
        
        # Filtro por Atleta (considera posição)
        if st.session_state.atletas_selecionados:
            st.markdown("---")
            st.markdown("<h3 style='color: #1e3c72;'>🔍 Filtro por Atleta</h3>", unsafe_allow_html=True)
            
            # Filtrar atletas pela posição selecionada
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
                    "Selecione os atletas:",
                    options=atletas_disponiveis,
                    default=[a for a in st.session_state.atletas_selecionados if a in atletas_disponiveis]
                )
        
        # Configurações
        st.markdown("---")
        st.markdown("<h3 style='color: #1e3c72;'>⚙️ Configurações</h3>", unsafe_allow_html=True)
        
        n_classes = st.slider("Número de classes (faixas) no histograma:", 3, 20, 5)
        
        # Botão Processar
        st.markdown("---")
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button("🔄 Processar Análise", use_container_width=True, disabled=not pode_processar):
            st.session_state.process_button = True
            st.rerun()

# --- ÁREA PRINCIPAL ---
if st.session_state.get('process_button', False) and st.session_state.df_completo is not None:
    
    with st.spinner('Gerando análises...'):
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
            st.markdown("<h2 style='color: #1e3c72; text-align: center;'>📊 Visão Geral</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Posições", len(posicoes_selecionadas), "📍", "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)")
            with col2:
                metric_card("Períodos", len(periodos_selecionados), "📅", "linear-gradient(135deg, #2a5298 0%, #1e3c72 100%)")
            with col3:
                metric_card("Atletas", len(atletas_selecionados), "👥", "linear-gradient(135deg, #2c3e50 0%, #3498db 100%)")
            with col4:
                metric_card("Observações", len(df_filtrado), "📊", "linear-gradient(135deg, #3498db 0%, #2c3e50 100%)")
            
            st.markdown("---")
            
            # Organizar em abas
            tab1, tab2, tab3 = st.tabs([
                "📊 Distribuição", 
                "📈 Estatísticas & Temporal", 
                "📦 Boxplots"
            ])
            
            with tab1:
                st.markdown("<h3 style='color: #1e3c72; text-align: center;'>Análise de Distribuição</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma formal
                    dados_hist = df_filtrado[variavel_analise].dropna()
                    
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=dados_hist,
                        nbinsx=n_classes,
                        name='Frequência',
                        marker_color='#1e3c72',
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
                        annotation_position="top"
                    )
                    
                    mediana_hist = dados_hist.median()
                    fig_hist.add_vline(
                        x=mediana_hist,
                        line_dash="dot",
                        line_color="#3498db",
                        line_width=2,
                        annotation_text=f"Mediana: {mediana_hist:.2f}",
                        annotation_position="bottom"
                    )
                    
                    fig_hist.update_layout(
                        title=f"Histograma - {variavel_analise}",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#1e3c72', size=11),
                        title_font=dict(color='#1e3c72', size=14),
                        xaxis_title=variavel_analise,
                        yaxis_title="Frequência",
                        showlegend=False,
                        bargap=0.1
                    )
                    fig_hist.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                    fig_hist.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                    
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
                        marker=dict(color='#1e3c72', size=6, opacity=0.7),
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
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#1e3c72', size=11),
                        title_font=dict(color='#1e3c72', size=14),
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis Observados"
                    )
                    fig_qq.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                    fig_qq.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # Tabela de Frequência
                st.markdown("---")
                st.markdown("<h4 style='color: #1e3c72; text-align: center;'>Tabela de Frequência</h4>", unsafe_allow_html=True)
                
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
                st.markdown("<h3 style='color: #1e3c72; text-align: center;'>Estatísticas e Evolução Temporal</h3>", unsafe_allow_html=True)
                
                # Estatísticas descritivas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    media = df_filtrado[variavel_analise].mean()
                    mediana = df_filtrado[variavel_analise].median()
                    moda = df_filtrado[variavel_analise].mode().iloc[0] if not df_filtrado[variavel_analise].mode().empty else 'N/A'
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #1e3c72; margin: 0;">📊 Medidas de Tendência</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Média:</strong> {media:.3f}</p>
                        <p><strong>Mediana:</strong> {mediana:.3f}</p>
                        <p><strong>Moda:</strong> {moda}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    desvio = df_filtrado[variavel_analise].std()
                    variancia = df_filtrado[variavel_analise].var()
                    cv = (desvio / media) * 100 if media != 0 else 0
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #1e3c72; margin: 0;">📈 Medidas de Dispersão</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Desvio Padrão:</strong> {desvio:.3f}</p>
                        <p><strong>Variância:</strong> {variancia:.3f}</p>
                        <p><strong>CV:</strong> {cv:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    q1 = df_filtrado[variavel_analise].quantile(0.25)
                    q3 = df_filtrado[variavel_analise].quantile(0.75)
                    iqr = q3 - q1
                    amplitude_total = df_filtrado[variavel_analise].max() - df_filtrado[variavel_analise].min()
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #1e3c72; margin: 0;">📐 Quartis e Amplitude</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Q1 (25%):</strong> {q1:.3f}</p>
                        <p><strong>Q3 (75%):</strong> {q3:.3f}</p>
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
                        <h4 style="color: #1e3c72; margin: 0;">📏 Assimetria</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
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
                        <h4 style="color: #1e3c72; margin: 0;">📐 Curtose</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Valor:</strong> {curtose:.3f}</p>
                        <p><strong>Interpretação:</strong> {interpretacao_curt}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Intervalo de Confiança
                st.markdown("---")
                st.markdown("<h4 style='color: #1e3c72; text-align: center;'>Intervalo de Confiança (95%) para a Média</h4>", unsafe_allow_html=True)
                
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
                        marker=dict(color='#1e3c72', size=15),
                        error_y=dict(
                            type='constant',
                            value=(ic_sup - media),
                            color='#3498db',
                            thickness=2,
                            width=10
                        ),
                        name='Média'
                    ))
                    
                    fig_ic.update_layout(
                        title="Intervalo de Confiança (95%)",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#1e3c72', size=11),
                        showlegend=False,
                        yaxis_title=variavel_analise
                    )
                    fig_ic.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                    fig_ic.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                    
                    st.plotly_chart(fig_ic, use_container_width=True)
                
                # Teste de Normalidade
                st.markdown("---")
                st.markdown("<h4 style='color: #1e3c72; text-align: center;'>Teste de Normalidade</h4>", unsafe_allow_html=True)
                
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
                
                # Gráfico Temporal
                st.markdown("---")
                st.markdown("<h4 style='color: #1e3c72; text-align: center;'>Evolução Temporal</h4>", unsafe_allow_html=True)
                
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                media_tempo = df_tempo[variavel_analise].mean()
                desvio_tempo = df_tempo[variavel_analise].std()
                n_tempo = len(df_tempo)
                erro_tempo = desvio_tempo / np.sqrt(n_tempo)
                t_tempo = stats.t.ppf(0.975, n_tempo-1) if n_tempo > 1 else 1
                ic_inf_tempo = media_tempo - t_tempo * erro_tempo
                ic_sup_tempo = media_tempo + t_tempo * erro_tempo
                
                fig_tempo = go.Figure()
                
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'],
                    y=df_tempo[variavel_analise],
                    mode='lines+markers',
                    name='Valores',
                    line=dict(color='#1e3c72', width=2),
                    marker=dict(color='#1e3c72', size=6),
                    hovertemplate='Minuto: %{x}<br>Valor: %{y:.2f}<extra></extra>'
                ))
                
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'].tolist() + df_tempo['Minuto'].tolist()[::-1],
                    y=[ic_sup_tempo] * len(df_tempo) + [ic_inf_tempo] * len(df_tempo),
                    fill='toself',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='IC 95%',
                    showlegend=True
                ))
                
                fig_tempo.add_hline(
                    y=media_tempo,
                    line_dash="dash",
                    line_color="#e74c3c",
                    line_width=2,
                    annotation_text=f"Média: {media_tempo:.2f}",
                    annotation_position="left"
                )
                
                fig_tempo.update_layout(
                    title=f"Evolução Temporal - {variavel_analise}",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#1e3c72', size=11),
                    title_font=dict(color='#1e3c72', size=14),
                    xaxis_title="Minuto",
                    yaxis_title=variavel_analise,
                    hovermode='x unified'
                )
                fig_tempo.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'), tickangle=-45)
                fig_tempo.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                
                st.plotly_chart(fig_tempo, use_container_width=True)
                
                # Resumo por Atleta, Posição e Período
                st.markdown("---")
                st.markdown("<h4 style='color: #1e3c72; text-align: center;'>Resumo por Atleta, Posição e Período</h4>", unsafe_allow_html=True)
                
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
                                resumo.append({
                                    'Atleta': nome,
                                    'Posição': posicao,
                                    'Período': periodo,
                                    f'Máx {variavel_analise}': dados[variavel_analise].max(),
                                    f'Mín {variavel_analise}': dados[variavel_analise].min(),
                                    'Amplitude': dados[variavel_analise].max() - dados[variavel_analise].min(),
                                    'Média': dados[variavel_analise].mean(),
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
                            'Nº Amostras': '{:.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            
            with tab3:
                st.markdown("<h3 style='color: #1e3c72; text-align: center;'>Análise por Boxplot</h3>", unsafe_allow_html=True)
                
                # Boxplot por posição
                st.markdown("<h4 style='color: #1e3c72;'>📍 Distribuição por Posição</h4>", unsafe_allow_html=True)
                
                fig_box_pos = go.Figure()
                for posicao in posicoes_selecionadas:
                    dados_pos = df_filtrado[df_filtrado['Posição'] == posicao][variavel_analise]
                    if len(dados_pos) > 0:
                        fig_box_pos.add_trace(go.Box(
                            y=dados_pos,
                            name=posicao,
                            boxmean='sd',
                            marker_color='#1e3c72',
                            line_color='#1e3c72',
                            fillcolor='rgba(30, 60, 114, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8,
                            hovertemplate='Posição: %{x}<br>Valor: %{y:.2f}<br>Mediana: %{median:.2f}<extra></extra>'
                        ))
                
                fig_box_pos.update_layout(
                    title=f"Distribuição de {variavel_analise} por Posição",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#1e3c72', size=11),
                    title_font=dict(color='#1e3c72', size=14),
                    yaxis_title=variavel_analise,
                    showlegend=False
                )
                fig_box_pos.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                fig_box_pos.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                # Boxplot por atleta
                st.markdown("<h4 style='color: #1e3c72;'>👥 Distribuição por Atleta</h4>", unsafe_allow_html=True)
                
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
                            marker_color='#3498db',
                            line_color='#1e3c72',
                            fillcolor='rgba(52, 152, 219, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_atl.update_layout(
                    title=f"Distribuição de {variavel_analise} por Atleta",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#1e3c72', size=11),
                    title_font=dict(color='#1e3c72', size=14),
                    yaxis_title=variavel_analise,
                    showlegend=False,
                    height=400
                )
                fig_box_atl.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'), tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#1e3c72'))
                st.plotly_chart(fig_box_atl, use_container_width=True)
                
                # Estatísticas por atleta
                with st.expander("📊 Estatísticas detalhadas por atleta"):
                    stats_atletas = []
                    for atleta in atletas_selecionados[:20]:
                        dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                        if len(dados_atl) > 0:
                            q1 = dados_atl.quantile(0.25)
                            q3 = dados_atl.quantile(0.75)
                            iqr = q3 - q1
                            stats_atletas.append({
                                'Atleta': atleta,
                                'Média': dados_atl.mean(),
                                'Mediana': dados_atl.median(),
                                'Desvio Padrão': dados_atl.std(),
                                'Mínimo': dados_atl.min(),
                                'Q1': q1,
                                'Q3': q3,
                                'Máximo': dados_atl.max(),
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
            
            # Dados brutos
            with st.expander("📋 Visualizar dados brutos filtrados"):
                st.dataframe(df_filtrado, use_container_width=True)
    
    # Reset do botão
    st.session_state.process_button = False

else:
    # Tela inicial com exemplo de dados
    if st.session_state.df_completo is None:
        st.info("👈 **Passo 1:** Faça upload de um ou mais arquivos CSV para começar")
        st.markdown("""
        ### 📋 Formato esperado do arquivo:
        
        **Primeira coluna:** Identificação no formato `Nome-Período-Minuto`  
        **Segunda coluna:** Posição do atleta (ex: Atacante, Meio-campo, Zagueiro, Goleiro)  
        **Demais colunas (3+):** Variáveis numéricas para análise
        
        **Exemplo:**
        ```
        Nome-Período-Minuto;Posição;Distancia Total;Velocidade Maxima;Aceleracao Max
        Mariano-1 TEMPO 00:00-01:00;Atacante;250;23;3.6
        Maria-SEGUNDO TEMPO 05:00-06:00;Meio-campo;127;29;4.2
        Joao-2 TEMPO 44:00-45:00;Zagueiro;200;33;4.9
        Marta-PRIMEIRO TEMPO 11:00-12:00;Atacante;90;27;3.1
        Pedro-1 TEMPO 15:00-16:00;Goleiro;45;15;2.8
        Ana-SEGUNDO TEMPO 22:00-23:00;Meio-campo;180;31;4.5
        ```
        
        **Componentes da primeira coluna:**
        - **Nome:** Primeira parte antes do primeiro hífen "-" (ex: Mariano, Maria, Joao, Marta, Pedro, Ana)
        - **Período:** Texto entre o "nome" e o 14º último caractere (ex: 1 TEMPO, SEGUNDO TEMPO, 2 TEMPO, PRIMEIRO TEMPO)
        - **Minuto:** Últimos 13 caracteres (ex: 00:00-01:00, 05:00-06:00, 44:00-45:00, 11:00-12:00)
        
        **💡 Dica:** Você pode selecionar múltiplos arquivos CSV com a **mesma estrutura** de colunas. O sistema verificará automaticamente se as estruturas são compatíveis.
        """)
        
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
            # Mostrar informação dos arquivos
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