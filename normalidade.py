import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import time

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide", initial_sidebar_state="expanded")

# Tema personalizado com CSS
st.markdown("""
<style>
    /* Tema escuro gradiente */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
        color: white;
    }
    
    /* Cards com gradiente */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        text-align: center;
        color: white;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Títulos com efeito glow */
    h1, h2, h3 {
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    /* Abas personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 50px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Expander personalizado */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Dataframe personalizado */
    .dataframe {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Botões com gradiente */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar personalizada */
    .css-1d391kg {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Métricas */
    .metric-container {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ **Teste de Normalidade dos Dados - Análise Avançada**")
st.markdown("<p style='color: #a8b2d1; font-size: 1.2rem;'>Dashboard interativo para análise estatística com visualizações dinâmicas</p>", unsafe_allow_html=True)

# Inicializar session state para manter os dados entre interações
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
if 'screen_width' not in st.session_state:
    st.session_state.screen_width = 1200

# --- FUNÇÕES AUXILIARES ---
def interpretar_teste(p_valor, nome_teste):
    """Função auxiliar para interpretar resultados do teste de normalidade"""
    if p_valor < 0.0001:
        p_text = f"{p_valor:.2e}"
    else:
        p_text = f"{p_valor:.5f}"
    
    if p_valor > 0.05:
        cor = "#4CAF50"
        emoji = "✅"
        texto = "Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados"
    else:
        cor = "#FF9800"
        emoji = "⚠️"
        texto = "Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados"
    
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px; border-left: 5px solid {cor};">
        <h4 style="color: white;">{emoji} Resultado do Teste</h4>
        <p style="color: #a8b2d1;"><strong>Teste utilizado:</strong> {nome_teste}</p>
        <p style="color: #a8b2d1;"><strong>Valor de p:</strong> <span style="color: {cor};">{p_text}</span></p>
        <p style="color: #a8b2d1;">{texto}</p>
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

def criar_colunas_responsivas(num_itens):
    """Ajusta número de colunas baseado no tamanho da tela"""
    screen_width = st.session_state.get('screen_width', 1200)
    
    if screen_width < 768:
        return 1
    elif screen_width < 992:
        return 2
    else:
        return min(4, num_itens)

def metric_card(titulo, valor, icone, cor_gradiente):
    """Cria um card de métrica estilizado"""
    st.markdown(f"""
    <div class="metric-card" style="background: {cor_gradiente};">
        <h3 style="margin: 0; font-size: 1.2rem;">{icone} {titulo}</h3>
        <h2 style="margin: 10px 0; font-size: 2.5rem; font-weight: bold;">{valor}</h2>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>📂 Upload dos Dados</h2>", unsafe_allow_html=True)
    
    upload_files = st.file_uploader(
        "Escolha os arquivos CSV:", 
        type=['csv'],
        accept_multiple_files=True,
        help="Selecione um ou mais arquivos CSV com a mesma estrutura"
    )
    
    # Processar arquivos quando enviados
    if upload_files:
        with st.spinner('🔄 Processando arquivos...'):
            time.sleep(1)  # Animação
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
                            arquivos_invalidos.append(f"{uploaded_file.name} (menos de 3 colunas)")
                    except Exception as e:
                        arquivos_invalidos.append(f"{uploaded_file.name} (erro)")
                
                if dataframes:
                    estruturas_ok, estrutura_referencia = verificar_estruturas_arquivos(dataframes)
                    
                    if not estruturas_ok:
                        st.error("❌ Arquivos com estruturas diferentes!")
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
                                
                                st.success(f"✅ {len(upload_files)} arquivo(s) processado(s)!")
                                st.balloons()
                                
                                if periodos_unicos:
                                    st.caption(f"📌 Períodos: {', '.join(periodos_unicos[:3])}...")
                                if posicoes_unicas:
                                    st.caption(f"📍 Posições: {', '.join(posicoes_unicas[:3])}...")
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
    
    # Filtros
    if st.session_state.df_completo is not None:
        st.markdown("---")
        
        # Seleção de Variável
        if st.session_state.variaveis_quantitativas:
            st.markdown("<h3 style='text-align: center;'>📈 Seleção da Variável</h3>", unsafe_allow_html=True)
            
            current_index = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                current_index = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            variavel_selecionada = st.selectbox(
                "Escolha a variável:",
                options=st.session_state.variaveis_quantitativas,
                index=current_index
            )
            st.session_state.variavel_selecionada = variavel_selecionada
        
        # Filtro por Posição
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>📍 Filtro por Posição</h3>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Selecionar todas", value=True)
            if selecionar_todos:
                st.session_state.posicoes_selecionadas = st.session_state.todos_posicoes.copy()
            else:
                st.session_state.posicoes_selecionadas = st.multiselect(
                    "Selecione:",
                    options=st.session_state.todos_posicoes,
                    default=st.session_state.posicoes_selecionadas
                )
        
        # Filtro por Período
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>📅 Filtro por Período</h3>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Selecionar todos períodos", value=True)
            if selecionar_todos:
                st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                st.session_state.ordem_personalizada = st.session_state.todos_periodos.copy()
            else:
                st.session_state.periodos_selecionados = st.multiselect(
                    "Selecione:",
                    options=st.session_state.todos_periodos,
                    default=st.session_state.periodos_selecionados
                )
        
        # Filtro por Atleta
        if st.session_state.atletas_selecionados:
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>🔍 Filtro por Atleta</h3>", unsafe_allow_html=True)
            
            # Atualizar lista baseada nos outros filtros
            df_temp = st.session_state.df_completo.copy()
            if st.session_state.posicoes_selecionadas:
                df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
            if st.session_state.periodos_selecionados:
                df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
            
            atletas_disponiveis = sorted(df_temp['Nome'].unique())
            
            selecionar_todos = st.checkbox("Selecionar todos atletas", value=True)
            if selecionar_todos:
                st.session_state.atletas_selecionados = atletas_disponiveis
            else:
                st.session_state.atletas_selecionados = st.multiselect(
                    "Selecione:",
                    options=atletas_disponiveis,
                    default=[a for a in st.session_state.atletas_selecionados if a in atletas_disponiveis]
                )
        
        # Configurações
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>⚙️ Configurações</h3>", unsafe_allow_html=True)
        
        n_classes = st.slider("Número de classes:", 3, 20, 5)
        
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
        time.sleep(1)
        
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
            st.warning("⚠️ Nenhum dado encontrado")
        else:
            # Métricas principais em cards
            st.markdown("<h2 style='text-align: center;'>📊 Visão Geral</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Posições", len(posicoes_selecionadas), "📍", 
                          "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
            with col2:
                metric_card("Períodos", len(periodos_selecionados), "📅",
                          "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)")
            with col3:
                metric_card("Atletas", len(atletas_selecionados), "👥",
                          "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)")
            with col4:
                metric_card("Observações", len(df_filtrado), "📊",
                          "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)")
            
            st.markdown("---")
            
            # Organizar em abas
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Distribuição", "📈 Evolução Temporal", "📦 Boxplots", 
                "🔬 Estatísticas", "🔥 Correlações"
            ])
            
            with tab1:
                st.markdown("<h3 style='text-align: center;'>📊 Análise de Distribuição</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma interativo
                    fig_hist = px.histogram(
                        df_filtrado, 
                        x=variavel_analise,
                        nbins=n_classes,
                        title=f"Histograma - {variavel_analise}",
                        color_discrete_sequence=['#667eea'],
                        opacity=0.8,
                        marginal="box"  # Adiciona boxplot na margem
                    )
                    fig_hist.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white',
                        showlegend=False,
                        hoverlabel=dict(bgcolor="#667eea", font_size=12)
                    )
                    fig_hist.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                    fig_hist.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # QQ Plot interativo
                    dados_qq = df_filtrado[variavel_analise].dropna()
                    quantis_teoricos = stats.norm.ppf(np.linspace(0.01, 0.99, len(dados_qq)))
                    quantis_observados = np.sort(dados_qq)
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=quantis_observados,
                        mode='markers',
                        name='Dados',
                        marker=dict(color='#667eea', size=8, opacity=0.6)
                    ))
                    
                    # Linha de referência
                    z = np.polyfit(quantis_teoricos, quantis_observados, 1)
                    linha_ref = np.poly1d(z)
                    fig_qq.add_trace(go.Scatter(
                        x=quantis_teoricos,
                        y=linha_ref(quantis_teoricos),
                        mode='lines',
                        name='Referência',
                        line=dict(color='#f093fb', width=2, dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title=f"QQ Plot - {variavel_analise}",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white',
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis Observados",
                        hoverlabel=dict(bgcolor="#667eea", font_size=12)
                    )
                    fig_qq.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                    fig_qq.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # Tabela de Frequência
                st.markdown("---")
                st.markdown("<h3 style='text-align: center;'>📋 Tabela de Frequência</h3>", unsafe_allow_html=True)
                
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
                    'Percentual': [(contagens.get(r, 0) / len(df_filtrado) * 100) for r in rotulos]
                })
                freq_table['Frequência Acumulada'] = freq_table['Frequência'].cumsum()
                freq_table['Percentual Acumulado'] = freq_table['Percentual'].cumsum()
                
                # Gráfico de barras da frequência
                fig_freq = px.bar(
                    freq_table,
                    x='Faixa de Valores',
                    y='Frequência',
                    title="Distribuição de Frequências",
                    color='Frequência',
                    color_continuous_scale=['#667eea', '#764ba2'],
                    text='Frequência'
                )
                fig_freq.update_traces(textposition='outside')
                fig_freq.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white',
                    showlegend=False,
                    xaxis_tickangle=-45
                )
                fig_freq.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_freq.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig_freq, use_container_width=True)
                
                # Tabela
                st.dataframe(
                    freq_table.style.format({
                        'Frequência': '{:.0f}',
                        'Percentual': '{:.2f}%',
                        'Frequência Acumulada': '{:.0f}',
                        'Percentual Acumulado': '{:.2f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab2:
                st.markdown("<h3 style='text-align: center;'>⏱️ Evolução Temporal</h3>", unsafe_allow_html=True)
                
                # Ordenar dados
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                # Calcular estatísticas
                media_valor = df_tempo[variavel_analise].mean()
                desvio_padrao = df_tempo[variavel_analise].std()
                n_total = len(df_tempo)
                erro_padrao = desvio_padrao / np.sqrt(n_total)
                limiar_80 = df_tempo[variavel_analise].max() * 0.8
                
                # IC
                t_critico = stats.t.ppf(0.975, n_total-1) if n_total > 1 else 1
                ic_inferior = media_valor - t_critico * erro_padrao
                ic_superior = media_valor + t_critico * erro_padrao
                
                # Gráfico temporal interativo
                fig_tempo = go.Figure()
                
                # Barras coloridas
                cores = ['#ff6b6b' if v > limiar_80 else '#667eea' for v in df_tempo[variavel_analise]]
                
                fig_tempo.add_trace(go.Bar(
                    x=df_tempo['Minuto'],
                    y=df_tempo[variavel_analise],
                    marker_color=cores,
                    name=variavel_analise,
                    hovertemplate='Minuto: %{x}<br>Valor: %{y:.2f}<extra></extra>',
                    opacity=0.8
                ))
                
                # Linhas de referência
                fig_tempo.add_hline(
                    y=media_valor,
                    line_dash="dash",
                    line_color="white",
                    annotation_text=f"Média: {media_valor:.2f}",
                    annotation_position="top left",
                    line_width=2
                )
                
                fig_tempo.add_hline(
                    y=limiar_80,
                    line_dash="dot",
                    line_color="#ffa502",
                    annotation_text=f"80% Máx: {limiar_80:.2f}",
                    annotation_position="top right",
                    line_width=2
                )
                
                # Banda de IC
                fig_tempo.add_hrect(
                    y0=ic_inferior,
                    y1=ic_superior,
                    fillcolor="rgba(102, 126, 234, 0.2)",
                    line_width=0,
                    annotation_text=f"IC 95% [{ic_inferior:.2f}, {ic_superior:.2f}]",
                    annotation_position="bottom right"
                )
                
                fig_tempo.update_layout(
                    title=f"Evolução Temporal - {variavel_analise}",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white',
                    xaxis_title="Minuto",
                    yaxis_title=variavel_analise,
                    hovermode='x unified',
                    showlegend=False,
                    xaxis_tickangle=-45
                )
                fig_tempo.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_tempo.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig_tempo, use_container_width=True)
                
                # Estatísticas temporais
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #a8b2d1;">📈 Tendência</h4>
                        <p style="color: white; font-size: 1.2rem;">{variavel_analise}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #a8b2d1;">📊 Variação</h4>
                        <p style="color: white; font-size: 1.2rem;">CV: {(desvio_padrao/media_valor*100):.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #a8b2d1;">🎯 Amplitude</h4>
                        <p style="color: white; font-size: 1.2rem;">{df_tempo[variavel_analise].max() - df_tempo[variavel_analise].min():.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("<h3 style='text-align: center;'>📦 Análise por Boxplot</h3>", unsafe_allow_html=True)
                
                # Boxplot por posição
                st.markdown("<h4 style='text-align: center;'>📍 Distribuição por Posição</h4>", unsafe_allow_html=True)
                
                fig_box_pos = go.Figure()
                for posicao in posicoes_selecionadas:
                    dados_pos = df_filtrado[df_filtrado['Posição'] == posicao][variavel_analise]
                    if len(dados_pos) > 0:
                        fig_box_pos.add_trace(go.Box(
                            y=dados_pos,
                            name=posicao,
                            boxmean='sd',
                            marker_color='#667eea',
                            line_color='white',
                            fillcolor='rgba(102, 126, 234, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_pos.update_layout(
                    title=f"Distribuição de {variavel_analise} por Posição",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white',
                    yaxis_title=variavel_analise,
                    showlegend=False
                )
                fig_box_pos.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_box_pos.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                # Boxplot por atleta
                st.markdown("<h4 style='text-align: center;'>👥 Distribuição por Atleta</h4>", unsafe_allow_html=True)
                
                # Limitar número de atletas para não sobrecarregar
                atletas_plot = atletas_selecionados[:15]  # Máximo 15 atletas
                if len(atletas_selecionados) > 15:
                    st.info(f"ℹ️ Mostrando apenas 15 de {len(atletas_selecionados)} atletas")
                
                fig_box_atleta = go.Figure()
                for atleta in atletas_plot:
                    dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                    if len(dados_atl) > 0:
                        fig_box_atleta.add_trace(go.Box(
                            y=dados_atl,
                            name=atleta[:15] + "..." if len(atleta) > 15 else atleta,
                            boxmean='sd',
                            marker_color='#f093fb',
                            line_color='white',
                            fillcolor='rgba(240, 147, 251, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_atleta.update_layout(
                    title=f"Distribuição de {variavel_analise} por Atleta",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white',
                    yaxis_title=variavel_analise,
                    showlegend=False,
                    height=max(400, len(atletas_plot) * 30)
                )
                fig_box_atleta.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_box_atleta.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig_box_atleta, use_container_width=True)
                
                # Estatísticas por atleta
                with st.expander("📊 Estatísticas detalhadas por atleta"):
                    stats_atletas = []
                    for atleta in atletas_selecionados:
                        dados_atl = df_filtrado[df_filtrado['Nome'] == atleta][variavel_analise]
                        if len(dados_atl) > 0:
                            q1 = dados_atl.quantile(0.25)
                            q3 = dados_atl.quantile(0.75)
                            iqr = q3 - q1
                            stats_atletas.append({
                                'Atleta': atleta,
                                'Média': dados_atl.mean(),
                                'Mediana': dados_atl.median(),
                                'DP': dados_atl.std(),
                                'Mín': dados_atl.min(),
                                'Q1': q1,
                                'Q3': q3,
                                'Máx': dados_atl.max(),
                                'IQR': iqr,
                                'Outliers': len(dados_atl[(dados_atl < q1 - 1.5*iqr) | (dados_atl > q3 + 1.5*iqr)]),
                                'N': len(dados_atl)
                            })
                    
                    df_stats = pd.DataFrame(stats_atletas)
                    st.dataframe(
                        df_stats.style.format({
                            'Média': '{:.2f}',
                            'Mediana': '{:.2f}',
                            'DP': '{:.2f}',
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
            
            with tab4:
                st.markdown("<h3 style='text-align: center;'>🔬 Estatísticas Avançadas</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Estatísticas descritivas
                    st.markdown("""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px;">
                        <h4 style="color: white;">📊 Estatísticas Descritivas</h4>
                    """, unsafe_allow_html=True)
                    
                    stats_desc = {
                        'Mínimo': df_filtrado[variavel_analise].min(),
                        'Máximo': df_filtrado[variavel_analise].max(),
                        'Amplitude': df_filtrado[variavel_analise].max() - df_filtrado[variavel_analise].min(),
                        'Média': df_filtrado[variavel_analise].mean(),
                        'Mediana': df_filtrado[variavel_analise].median(),
                        'Desvio Padrão': df_filtrado[variavel_analise].std(),
                        'Assimetria': df_filtrado[variavel_analise].skew(),
                        'Curtose': df_filtrado[variavel_analise].kurtosis(),
                        'Q1 (25%)': df_filtrado[variavel_analise].quantile(0.25),
                        'Q3 (75%)': df_filtrado[variavel_analise].quantile(0.75)
                    }
                    
                    for nome, valor in stats_desc.items():
                        st.markdown(f"""
                        <p style="color: #a8b2d1; margin: 5px 0;">
                            <strong>{nome}:</strong> <span style="color: white; float: right;">{valor:.3f}</span>
                        </p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    # Intervalo de Confiança
                    st.markdown("""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px;">
                        <h4 style="color: white;">🎯 Intervalo de Confiança (95%)</h4>
                    """, unsafe_allow_html=True)
                    
                    media = df_filtrado[variavel_analise].mean()
                    desvio = df_filtrado[variavel_analise].std()
                    n = len(df_filtrado)
                    erro = desvio / np.sqrt(n)
                    
                    if n > 30:
                        z = stats.norm.ppf(0.975)
                        ic_inf = media - z * erro
                        ic_sup = media + z * erro
                        dist = "Normal"
                    else:
                        t = stats.t.ppf(0.975, n-1)
                        ic_inf = media - t * erro
                        ic_sup = media + t * erro
                        dist = "t-Student"
                    
                    st.markdown(f"""
                    <p style="color: #a8b2d1;"><strong>Média:</strong> <span style="color: white;">{media:.3f}</span></p>
                    <p style="color: #a8b2d1;"><strong>Erro Padrão:</strong> <span style="color: white;">{erro:.3f}</span></p>
                    <p style="color: #a8b2d1;"><strong>IC Inferior:</strong> <span style="color: #ff6b6b;">{ic_inf:.3f}</span></p>
                    <p style="color: #a8b2d1;"><strong>IC Superior:</strong> <span style="color: #4CAF50;">{ic_sup:.3f}</span></p>
                    <p style="color: #a8b2d1;"><strong>Distribuição:</strong> <span style="color: white;">{dist}</span></p>
                    <p style="color: #a8b2d1; font-size: 0.9rem; margin-top: 10px;">
                        📌 Há 95% de confiança de que a verdadeira média populacional está entre {ic_inf:.3f} e {ic_sup:.3f}
                    </p>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Teste de Normalidade
                st.markdown("<h4 style='text-align: center;'>🧪 Teste de Normalidade</h4>", unsafe_allow_html=True)
                
                dados_teste = df_filtrado[variavel_analise].dropna()
                n_teste = len(dados_teste)
                
                if n_teste < 3:
                    st.error("❌ Amostra muito pequena (n < 3)")
                elif n_teste > 5000:
                    k2, p = stats.normaltest(dados_teste)
                    interpretar_teste(p, "D'Agostino-Pearson")
                else:
                    try:
                        shapiro = stats.shapiro(dados_teste)
                        interpretar_teste(shapiro.pvalue, "Shapiro-Wilk")
                    except:
                        st.error("❌ Erro no teste")
                
                # Resumo por grupos
                st.markdown("---")
                st.markdown("<h4 style='text-align: center;'>🏃 Resumo por Atleta, Posição e Período</h4>", unsafe_allow_html=True)
                
                # Tabela resumo com cores alternadas
                resumo = []
                for nome in atletas_selecionados[:10]:  # Limitar para performance
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
                                    'Média': dados[variavel_analise].mean(),
                                    'Máx': dados[variavel_analise].max(),
                                    'Mín': dados[variavel_analise].min(),
                                    'Amplitude': dados[variavel_analise].max() - dados[variavel_analise].min(),
                                    'N': len(dados)
                                })
                
                if resumo:
                    df_resumo = pd.DataFrame(resumo)
                    st.dataframe(
                        df_resumo.style.format({
                            'Média': '{:.2f}',
                            'Máx': '{:.2f}',
                            'Mín': '{:.2f}',
                            'Amplitude': '{:.2f}',
                            'N': '{:.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            
            with tab5:
                if len(st.session_state.variaveis_quantitativas) > 1:
                    st.markdown("<h3 style='text-align: center;'>🔥 Matriz de Correlação</h3>", unsafe_allow_html=True)
                    
                    # Selecionar variáveis para correlação
                    vars_corr = st.multiselect(
                        "Selecione as variáveis para correlação:",
                        options=st.session_state.variaveis_quantitativas,
                        default=st.session_state.variaveis_quantitativas[:min(5, len(st.session_state.variaveis_quantitativas))]
                    )
                    
                    if len(vars_corr) >= 2:
                        df_corr = df_filtrado[vars_corr].corr()
                        
                        # Heatmap interativo
                        fig_corr = px.imshow(
                            df_corr,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Matriz de Correlação",
                            zmin=-1, zmax=1
                        )
                        fig_corr.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            title_font_color='white'
                        )
                        fig_corr.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                        fig_corr.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Gráfico de dispersão para pares
                        if len(vars_corr) == 2:
                            fig_scatter = px.scatter(
                                df_filtrado,
                                x=vars_corr[0],
                                y=vars_corr[1],
                                color='Posição',
                                title=f"Relação entre {vars_corr[0]} e {vars_corr[1]}",
                                opacity=0.7,
                                trendline="ols"
                            )
                            fig_scatter.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white',
                                title_font_color='white'
                            )
                            fig_scatter.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                            fig_scatter.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                            st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("ℹ️ São necessárias pelo menos 2 variáveis para análise de correlação")
            
            # Dados brutos
            with st.expander("📋 Visualizar dados brutos filtrados"):
                st.dataframe(df_filtrado, use_container_width=True)
            
            # Animação de conclusão
            st.balloons()
    
    # Reset do botão
    st.session_state.process_button = False

else:
    # Tela inicial
    if st.session_state.df_completo is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <h1 style="font-size: 4rem;">👋</h1>
                <h2 style="color: white;">Bem-vindo ao Dashboard de Análise!</h2>
                <p style="color: #a8b2d1; font-size: 1.2rem;">
                    Faça upload dos seus arquivos CSV no menu lateral para começar a análise.
                </p>
                <div style="margin-top: 30px;">
                    <p style="color: #a8b2d1;">📁 Formato esperado:</p>
                    <p style="color: #a8b2d1;">Coluna 1: Nome-Período-Minuto</p>
                    <p style="color: #a8b2d1;">Coluna 2: Posição</p>
                    <p style="color: #a8b2d1;">Colunas 3+: Variáveis numéricas</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Selecione os filtros e clique em 'Processar Análise'")
        
        with st.expander("📋 Preview dos dados"):
            if st.session_state.upload_files_names:
                st.caption(f"**Arquivos:** {', '.join(st.session_state.upload_files_names)}")
            st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
            st.caption(f"**Total:** {len(st.session_state.df_completo)} observações")
            st.caption(f"**Variáveis:** {', '.join(st.session_state.variaveis_quantitativas)}")