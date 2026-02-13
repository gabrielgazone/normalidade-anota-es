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
    /* Tema corporativo */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    /* Cards com gradiente corporativo */
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
        transition: transform 0.2s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(44, 62, 80, 0.15);
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    
    /* Abas personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        color: #2c3e50;
    }
    .stTabs [aria-selected="true"] {
        background: #2c3e50 !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        color: #2c3e50;
    }
    
    /* Botões */
    .stButton > button {
        background: #2c3e50;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #3498db;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
        border-right: 1px solid #e9ecef;
    }
    
    /* Métricas */
    .metric-container {
        background: white;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Dataframe */
    .dataframe {
        background: white;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    /* Texto */
    p, li, .caption {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 **Análise de Normalidade - Dashboard Corporativo**")
st.markdown("<p style='color: #2c3e50; font-size: 1.1rem;'>Análise estatística profissional com visualizações interativas</p>", unsafe_allow_html=True)

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
        status = "✅ Não rejeita H0 - Dados normais"
        cor = "#27ae60"
    else:
        status = "⚠️ Rejeita H0 - Dados não normais"
        cor = "#e74c3c"
    
    st.markdown(f"""
    <div style="background: white; border-radius: 8px; padding: 20px; border-left: 5px solid {cor}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h4 style="color: #2c3e50; margin: 0 0 10px 0;">{status}</h4>
        <p style="color: #2c3e50; margin: 5px 0;"><strong>Teste:</strong> {nome_teste}</p>
        <p style="color: #2c3e50; margin: 5px 0;"><strong>Valor de p:</strong> <span style="color: {cor};">{p_text}</span></p>
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

def metric_card(titulo, valor, icone, cor):
    """Cria um card de métrica estilizado"""
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, {cor} 0%, {cor}dd 100%);">
        <h3 style="margin: 0; font-size: 1rem; font-weight: normal; opacity: 0.9;">{icone} {titulo}</h3>
        <h2 style="margin: 10px 0; font-size: 2rem; font-weight: bold;">{valor}</h2>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color: #2c3e50; text-align: center;'>📂 Upload dos Dados</h2>", unsafe_allow_html=True)
    
    upload_files = st.file_uploader(
        "Escolha os arquivos CSV:", 
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
            st.markdown("<h3 style='color: #2c3e50;'>📈 Variável</h3>", unsafe_allow_html=True)
            
            current_index = 0
            if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
                current_index = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
            
            variavel_selecionada = st.selectbox(
                "Selecione:",
                options=st.session_state.variaveis_quantitativas,
                index=current_index,
                label_visibility="collapsed"
            )
            st.session_state.variavel_selecionada = variavel_selecionada
        
        # Filtro por Posição
        if st.session_state.todos_posicoes:
            st.markdown("---")
            st.markdown("<h3 style='color: #2c3e50;'>📍 Posição</h3>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Todas as posições", value=True)
            if selecionar_todos:
                st.session_state.posicoes_selecionadas = st.session_state.todos_posicoes.copy()
            else:
                st.session_state.posicoes_selecionadas = st.multiselect(
                    "Selecione:",
                    options=st.session_state.todos_posicoes,
                    default=st.session_state.posicoes_selecionadas,
                    label_visibility="collapsed"
                )
        
        # Filtro por Período
        if st.session_state.todos_periodos:
            st.markdown("---")
            st.markdown("<h3 style='color: #2c3e50;'>📅 Período</h3>", unsafe_allow_html=True)
            
            selecionar_todos = st.checkbox("Todos os períodos", value=True)
            if selecionar_todos:
                st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
                st.session_state.ordem_personalizada = st.session_state.todos_periodos.copy()
            else:
                st.session_state.periodos_selecionados = st.multiselect(
                    "Selecione:",
                    options=st.session_state.todos_periodos,
                    default=st.session_state.periodos_selecionados,
                    label_visibility="collapsed"
                )
        
        # Filtro por Atleta (agora considera posição)
        if st.session_state.atletas_selecionados:
            st.markdown("---")
            st.markdown("<h3 style='color: #2c3e50;'>👤 Atleta</h3>", unsafe_allow_html=True)
            
            # Filtrar atletas pela posição selecionada
            df_temp = st.session_state.df_completo.copy()
            if st.session_state.posicoes_selecionadas:
                df_temp = df_temp[df_temp['Posição'].isin(st.session_state.posicoes_selecionadas)]
            if st.session_state.periodos_selecionados:
                df_temp = df_temp[df_temp['Período'].isin(st.session_state.periodos_selecionados)]
            
            atletas_disponiveis = sorted(df_temp['Nome'].unique())
            
            selecionar_todos = st.checkbox("Todos os atletas", value=True)
            if selecionar_todos:
                st.session_state.atletas_selecionados = atletas_disponiveis
            else:
                st.session_state.atletas_selecionados = st.multiselect(
                    "Selecione:",
                    options=atletas_disponiveis,
                    default=[a for a in st.session_state.atletas_selecionados if a in atletas_disponiveis],
                    label_visibility="collapsed"
                )
        
        # Configurações
        st.markdown("---")
        st.markdown("<h3 style='color: #2c3e50;'>⚙️ Configurações</h3>", unsafe_allow_html=True)
        
        n_classes = st.slider("Número de classes:", 3, 20, 5)
        
        # Botão Processar
        st.markdown("---")
        pode_processar = (st.session_state.variavel_selecionada and 
                         st.session_state.posicoes_selecionadas and 
                         st.session_state.periodos_selecionados and 
                         st.session_state.atletas_selecionados)
        
        if st.button("Processar Análise", use_container_width=True, disabled=not pode_processar):
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
            st.warning("⚠️ Nenhum dado encontrado")
        else:
            # Métricas principais
            st.markdown("<h2 style='color: #2c3e50; text-align: center;'>📊 Visão Geral</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Posições", len(posicoes_selecionadas), "📍", "#2c3e50")
            with col2:
                metric_card("Períodos", len(periodos_selecionados), "📅", "#34495e")
            with col3:
                metric_card("Atletas", len(atletas_selecionados), "👥", "#3498db")
            with col4:
                metric_card("Observações", len(df_filtrado), "📊", "#2980b9")
            
            st.markdown("---")
            
            # Organizar em abas
            tab1, tab2, tab3 = st.tabs([
                "📊 Distribuição", 
                "📈 Estatísticas & Temporal", 
                "📦 Boxplots"
            ])
            
            with tab1:
                st.markdown("<h3 style='color: #2c3e50; text-align: center;'>Análise de Distribuição</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma formal (versão corporativa)
                    dados_hist = df_filtrado[variavel_analise].dropna()
                    
                    # Criar histograma com estilo formal
                    fig_hist = go.Figure()
                    
                    # Adicionar histograma
                    fig_hist.add_trace(go.Histogram(
                        x=dados_hist,
                        nbinsx=n_classes,
                        name='Frequência',
                        marker_color='#2c3e50',
                        opacity=0.8,
                        hovertemplate='Faixa: %{x}<br>Frequência: %{y}<extra></extra>'
                    ))
                    
                    # Adicionar linha da média
                    media_hist = dados_hist.mean()
                    fig_hist.add_vline(
                        x=media_hist,
                        line_dash="dash",
                        line_color="#e74c3c",
                        line_width=2,
                        annotation_text=f"Média: {media_hist:.2f}",
                        annotation_position="top"
                    )
                    
                    # Adicionar linha da mediana
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
                        font=dict(color='#2c3e50', size=11),
                        title_font=dict(color='#2c3e50', size=14),
                        xaxis_title=variavel_analise,
                        yaxis_title="Frequência",
                        showlegend=False,
                        bargap=0.1,
                        hoverlabel=dict(bgcolor="#2c3e50", font_size=12)
                    )
                    fig_hist.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                    fig_hist.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # QQ Plot
                    dados_qq = df_filtrado[variavel_analise].dropna()
                    quantis_teoricos = stats.norm.ppf(np.linspace(0.01, 0.99, len(dados_qq)))
                    quantis_observados = np.sort(dados_qq)
                    
                    # Calcular R² para a linha de referência
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
                        marker=dict(color='#2c3e50', size=6, opacity=0.7),
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
                        font=dict(color='#2c3e50', size=11),
                        title_font=dict(color='#2c3e50', size=14),
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis Observados",
                        hoverlabel=dict(bgcolor="#2c3e50", font_size=12)
                    )
                    fig_qq.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                    fig_qq.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # Tabela de Frequência
                st.markdown("---")
                st.markdown("<h4 style='color: #2c3e50; text-align: center;'>Tabela de Frequência</h4>", unsafe_allow_html=True)
                
                minimo = df_filtrado[variavel_analise].min()
                maximo = df_filtrado[variavel_analise].max()
                amplitude = maximo - minimo
                largura_classe = amplitude / n_classes if amplitude > 0 else 1
                
                limites = [minimo + i * largura_classe for i in range(n_classes + 1)]
                rotulos = [f"[{limites[i]:.2f} - {limites[i+1]:.2f})" for i in range(n_classes)]
                
                categorias = pd.cut(df_filtrado[variavel_analise], bins=limites, labels=rotulos, include_lowest=True, right=False)
                contagens = categorias.value_counts()
                
                freq_table = pd.DataFrame({
                    'Faixa': rotulos,
                    'Frequência': [int(contagens.get(r, 0)) for r in rotulos],
                    'Percentual': [contagens.get(r, 0) / len(df_filtrado) * 100 for r in rotulos]
                })
                freq_table['Freq. Acum.'] = freq_table['Frequência'].cumsum()
                freq_table['Perc. Acum.'] = freq_table['Percentual'].cumsum()
                
                # Gráfico de barras da frequência
                fig_freq = px.bar(
                    freq_table,
                    x='Faixa',
                    y='Frequência',
                    title="Distribuição de Frequências",
                    text='Frequência',
                    color_discrete_sequence=['#2c3e50']
                )
                fig_freq.update_traces(textposition='outside', textfont_color='#2c3e50')
                fig_freq.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2c3e50', size=11),
                    title_font=dict(color='#2c3e50', size=14),
                    showlegend=False,
                    xaxis_tickangle=-45
                )
                fig_freq.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                fig_freq.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                st.plotly_chart(fig_freq, use_container_width=True)
                
                # Tabela
                st.dataframe(
                    freq_table.style.format({
                        'Frequência': '{:.0f}',
                        'Percentual': '{:.2f}%',
                        'Freq. Acum.': '{:.0f}',
                        'Perc. Acum.': '{:.2f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab2:
                st.markdown("<h3 style='color: #2c3e50; text-align: center;'>Estatísticas e Evolução Temporal</h3>", unsafe_allow_html=True)
                
                # Estatísticas em cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    media = df_filtrado[variavel_analise].mean()
                    desvio = df_filtrado[variavel_analise].std()
                    cv = (desvio / media) * 100 if media != 0 else 0
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #2c3e50; margin: 0;">📊 Medidas de Tendência</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Média:</strong> {media:.3f}</p>
                        <p><strong>Mediana:</strong> {df_filtrado[variavel_analise].median():.3f}</p>
                        <p><strong>Moda:</strong> {df_filtrado[variavel_analise].mode().iloc[0] if not df_filtrado[variavel_analise].mode().empty else 'N/A'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    q1 = df_filtrado[variavel_analise].quantile(0.25)
                    q3 = df_filtrado[variavel_analise].quantile(0.75)
                    iqr = q3 - q1
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #2c3e50; margin: 0;">📈 Medidas de Dispersão</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Desvio Padrão:</strong> {desvio:.3f}</p>
                        <p><strong>CV:</strong> {cv:.1f}%</p>
                        <p><strong>IQR:</strong> {iqr:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    assimetria = df_filtrado[variavel_analise].skew()
                    curtose = df_filtrado[variavel_analise].kurtosis()
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: #2c3e50; margin: 0;">📐 Forma da Distribuição</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Assimetria:</strong> {assimetria:.3f}</p>
                        <p><strong>Curtose:</strong> {curtose:.3f}</p>
                        <p><strong>Amplitude:</strong> {df_filtrado[variavel_analise].max() - df_filtrado[variavel_analise].min():.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Intervalo de Confiança
                st.markdown("---")
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
                        <h4 style="color: #2c3e50; margin: 0;">🎯 IC 95% para Média</h4>
                        <hr style="margin: 10px 0; border-color: #e9ecef;">
                        <p><strong>Limite Inferior:</strong> {ic_inf:.3f}</p>
                        <p><strong>Média:</strong> {media:.3f}</p>
                        <p><strong>Limite Superior:</strong> {ic_sup:.3f}</p>
                        <p><small>Distribuição: {dist}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_ic2:
                    # Gráfico do IC
                    fig_ic = go.Figure()
                    
                    fig_ic.add_trace(go.Scatter(
                        x=['IC 95%'],
                        y=[media],
                        mode='markers',
                        marker=dict(color='#2c3e50', size=15),
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
                        font=dict(color='#2c3e50', size=11),
                        showlegend=False,
                        yaxis_title=variavel_analise
                    )
                    fig_ic.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                    fig_ic.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                    
                    st.plotly_chart(fig_ic, use_container_width=True)
                
                # Teste de Normalidade
                st.markdown("---")
                st.markdown("<h4 style='color: #2c3e50; text-align: center;'>🧪 Teste de Normalidade</h4>", unsafe_allow_html=True)
                
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
                
                # Gráfico Temporal
                st.markdown("---")
                st.markdown("<h4 style='color: #2c3e50; text-align: center;'>⏱️ Evolução Temporal</h4>", unsafe_allow_html=True)
                
                df_tempo = df_filtrado.sort_values('Minuto').reset_index(drop=True)
                
                # Estatísticas para o gráfico
                media_tempo = df_tempo[variavel_analise].mean()
                desvio_tempo = df_tempo[variavel_analise].std()
                n_tempo = len(df_tempo)
                erro_tempo = desvio_tempo / np.sqrt(n_tempo)
                t_tempo = stats.t.ppf(0.975, n_tempo-1) if n_tempo > 1 else 1
                ic_inf_tempo = media_tempo - t_tempo * erro_tempo
                ic_sup_tempo = media_tempo + t_tempo * erro_tempo
                
                fig_tempo = go.Figure()
                
                # Linha temporal
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'],
                    y=df_tempo[variavel_analise],
                    mode='lines+markers',
                    name='Valores',
                    line=dict(color='#2c3e50', width=2),
                    marker=dict(color='#2c3e50', size=6),
                    hovertemplate='Minuto: %{x}<br>Valor: %{y:.2f}<extra></extra>'
                ))
                
                # Banda de IC
                fig_tempo.add_trace(go.Scatter(
                    x=df_tempo['Minuto'].tolist() + df_tempo['Minuto'].tolist()[::-1],
                    y=[ic_sup_tempo] * len(df_tempo) + [ic_inf_tempo] * len(df_tempo),
                    fill='toself',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='IC 95%',
                    showlegend=True
                ))
                
                # Linha da média
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
                    font=dict(color='#2c3e50', size=11),
                    title_font=dict(color='#2c3e50', size=14),
                    xaxis_title="Minuto",
                    yaxis_title=variavel_analise,
                    hovermode='x unified',
                    hoverlabel=dict(bgcolor="#2c3e50", font_size=12)
                )
                fig_tempo.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'), tickangle=-45)
                fig_tempo.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                
                st.plotly_chart(fig_tempo, use_container_width=True)
            
            with tab3:
                st.markdown("<h3 style='color: #2c3e50; text-align: center;'>Análise por Boxplot</h3>", unsafe_allow_html=True)
                
                # Boxplot por posição
                st.markdown("<h4 style='color: #2c3e50;'>📍 Distribuição por Posição</h4>", unsafe_allow_html=True)
                
                fig_box_pos = go.Figure()
                for posicao in posicoes_selecionadas:
                    dados_pos = df_filtrado[df_filtrado['Posição'] == posicao][variavel_analise]
                    if len(dados_pos) > 0:
                        fig_box_pos.add_trace(go.Box(
                            y=dados_pos,
                            name=posicao,
                            boxmean='sd',
                            marker_color='#2c3e50',
                            line_color='#2c3e50',
                            fillcolor='rgba(44, 62, 80, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8,
                            hovertemplate='Posição: %{x}<br>Valor: %{y:.2f}<br>Mediana: %{median:.2f}<extra></extra>'
                        ))
                
                fig_box_pos.update_layout(
                    title=f"Distribuição por Posição - {variavel_analise}",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2c3e50', size=11),
                    title_font=dict(color='#2c3e50', size=14),
                    yaxis_title=variavel_analise,
                    showlegend=False
                )
                fig_box_pos.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                fig_box_pos.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
                st.plotly_chart(fig_box_pos, use_container_width=True)
                
                # Boxplot por atleta (limitado)
                st.markdown("<h4 style='color: #2c3e50;'>👥 Distribuição por Atleta</h4>", unsafe_allow_html=True)
                
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
                            line_color='#2c3e50',
                            fillcolor='rgba(52, 152, 219, 0.7)',
                            jitter=0.3,
                            pointpos=-1.8,
                            opacity=0.8
                        ))
                
                fig_box_atl.update_layout(
                    title=f"Distribuição por Atleta - {variavel_analise}",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2c3e50', size=11),
                    title_font=dict(color='#2c3e50', size=14),
                    yaxis_title=variavel_analise,
                    showlegend=False,
                    height=400
                )
                fig_box_atl.update_xaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'), tickangle=-45)
                fig_box_atl.update_yaxes(gridcolor='#e9ecef', tickfont=dict(color='#2c3e50'))
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
            
            # Dados brutos
            with st.expander("📋 Dados brutos filtrados"):
                st.dataframe(df_filtrado, use_container_width=True)
    
    # Reset do botão
    st.session_state.process_button = False

else:
    # Tela inicial
    if st.session_state.df_completo is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 40px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <h1 style="font-size: 3rem; color: #2c3e50;">📊</h1>
                <h2 style="color: #2c3e50;">Dashboard de Análise Estatística</h2>
                <p style="color: #2c3e50; font-size: 1.1rem;">
                    Faça upload dos seus arquivos CSV no menu lateral para iniciar a análise.
                </p>
                <div style="margin-top: 30px; text-align: left;">
                    <p style="color: #2c3e50;"><strong>Formato esperado:</strong></p>
                    <ul style="color: #2c3e50;">
                        <li>Coluna 1: Identificação (Nome-Período-Minuto)</li>
                        <li>Coluna 2: Posição do atleta</li>
                        <li>Colunas 3+: Variáveis numéricas</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Selecione os filtros no menu lateral e clique em 'Processar Análise'")
        
        with st.expander("📋 Preview dos dados carregados"):
            if st.session_state.upload_files_names:
                st.caption(f"**Arquivos:** {', '.join(st.session_state.upload_files_names)}")
            st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
            st.caption(f"**Total:** {len(st.session_state.df_completo)} observações")
            st.caption(f"**Variáveis:** {', '.join(st.session_state.variaveis_quantitativas)}")