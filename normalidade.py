import streamlit as st

# 🔥 FORÇAR LIMPEZA TOTAL DO CACHE - APAGUE APÓS USAR 🔥
st.cache_data.clear()  # Limpa cache de dados
st.cache_resource.clear()  # Limpa cache de recursos

# Opcional: limpar cache antigo (legado)
try:
    from streamlit import caching
    caching.clear_cache()
except:
    pass



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide")
st.title("Teste de Normalidade dos Dados")

# Inicializar session state
if 'df_completo' not in st.session_state:
    st.session_state.df_completo = None
if 'atletas_selecionados' not in st.session_state:
    st.session_state.atletas_selecionados = []
if 'processar' not in st.session_state:
    st.session_state.processar = False
if 'lista_atletas' not in st.session_state:
    st.session_state.lista_atletas = []
if 'selecionar_todos' not in st.session_state:
    st.session_state.selecionar_todos = True
if 'upload_concluido' not in st.session_state:
    st.session_state.upload_concluido = False

with st.sidebar:
    upload_file = st.file_uploader(
        "Escolha o arquivo:", 
        type=['csv'],
        accept_multiple_files=False,
        key="file_uploader"
    )
    
    n_classes = st.slider(
        "Número de classes (faixas):", 
        min_value=3, 
        max_value=20, 
        value=5,
        key="n_classes"
    )
    
    # Processar arquivo quando enviado
   # Processar arquivo quando enviado
if upload_file is not None and not st.session_state.upload_concluido:
    try:
        data = pd.read_csv(upload_file)

        # Validação básica
        if data.shape[1] < 2 or data.empty:
            st.error("O arquivo precisa ter pelo menos duas colunas com dados válidos.")
        else:
            # Garantir que a segunda coluna seja numérica
            data[data.columns[1]] = pd.to_numeric(
                data[data.columns[1]], errors="coerce"
            )

            # Remover linhas com NaN na coluna de valores
            data = data.dropna(subset=[data.columns[1]])

            if data.empty:
                st.error("Não há valores numéricos válidos na segunda coluna.")
            else:
                # Criar DataFrame estruturado de forma segura
                df_completo = pd.DataFrame({
                    "Nome": data.iloc[:, 0].astype(str).str.split("-").str[0].str.strip(),
                    "Minuto": data.iloc[:, 0].astype(str).str[-13:].str.strip(),
                    "Valor": data.iloc[:, 1].values
                })

                # Remover nomes vazios
                df_completo = df_completo[df_completo["Nome"].str.len() > 0]

                if df_completo.empty:
                    st.error("Nenhum dado válido encontrado após processamento.")
                else:
                    # Atualizar session state
                    st.session_state.df_completo = df_completo.copy()
                    st.session_state.lista_atletas = sorted(
                        df_completo["Nome"].unique().tolist()
                    )
                    st.session_state.atletas_selecionados = (
                        st.session_state.lista_atletas.copy()
                    )
                    st.session_state.selecionar_todos = True
                    st.session_state.processar = False
                    st.session_state.upload_concluido = True

    except Exception as e:
        st.error(f"Erro ao ler arquivo: {str(e)}")
        st.session_state.upload_concluido = False

        data = pd.read_csv(upload_file)

        # Validação básica
        if data.shape[1] < 2 or data.empty:
            st.error("O arquivo precisa ter pelo menos duas colunas com dados válidos.")
        else:
            # Garantir que a segunda coluna seja numérica
            data[data.columns[1]] = pd.to_numeric(
                data[data.columns[1]], errors="coerce"
            )

            # Remover linhas com NaN na coluna de valores
            data = data.dropna(subset=[data.columns[1]])

            if data.empty:
                st.error("Não há valores numéricos válidos na segunda coluna.")
            else:
                # Criar DataFrame estruturado de forma segura
                df_completo = pd.DataFrame({
                    "Nome": data.iloc[:, 0].astype(str).str.split("-").str[0].str.strip(),
                    "Minuto": data.iloc[:, 0].astype(str).str[-13:].str.strip(),
                    "Valor": data.iloc[:, 1].values
                })

                # Remover nomes vazios
                df_completo = df_completo[df_completo["Nome"].str.len() > 0]

                if df_completo.empty:
                    st.error("Nenhum dado válido encontrado após processamento.")
                else:
                    # Atualizar session state
                    st.session_state.df_completo = df_completo.copy()
                    st.session_state.lista_atletas = sorted(
                        df_completo["Nome"].unique().tolist()
                    )
                    st.session_state.atletas_selecionados = (
                        st.session_state.lista_atletas.copy()
                    )
                    st.session_state.selecionar_todos = True
                    st.session_state.processar = False
                    st.session_state.upload_concluido = True

    except Exception as e:
        st.error(f"Erro ao ler arquivo: {str(e)}")
        st.session_state.upload_concluido = False
    
    # Reset do estado de upload se um novo arquivo for carregado
    if upload_file is None:
        st.session_state.upload_concluido = False
        st.session_state.df_completo = None
        st.session_state.lista_atletas = []
        st.session_state.atletas_selecionados = []
    
    # --- FILTRO POR ATLETA - SOMENTE SE EXISTIREM ATLETAS NA LISTA ---
    if st.session_state.lista_atletas and len(st.session_state.lista_atletas) > 0:
        st.markdown("---")
        st.subheader("🔍 Filtro por Atleta")
        
        # Checkbox para selecionar todos
        selecionar_todos = st.checkbox(
            "Selecionar todos os atletas",
            value=st.session_state.selecionar_todos,
            key="selecionar_todos_checkbox"
        )
        
        # Atualizar session_state.selecionar_todos
        st.session_state.selecionar_todos = selecionar_todos
        
        if selecionar_todos:
            st.session_state.atletas_selecionados = st.session_state.lista_atletas.copy()
            st.info(f"✅ {len(st.session_state.lista_atletas)} atletas selecionados")
            
            # Exibir multiselect desabilitado
            st.multiselect(
                "Selecione os atletas:",
                options=st.session_state.lista_atletas,
                default=st.session_state.lista_atletas,
                disabled=True,
                key="multiselect_disabled"
            )
        else:
            # Multiselect para seleção individual
            default_values = []
            if st.session_state.atletas_selecionados:
                default_values = [a for a in st.session_state.atletas_selecionados if a in st.session_state.lista_atletas]
            
            atletas_sel = st.multiselect(
                "Selecione os atletas:",
                options=st.session_state.lista_atletas,
                default=default_values,
                key="multiselect_enabled"
            )
            
            if atletas_sel:
                st.session_state.atletas_selecionados = atletas_sel.copy()
                st.caption(f"✅ {len(atletas_sel)} atletas selecionados")
            else:
                st.session_state.atletas_selecionados = []
                st.warning("Selecione pelo menos um atleta")
    
    # Botões
    col1, col2 = st.columns([1, 1])
    with col1:
        processar_click = st.button("Processar", type="primary", use_container_width=True)
    with col2:
        limpar_click = st.button("Limpar", type="secondary", use_container_width=True)
    
    if limpar_click:
        st.session_state.processar = False
        st.session_state.df_completo = None
        st.session_state.lista_atletas = []
        st.session_state.atletas_selecionados = []
        st.session_state.selecionar_todos = True
        st.session_state.upload_concluido = False
        st.rerun()
    
    if processar_click:
        if st.session_state.df_completo is not None and st.session_state.atletas_selecionados:
            st.session_state.processar = True
        else:
            st.error("Selecione pelo menos um atleta antes de processar")
            st.session_state.processar = False

# --- ÁREA PRINCIPAL ---
if st.session_state.processar and st.session_state.df_completo is not None and st.session_state.atletas_selecionados:
    df_completo = st.session_state.df_completo
    atletas_selecionados = st.session_state.atletas_selecionados
    
    # Aplicar filtro
    df_filtrado = df_completo[df_completo['Nome'].isin(atletas_selecionados)].copy()
    
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os atletas selecionados")
        st.session_state.processar = False
    else:
        # --- GRÁFICOS ---
        col1, col2 = st.columns(2)
        with col1:
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(
                df_filtrado['Valor'], 
                bins=n_classes,
                color='blue', 
                alpha=0.7, 
                rwidth=0.85
            )
            ax_hist.set_title(f"Histograma - {len(atletas_selecionados)} atleta(s)")
            ax_hist.set_xlabel("Valores")
            ax_hist.set_ylabel("Frequência")
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        
        with col2:
            fig_qq, ax_qq = plt.subplots()
            stats.probplot(
                df_filtrado['Valor'], 
                dist='norm', 
                plot=ax_qq
            )
            ax_qq.set_title(f"QQ Plot - {len(atletas_selecionados)} atleta(s)")
            st.pyplot(fig_qq)
            plt.close(fig_qq)
        
        # --- TABELA DE FREQUÊNCIA ---
        st.subheader("📊 Tabela de Frequência")
        
        minimo = df_filtrado['Valor'].min()
        maximo = df_filtrado['Valor'].max()
        amplitude_total = maximo - minimo
        largura_classe = amplitude_total / n_classes if amplitude_total > 0 else 1
        
        limites = [minimo + i * largura_classe for i in range(n_classes + 1)]
        
        rotulos = []
        for i in range(n_classes):
            inicio = limites[i]
            fim = limites[i + 1]
            rotulos.append(f"[{inicio:.2f} - {fim:.2f})")
        
        categorias = pd.cut(
            df_filtrado['Valor'], 
            bins=limites, 
            labels=rotulos, 
            include_lowest=True, 
            right=False
        )
        
        freq_table = pd.DataFrame({
            'Faixa de Valores': rotulos,
            'Frequência': [0] * n_classes
        })
        
        contagens = categorias.value_counts()
        for i, rotulo in enumerate(rotulos):
            if rotulo in contagens.index:
                freq_table.loc[i, 'Frequência'] = int(contagens[rotulo])
        
        freq_table['Percentual (%)'] = (
            freq_table['Frequência'] / len(df_filtrado) * 100
        ).round(2)
        freq_table['Frequência Acumulada'] = freq_table['Frequência'].cumsum()
        freq_table['Percentual Acumulado (%)'] = freq_table['Percentual (%)'].cumsum()
        
        st.dataframe(
            freq_table.style.format({
                'Frequência': '{:.0f}',
                'Percentual (%)': '{:.2f}%',
                'Frequência Acumulada': '{:.0f}',
                'Percentual Acumulado (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # --- ESTATÍSTICAS DESCRITIVAS GERAIS ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mínimo", f"{minimo:.2f}")
        with col2:
            st.metric("Máximo", f"{maximo:.2f}")
        with col3:
            st.metric("Amplitude", f"{amplitude_total:.2f}")
        with col4:
            st.metric("Nº de Classes", n_classes)
        
        # --- TABELA RESUMO POR ATLETA ---
        st.subheader("🏃 Resumo por Atleta")
        
        resumo_atletas = []
        
        for nome in atletas_selecionados:
            dados_atleta = df_filtrado[df_filtrado['Nome'] == nome]
            
            if not dados_atleta.empty:
                idx_max = dados_atleta['Valor'].idxmax()
                valor_max = dados_atleta.loc[idx_max, 'Valor']
                minuto_max = dados_atleta.loc[idx_max, 'Minuto']
                
                idx_min = dados_atleta['Valor'].idxmin()
                valor_min = dados_atleta.loc[idx_min, 'Valor']
                minuto_min = dados_atleta.loc[idx_min, 'Minuto']
                
                amplitude = valor_max - valor_min
                
                resumo_atletas.append({
                    'Atleta': nome,
                    'Valor Máximo': valor_max,
                    'Minuto do Máximo': minuto_max,
                    'Valor Mínimo': valor_min,
                    'Minuto do Mínimo': minuto_min,
                    'Amplitude (Máx - Mín)': amplitude,
                    'Nº Amostras': len(dados_atleta)
                })
        
        if resumo_atletas:
            df_resumo = pd.DataFrame(resumo_atletas)
            df_resumo = df_resumo.sort_values('Atleta').reset_index(drop=True)
            
            st.dataframe(
                df_resumo.style.format({
                    'Valor Máximo': '{:.2f}',
                    'Valor Mínimo': '{:.2f}',
                    'Amplitude (Máx - Mín)': '{:.2f}',
                    'Nº Amostras': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        # --- TESTE DE NORMALIDADE ---
        st.subheader("🧪 Resultado do Teste de Normalidade")
        st.write(f"**Tamanho da amostra:** {len(df_filtrado)}")
        
        if len(df_filtrado) < 3:
            st.error("Amostra muito pequena (n < 3). Teste não aplicável.")
        elif len(df_filtrado) > 5000:
            st.warning("Amostra grande demais para Shapiro-Wilk. Usando teste D'Agostino-Pearson.")
            k2, p_value = stats.normaltest(df_filtrado['Valor'])
            if p_value < 0.0001:
                st.write(f"**Valor de p:** {p_value:.2e} (notação científica)")
            else:
                st.write(f"**Valor de p:** {p_value:.5f}")
            if p_value > 0.05:
                st.success("Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
            else:
                st.warning("Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
        else:
            shapiro_test = stats.shapiro(df_filtrado['Valor'])
            p_valor = shapiro_test.pvalue
            if p_valor < 0.0001:
                st.write(f"**Valor de p:** {p_valor:.2e} (notação científica)")
            else:
                st.write(f"**Valor de p:** {p_valor:.5f}")
            if p_valor > 0.05:
                st.success("Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
            else:
                st.warning("Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
        
        # --- GRÁFICO DE LINHA DO TEMPO (INTERATIVO COM PLOTLY) ---
        st.subheader("⏱️ Evolução Temporal dos Valores")
        
        # Ordenar por minuto
        df_tempo = df_filtrado.copy()
        df_tempo = df_tempo.sort_values('Minuto').reset_index(drop=True)
        
        # Calcular métricas
        media_valor = df_tempo['Valor'].mean()
        max_valor = df_tempo['Valor'].max()
        limiar_80 = max_valor * 0.8
        limiar_70 = max_valor * 0.7
        
        # Criar coluna de cores baseada nos limiares
        def get_color(valor):
            if valor > limiar_80:
                return 'red'
            elif valor > limiar_70:
                return 'gold'
            else:
                return 'steelblue'
        
        df_tempo['Cor'] = df_tempo['Valor'].apply(get_color)
        
        # Criar gráfico interativo com Plotly
        fig_tempo = go.Figure()
        
        # Adicionar barras
        fig_tempo.add_trace(go.Bar(
            x=df_tempo['Minuto'],
            y=df_tempo['Valor'],
            marker_color=df_tempo['Cor'],
            hovertemplate='<b>Minuto:</b> %{x}<br><b>Valor:</b> %{y:.2f}<br><extra></extra>',
            name='Valor',
            opacity=0.7
        ))
        
        # Adicionar linha da média
        fig_tempo.add_trace(go.Scatter(
            x=df_tempo['Minuto'],
            y=[media_valor] * len(df_tempo),
            mode='lines',
            name=f'Média: {media_valor:.2f}',
            line=dict(color='black', width=2, dash='dash'),
            hovertemplate=f'Média: {media_valor:.2f}<extra></extra>'
        ))
        
        # Adicionar linha de 80%
        fig_tempo.add_trace(go.Scatter(
            x=df_tempo['Minuto'],
            y=[limiar_80] * len(df_tempo),
            mode='lines',
            name=f'80% do Máx: {limiar_80:.2f}',
            line=dict(color='orange', width=1, dash='dot'),
            hovertemplate=f'80% do Máximo: {limiar_80:.2f}<extra></extra>'
        ))
        
        # Adicionar linha de 70%
        fig_tempo.add_trace(go.Scatter(
            x=df_tempo['Minuto'],
            y=[limiar_70] * len(df_tempo),
            mode='lines',
            name=f'70% do Máx: {limiar_70:.2f}',
            line=dict(color='gold', width=1, dash='dot'),
            hovertemplate=f'70% do Máximo: {limiar_70:.2f}<extra></extra>'
        ))
        
        # Layout do gráfico
        fig_tempo.update_layout(
            title=f"Evolução Temporal - {len(atletas_selecionados)} atleta(s)",
            xaxis_title="Minuto",
            yaxis_title="Valor",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            )
        )
        
        # Configurar grid
        fig_tempo.update_xaxes(gridcolor='lightgrey', gridwidth=0.5, tickangle=45)
        fig_tempo.update_yaxes(gridcolor='lightgrey', gridwidth=0.5)
        
        # Exibir gráfico
        st.plotly_chart(fig_tempo, use_container_width=True)
        
        # Legenda explicativa
        st.caption(
            "🔵 Barras azuis: valores ≤ 70% do máximo | "
            "🟡 Barras amarelas: 70% < valores ≤ 80% do máximo | "
            "🔴 Barras vermelhas: valores > 80% do máximo | "
            "⚫ Linha tracejada preta: média"
        )
        
        # --- TABELA DE FREQUÊNCIA POR PERCENTIS (10 em 10%) ---
        st.subheader("📊 Tabela de Frequência - Distribuição por Percentis (10% em 10%)")
        
        # Calcular percentis
        percentis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        limites_percentis = [df_filtrado['Valor'].quantile(p/100) for p in percentis]
        
        # Criar rótulos para as faixas percentílicas
        rotulos_percentis = []
        inicio = df_filtrado['Valor'].min()
        
        for i, p in enumerate(percentis):
            if i == 0:
                rotulos_percentis.append(f"[{inicio:.2f} - {limites_percentis[i]:.2f})")
            else:
                rotulos_percentis.append(f"[{limites_percentis[i-1]:.2f} - {limites_percentis[i]:.2f})")
        
        # Categorizar os dados por percentis
        categorias_percentis = pd.cut(
            df_filtrado['Valor'],
            bins=[inicio] + limites_percentis,
            labels=rotulos_percentis,
            include_lowest=True,
            right=False
        )
        
        # Criar tabela de frequência para percentis
        freq_percentil = pd.DataFrame({
            'Faixa de Valores (Percentil)': rotulos_percentis,
            'Percentil': [f"{p}%" for p in percentis],
            'Frequência': [0] * len(percentis)
        })
        
        # Preencher frequências
        contagens_percentis = categorias_percentis.value_counts().sort_index()
        for i, rotulo in enumerate(rotulos_percentis):
            if rotulo in contagens_percentis.index:
                freq_percentil.loc[i, 'Frequência'] = int(contagens_percentis[rotulo])
        
        # Calcular percentuais e acumulados
        freq_percentil['Percentual (%)'] = (
            freq_percentil['Frequência'] / len(df_filtrado) * 100
        ).round(2)
        freq_percentil['Frequência Acumulada'] = freq_percentil['Frequência'].cumsum()
        freq_percentil['Percentual Acumulado (%)'] = freq_percentil['Percentual (%)'].cumsum()
        
        # Exibir tabela
        st.dataframe(
            freq_percentil[['Faixa de Valores (Percentil)', 'Frequência', 'Percentual (%)', 
                           'Frequência Acumulada', 'Percentual Acumulado (%)']].style.format({
                'Frequência': '{:.0f}',
                'Percentual (%)': '{:.2f}%',
                'Frequência Acumulada': '{:.0f}',
                'Percentual Acumulado (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Estatísticas adicionais dos percentis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mediana (50%)", f"{limites_percentis[4]:.2f}")
        with col2:
            st.metric("Q1 (25%)", f"{df_filtrado['Valor'].quantile(0.25):.2f}")
        with col3:
            st.metric("Q3 (75%)", f"{df_filtrado['Valor'].quantile(0.75):.2f}")

elif st.session_state.df_completo is None:
    st.info("👈 Faça upload de um arquivo CSV para começar")
else:
    st.info("👈 Selecione os atletas e clique em 'Processar'")