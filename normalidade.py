import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide")
st.title("📊 Teste de Normalidade dos Dados - Múltiplas Variáveis")

# Inicializar session state para manter os dados entre interações
if 'df_completo' not in st.session_state:
    st.session_state.df_completo = None
if 'variaveis_quantitativas' not in st.session_state:
    st.session_state.variaveis_quantitativas = []
if 'variavel_selecionada' not in st.session_state:
    st.session_state.variavel_selecionada = None
if 'atletas_selecionados' not in st.session_state:
    st.session_state.atletas_selecionados = []
if 'periodos_selecionados' not in st.session_state:
    st.session_state.periodos_selecionados = []
if 'todos_periodos' not in st.session_state:
    st.session_state.todos_periodos = []
if 'process_button_disabled' not in st.session_state:
    st.session_state.process_button_disabled = True
if 'ordem_personalizada' not in st.session_state:
    st.session_state.ordem_personalizada = []

# --- FUNÇÕES AUXILIARES ---
def interpretar_teste(p_valor, nome_teste):
    """Função auxiliar para interpretar resultados do teste de normalidade"""
    st.write(f"**Teste utilizado:** {nome_teste}")
    if p_valor < 0.0001:
        st.write(f"**Valor de p:** {p_valor:.2e} (notação científica)")
    else:
        st.write(f"**Valor de p:** {p_valor:.5f}")
    
    if p_valor > 0.05:
        st.success("✅ Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
    else:
        st.warning("⚠️ Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")

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

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Upload dos Dados")
    upload_file = st.file_uploader(
        "Escolha o arquivo CSV:", 
        type=['csv'],
        accept_multiple_files=False,
        help="Formato: Primeira coluna = Identificação (Nome-Período-Minuto), Demais colunas = Variáveis numéricas"
    )
    
    # Processar arquivo quando enviado
    if upload_file is not None:
        try:
            data = pd.read_csv(upload_file)
            
            if data.shape[1] >= 2 and not data.empty:
                primeira_coluna = data.iloc[:, 0].astype(str)
                
                nomes = primeira_coluna.str.split('-').str[0].str.strip()
                minutos = primeira_coluna.str[-13:].str.strip()
                periodos = primeira_coluna.apply(extrair_periodo)
                
                periodos_unicos = sorted([p for p in periodos.unique() if p and p.strip() != ""])
                
                variaveis_quant = []
                dados_quantitativos = {}
                
                for col_idx in range(1, data.shape[1]):
                    nome_var = data.columns[col_idx]
                    valores = pd.to_numeric(data.iloc[:, col_idx], errors='coerce')
                    
                    if not valores.dropna().empty:
                        variaveis_quant.append(nome_var)
                        dados_quantitativos[nome_var] = valores.reset_index(drop=True)
                
                if variaveis_quant:
                    df_completo = pd.DataFrame({
                        'Nome': nomes.reset_index(drop=True),
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
                        st.session_state.todos_periodos = periodos_unicos
                        st.session_state.periodos_selecionados = periodos_unicos.copy()
                        st.session_state.ordem_personalizada = periodos_unicos.copy()
                        
                        if variaveis_quant and st.session_state.variavel_selecionada is None:
                            st.session_state.variavel_selecionada = variaveis_quant[0]
                        
                        st.success(f"✅ Arquivo carregado! {len(variaveis_quant)} variáveis, {len(periodos_unicos)} períodos.")
                        
                        if periodos_unicos:
                            st.info(f"📌 Períodos: {', '.join(periodos_unicos[:3])}{'...' if len(periodos_unicos) > 3 else ''}")
                else:
                    st.error("❌ Nenhuma variável numérica válida encontrada nas colunas 2+")
            else:
                st.error("❌ Arquivo deve ter pelo menos 2 colunas")
                
        except Exception as e:
            st.error(f"❌ Erro ao ler arquivo: {str(e)}")
    
    # --- SELEÇÃO DE VARIÁVEL ---
    if st.session_state.df_completo is not None and st.session_state.variaveis_quantitativas:
        st.markdown("---")
        st.header("📈 Seleção da Variável")
        
        current_index = 0
        if st.session_state.variavel_selecionada in st.session_state.variaveis_quantitativas:
            current_index = st.session_state.variaveis_quantitativas.index(st.session_state.variavel_selecionada)
        
        variavel_selecionada = st.selectbox(
            "Escolha a variável para análise:",
            options=st.session_state.variaveis_quantitativas,
            index=current_index,
            key="select_variavel"
        )
        st.session_state.variavel_selecionada = variavel_selecionada
        
        df_temp = st.session_state.df_completo[variavel_selecionada].dropna()
        if not df_temp.empty:
            st.caption(f"📊 {len(df_temp)} obs | Média: {df_temp.mean():.2f} | DP: {df_temp.std():.2f}")
    
    # --- FILTRO POR PERÍODO ---
    if st.session_state.df_completo is not None and st.session_state.todos_periodos:
        st.markdown("---")
        st.header("📅 Filtro por Período")
        
        lista_periodos = st.session_state.todos_periodos
        
        if not st.session_state.periodos_selecionados and lista_periodos:
            st.session_state.periodos_selecionados = lista_periodos.copy()
        
        selecionar_todos_periodos = st.checkbox(
            "Selecionar todos os períodos",
            value=len(st.session_state.periodos_selecionados) == len(lista_periodos) if lista_periodos else True,
            key="selecionar_todos_periodos"
        )
        
        if selecionar_todos_periodos:
            st.session_state.periodos_selecionados = lista_periodos.copy()
            st.session_state.ordem_personalizada = lista_periodos.copy()
            st.info(f"✅ {len(lista_periodos)} períodos selecionados")
        else:
            periodos_sel = st.multiselect(
                "Selecione os períodos:",
                options=lista_periodos,
                default=st.session_state.periodos_selecionados if st.session_state.periodos_selecionados else lista_periodos[:1],
                key="multiselect_periodos"
            )
            
            if periodos_sel:
                st.session_state.periodos_selecionados = periodos_sel
                st.session_state.ordem_personalizada = periodos_sel.copy()
                st.caption(f"✅ {len(periodos_sel)} períodos selecionados")
            else:
                st.session_state.periodos_selecionados = []
                st.warning("⚠️ Selecione pelo menos um período")
    
    # --- FILTRO POR ATLETA ---
    if st.session_state.df_completo is not None:
        st.markdown("---")
        st.header("🔍 Filtro por Atleta")
        
        df_temp_atletas = st.session_state.df_completo.copy()
        
        if st.session_state.periodos_selecionados:
            df_temp_atletas = df_temp_atletas[df_temp_atletas['Período'].isin(st.session_state.periodos_selecionados)]
        
        lista_atletas = sorted(df_temp_atletas['Nome'].unique())
        
        if lista_atletas:
            if st.session_state.atletas_selecionados:
                st.session_state.atletas_selecionados = [a for a in st.session_state.atletas_selecionados if a in lista_atletas]
            
            if not st.session_state.atletas_selecionados:
                st.session_state.atletas_selecionados = lista_atletas.copy()
        else:
            st.session_state.atletas_selecionados = []
        
        selecionar_todos_atletas = st.checkbox(
            "Selecionar todos os atletas",
            value=len(st.session_state.atletas_selecionados) == len(lista_atletas) if lista_atletas else True,
            key="selecionar_todos_atletas"
        )
        
        if selecionar_todos_atletas:
            st.session_state.atletas_selecionados = lista_atletas.copy()
            st.info(f"✅ {len(lista_atletas)} atletas selecionados")
        else:
            atletas_sel = st.multiselect(
                "Selecione os atletas:",
                options=lista_atletas,
                default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else lista_atletas[:1] if lista_atletas else [],
                key="multiselect_atletas"
            )
            
            if atletas_sel:
                st.session_state.atletas_selecionados = atletas_sel
                st.caption(f"✅ {len(atletas_sel)} atletas selecionados")
            else:
                st.session_state.atletas_selecionados = []
                st.warning("⚠️ Selecione pelo menos um atleta")
    
    # --- CONFIGURAÇÕES DO GRÁFICO ---
    st.markdown("---")
    st.header("⚙️ Configurações")
    
    n_classes = st.slider(
        "Número de classes (faixas) no histograma:", 
        min_value=3, 
        max_value=20, 
        value=5,
        help="Define quantas barras o histograma terá"
    )
    
    # ============= ORDENAÇÃO DO GRÁFICO TEMPORAL =============
    st.markdown("---")
    st.header("🔄 Ordenação do Eixo X")
    
    opcoes_ordenacao = ["⏫ Minuto (Crescente)", "⏬ Minuto (Decrescente)", 
                        "📋 Período (A-Z)", "📋 Período (Z-A)", 
                        "🎯 Ordem Personalizada"]
    
    ordem_opcao = st.radio(
        "Ordem do gráfico temporal:",
        options=opcoes_ordenacao,
        index=0,
        key="ordem_temporal"
    )
    
    # ORDEM PERSONALIZADA - VERSÃO 100% FUNCIONAL
    if ordem_opcao == "🎯 Ordem Personalizada" and st.session_state.periodos_selecionados:
        st.markdown("##### Defina a ordem dos períodos:")
        
        # Garantir que ordem_personalizada esteja sincronizada
        periodos_validos = st.session_state.periodos_selecionados
        
        if not st.session_state.ordem_personalizada:
            st.session_state.ordem_personalizada = periodos_validos.copy()
        else:
            # Remover períodos que não estão mais selecionados
            st.session_state.ordem_personalizada = [p for p in st.session_state.ordem_personalizada if p in periodos_validos]
            # Adicionar novos períodos no final
            for p in periodos_validos:
                if p not in st.session_state.ordem_personalizada:
                    st.session_state.ordem_personalizada.append(p)
        
        # MOSTRAR ORDEM ATUAL
        st.markdown("**Ordem atual no gráfico:**")
        for i, p in enumerate(st.session_state.ordem_personalizada):
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{i+1}. {p}")
        
        st.markdown("---")
        
        # CRIAR SELECT BOXES PARA CADA POSIÇÃO
        st.markdown("**Selecione a nova ordem:**")
        
        # Usar um formulário com chave única baseada nos períodos
        form_key = f"ordem_form_{len(periodos_validos)}_{abs(hash(tuple(periodos_validos)))}"
        
        with st.form(key=form_key):
            nova_ordem = []
            for i in range(len(periodos_validos)):
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.write(f"**Posição {i+1}:**")
                with col2:
                    valor_atual = st.session_state.ordem_personalizada[i] if i < len(st.session_state.ordem_personalizada) else periodos_validos[0]
                    if valor_atual not in periodos_validos:
                        valor_atual = periodos_validos[0]
                    
                    periodo_escolhido = st.selectbox(
                        f"pos_{i}",
                        options=periodos_validos,
                        index=periodos_validos.index(valor_atual),
                        key=f"ordem_select_{i}_{form_key}",
                        label_visibility="collapsed"
                    )
                    nova_ordem.append(periodo_escolhido)
            
            # BOTÃO DE SUBMIT DO FORMULÁRIO
            submit_button = st.form_submit_button("✅ Aplicar Nova Ordem", use_container_width=True, type="primary")
            
            if submit_button:
                # Verificar se todos os períodos estão presentes uma única vez
                if len(set(nova_ordem)) == len(nova_ordem) and set(nova_ordem) == set(periodos_validos):
                    st.session_state.ordem_personalizada = nova_ordem.copy()
                    st.success("✅ Ordem atualizada com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Cada período deve aparecer exatamente uma vez!")
    # ================================================================
    
    # --- BOTÃO PROCESSAR ---
    pode_processar = True
    
    if st.session_state.df_completo is not None:
        if 'variavel_selecionada' not in st.session_state or not st.session_state.variavel_selecionada:
            st.error("❌ Selecione uma variável para análise")
            pode_processar = False
        
        if 'periodos_selecionados' not in st.session_state or not st.session_state.periodos_selecionados:
            st.error("❌ Selecione pelo menos um período")
            pode_processar = False
            
        if 'atletas_selecionados' not in st.session_state or not st.session_state.atletas_selecionados:
            st.error("❌ Selecione pelo menos um atleta")
            pode_processar = False
    else:
        pode_processar = False
    
    process_button = st.button(
        "🔄 Processar Análise", 
        type="primary", 
        use_container_width=True,
        disabled=not pode_processar
    )

# --- ÁREA PRINCIPAL ---
if process_button and st.session_state.df_completo is not None and st.session_state.atletas_selecionados and st.session_state.periodos_selecionados and st.session_state.variavel_selecionada:
    
    df_completo = st.session_state.df_completo
    atletas_selecionados = st.session_state.atletas_selecionados
    periodos_selecionados = st.session_state.periodos_selecionados
    variavel_analise = st.session_state.variavel_selecionada
    
    df_filtrado = df_completo[
        df_completo['Nome'].isin(atletas_selecionados) & 
        df_completo['Período'].isin(periodos_selecionados)
    ].copy()
    
    df_filtrado = df_filtrado.dropna(subset=[variavel_analise])
    
    if df_filtrado.empty:
        st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados")
    else:
        st.header(f"📊 Análise de Normalidade: **{variavel_analise}**")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Períodos", f"{len(periodos_selecionados)}")
            if len(periodos_selecionados) <= 3:
                st.caption(f"{', '.join(periodos_selecionados)}")
        with col_f2:
            st.metric("Atletas", f"{len(atletas_selecionados)}")
        with col_f3:
            st.metric("Observações", f"{len(df_filtrado)}")
        
        # --- GRÁFICOS ---
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            ax_hist.hist(
                df_filtrado[variavel_analise], 
                bins=n_classes,
                color='steelblue', 
                alpha=0.7, 
                rwidth=0.85,
                edgecolor='black',
                linewidth=0.5
            )
            ax_hist.set_title(f"Histograma - {variavel_analise}", fontsize=14, fontweight='bold')
            ax_hist.set_xlabel(variavel_analise, fontsize=12)
            ax_hist.set_ylabel("Frequência", fontsize=12)
            ax_hist.grid(axis='y', alpha=0.3, linestyle='--')
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        
        with col2:
            fig_qq, ax_qq = plt.subplots(figsize=(8, 5))
            stats.probplot(
                df_filtrado[variavel_analise], 
                dist='norm', 
                plot=ax_qq
            )
            ax_qq.set_title(f"QQ Plot - {variavel_analise}", fontsize=14, fontweight='bold')
            ax_qq.set_xlabel("Quantis Teóricos", fontsize=12)
            ax_qq.set_ylabel("Quantis Observados", fontsize=12)
            ax_qq.grid(alpha=0.3, linestyle='--')
            st.pyplot(fig_qq)
            plt.close(fig_qq)
        
        # --- TABELA DE FREQUÊNCIA ---
        st.subheader("📋 Tabela de Frequência")
        
        minimo = df_filtrado[variavel_analise].min()
        maximo = df_filtrado[variavel_analise].max()
        amplitude_total = maximo - minimo
        largura_classe = amplitude_total / n_classes if amplitude_total > 0 else 1
        
        limites = [minimo + i * largura_classe for i in range(n_classes + 1)]
        
        rotulos = []
        for i in range(n_classes):
            inicio = limites[i]
            fim = limites[i + 1]
            rotulos.append(f"[{inicio:.2f} - {fim:.2f})")
        
        categorias = pd.cut(
            df_filtrado[variavel_analise], 
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
        
        freq_table['Percentual (%)'] = (freq_table['Frequência'] / len(df_filtrado) * 100).round(2)
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
        
        # --- ESTATÍSTICAS DESCRITIVAS ---
        st.subheader("📊 Estatísticas Descritivas")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mínimo", f"{minimo:.2f}")
        with col2:
            st.metric("Máximo", f"{maximo:.2f}")
        with col3:
            st.metric("Amplitude", f"{amplitude_total:.2f}")
        with col4:
            st.metric("Média", f"{df_filtrado[variavel_analise].mean():.2f}")
        with col5:
            st.metric("Desvio Padrão", f"{df_filtrado[variavel_analise].std():.2f}")
        
        col6, col7, col8, col9, col10 = st.columns(5)
        with col6:
            st.metric("Mediana", f"{df_filtrado[variavel_analise].median():.2f}")
        with col7:
            st.metric("Assimetria", f"{df_filtrado[variavel_analise].skew():.3f}")
        with col8:
            st.metric("Curtose", f"{df_filtrado[variavel_analise].kurtosis():.3f}")
        with col9:
            q1 = df_filtrado[variavel_analise].quantile(0.25)
            st.metric("Q1 (25%)", f"{q1:.2f}")
        with col10:
            q3 = df_filtrado[variavel_analise].quantile(0.75)
            st.metric("Q3 (75%)", f"{q3:.2f}")
        
        # --- TABELA RESUMO POR ATLETA E PERÍODO ---
        st.subheader("🏃 Resumo por Atleta e Período")
        
        resumo_atletas_periodos = []
        
        for nome in atletas_selecionados:
            for periodo in periodos_selecionados:
                dados_atleta_periodo = df_filtrado[
                    (df_filtrado['Nome'] == nome) & 
                    (df_filtrado['Período'] == periodo)
                ]
                
                if not dados_atleta_periodo.empty:
                    idx_max = dados_atleta_periodo[variavel_analise].idxmax()
                    valor_max = dados_atleta_periodo.loc[idx_max, variavel_analise]
                    minuto_max = dados_atleta_periodo.loc[idx_max, 'Minuto']
                    
                    idx_min = dados_atleta_periodo[variavel_analise].idxmin()
                    valor_min = dados_atleta_periodo.loc[idx_min, variavel_analise]
                    minuto_min = dados_atleta_periodo.loc[idx_min, 'Minuto']
                    
                    amplitude = valor_max - valor_min
                    
                    resumo_atletas_periodos.append({
                        'Atleta': nome,
                        'Período': periodo,
                        f'Máx {variavel_analise}': valor_max,
                        'Minuto do Máx': minuto_max,
                        f'Mín {variavel_analise}': valor_min,
                        'Minuto do Mín': minuto_min,
                        'Amplitude': amplitude,
                        'Média': dados_atleta_periodo[variavel_analise].mean(),
                        'Nº Amostras': len(dados_atleta_periodo)
                    })
        
        if resumo_atletas_periodos:
            df_resumo = pd.DataFrame(resumo_atletas_periodos)
            df_resumo = df_resumo.sort_values(['Atleta', 'Período']).reset_index(drop=True)
            
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
        
        # --- TESTE DE NORMALIDADE ---
        st.subheader("🧪 Resultado do Teste de Normalidade")
        
        dados_teste = df_filtrado[variavel_analise].dropna()
        n_amostra = len(dados_teste)
        
        st.write(f"**Tamanho da amostra:** {n_amostra}")
        st.write(f"**Variável analisada:** {variavel_analise}")
        
        if n_amostra < 3:
            st.error("❌ Amostra muito pequena (n < 3). Teste não aplicável.")
        elif n_amostra > 5000:
            st.info("ℹ️ Amostra grande demais para Shapiro-Wilk. Usando teste D'Agostino-Pearson.")
            try:
                k2, p_value = stats.normaltest(dados_teste)
                interpretar_teste(p_value, "D'Agostino-Pearson")
            except:
                st.warning("⚠️ Teste D'Agostino-Pearson não pôde ser calculado. Usando Kolmogorov-Smirnov.")
                try:
                    _, p_value = stats.kstest(dados_teste, 'norm', args=(dados_teste.mean(), dados_teste.std()))
                    interpretar_teste(p_value, "Kolmogorov-Smirnov")
                except:
                    st.error("❌ Não foi possível realizar nenhum teste de normalidade.")
        else:
            try:
                shapiro_test = stats.shapiro(dados_teste)
                p_valor = shapiro_test.pvalue
                interpretar_teste(p_valor, "Shapiro-Wilk")
            except Exception as e:
                st.error(f"❌ Erro no teste Shapiro-Wilk: {str(e)}")
        
        # --- GRÁFICO DE LINHA DO TEMPO COM ORDENAÇÃO FLEXÍVEL ---
        st.subheader("⏱️ Evolução Temporal dos Valores")
        
        # APLICAR ORDENAÇÃO CONFORME ESCOLHA DO USUÁRIO
        df_tempo = df_filtrado.copy()
        
        ordem_escolhida = st.session_state.ordem_temporal
        
        if ordem_escolhida == "⏫ Minuto (Crescente)":
            df_tempo = df_tempo.sort_values('Minuto')
        elif ordem_escolhida == "⏬ Minuto (Decrescente)":
            df_tempo = df_tempo.sort_values('Minuto', ascending=False)
        elif ordem_escolhida == "📋 Período (A-Z)":
            df_tempo = df_tempo.sort_values(['Período', 'Minuto'])
        elif ordem_escolhida == "📋 Período (Z-A)":
            df_tempo = df_tempo.sort_values(['Período', 'Minuto'], ascending=[False, True])
        elif ordem_escolhida == "🎯 Ordem Personalizada":
            # Usar ordem personalizada definida pelo usuário
            if st.session_state.ordem_personalizada:
                ordem_map = {periodo: i for i, periodo in enumerate(st.session_state.ordem_personalizada)}
                df_tempo['ordem_temp'] = df_tempo['Período'].map(ordem_map)
                df_tempo = df_tempo.sort_values(['ordem_temp', 'Minuto'])
                df_tempo = df_tempo.drop('ordem_temp', axis=1)
        
        df_tempo = df_tempo.reset_index(drop=True)
        
        # Calcular média e limiar de 80%
        media_valor = df_tempo[variavel_analise].mean()
        limiar_80 = df_tempo[variavel_analise].max() * 0.8
        
        # Criar gráfico
        fig_tempo, ax_tempo = plt.subplots(figsize=(14, 6))
        
        cores = ['red' if valor > limiar_80 else 'steelblue' for valor in df_tempo[variavel_analise]]
        
        bars = ax_tempo.bar(
            range(len(df_tempo)),
            df_tempo[variavel_analise],
            color=cores,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax_tempo.set_xticks(range(len(df_tempo)))
        ax_tempo.set_xticklabels(
            df_tempo['Minuto'], 
            rotation=45, 
            ha='right',
            fontsize=8
        )
        
        ax_tempo.axhline(
            y=media_valor,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'Média: {media_valor:.2f}'
        )
        
        ax_tempo.axhline(
            y=limiar_80,
            color='orange',
            linestyle=':',
            linewidth=1,
            alpha=0.5,
            label=f'80% do Máx: {limiar_80:.2f}'
        )
        
        ax_tempo.set_title(f"Evolução Temporal - {variavel_analise}", fontsize=14, fontweight='bold')
        ax_tempo.set_xlabel("Minuto", fontsize=12)
        ax_tempo.set_ylabel(variavel_analise, fontsize=12)
        ax_tempo.legend(loc='upper right')
        ax_tempo.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig_tempo)
        plt.close(fig_tempo)
        
        st.caption(
            "🔵 Barras azuis: valores ≤ 80% do máximo | "
            "🔴 Barras vermelhas: valores > 80% do máximo | "
            "⚫ Linha tracejada preta: média | "
            "🟠 Linha pontilhada laranja: 80% do valor máximo | "
            f"**Ordenação:** {ordem_escolhida}"
        )
        
        with st.expander("📋 Visualizar dados brutos filtrados"):
            st.dataframe(df_filtrado, use_container_width=True)

elif not process_button:
    if st.session_state.df_completo is None:
        st.info("👈 **Passo 1:** Faça upload de um arquivo CSV para começar")
        st.markdown("""
        ### 📋 Formato esperado do arquivo:
        
        **Primeira coluna:** Identificação no formato `Nome-Período-Minuto`  
        **Demais colunas:** Variáveis numéricas para análise
        
        **Exemplo:**
        ```
        Nome-Período-Minuto; Distancia Total; Velocidade Maxima; Acc Max
        Mariano-1 TEMPO 00:00-01:00,250,23,3.6
        Maria-SEGUNDO TEMPO 05:00-06:00,127,29,4.2
        Pele-2 TEMPO 44:00-45:00,200,33,4.9
        Marta-PRIMEIRO TEMPO 11:00-12:00,90,27,3.1
        ```
        
        **Componentes da primeira coluna:**
        - **Nome:** Primeira parte antes do primeiro hífen "-"
        - **Período:** Texto entre o "nome" e o 14º último caractere
        - **Minuto:** Últimos 13 caracteres
        """)
    else:
        st.info("👈 **Passo 2:** Selecione a variável, períodos, atletas e clique em 'Processar Análise'")
        
        with st.expander("📋 Preview dos dados carregados"):
            st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
            st.caption(f"**Variáveis disponíveis:** {', '.join(st.session_state.variaveis_quantitativas)}")
            if st.session_state.todos_periodos:
                st.caption(f"**Períodos disponíveis:** {', '.join(st.session_state.todos_periodos)}")