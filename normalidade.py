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

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Upload dos Dados")
    upload_file = st.file_uploader(
        "Escolha o arquivo CSV:", 
        type=['csv'],
        accept_multiple_files=False,
        help="Formato: Primeira coluna = Identificação (Nome-Minuto), Demais colunas = Variáveis numéricas"
    )
    
    # Processar arquivo quando enviado
    if upload_file is not None:
        try:
            # Carregar dados
            data = pd.read_csv(upload_file)
            
            # Verificar estrutura mínima
            if data.shape[1] >= 2 and not data.empty:
                # Processar primeira coluna (identificação)
                primeira_coluna = data.iloc[:, 0].astype(str)
                
                # Separar Nome e Minuto
                nomes = primeira_coluna.str.split('-').str[0].str.strip()
                
                # Extrair minuto (últimos caracteres após o último '-')
                minutos = primeira_coluna.str.split('-').str[-1].str.strip()
                
                # Identificar variáveis quantitativas (todas as colunas a partir da 2ª)
                variaveis_quant = []
                dados_quantitativos = {}
                
                for col_idx in range(1, data.shape[1]):
                    nome_var = data.columns[col_idx]
                    # Tentar converter para numérico
                    valores = pd.to_numeric(data.iloc[:, col_idx], errors='coerce')
                    
                    # Verificar se há pelo menos alguns valores não-nulos
                    if not valores.dropna().empty:
                        variaveis_quant.append(nome_var)
                        dados_quantitativos[nome_var] = valores.reset_index(drop=True)
                
                if variaveis_quant:
                    # Criar DataFrame base com identificação
                    df_completo = pd.DataFrame({
                        'Nome': nomes.reset_index(drop=True),
                        'Minuto': minutos.reset_index(drop=True)
                    })
                    
                    # Adicionar variáveis quantitativas
                    for var_nome, var_valores in dados_quantitativos.items():
                        df_completo[var_nome] = var_valores
                    
                    # Remover linhas sem nome
                    df_completo = df_completo[df_completo['Nome'].str.len() > 0]
                    
                    if not df_completo.empty:
                        st.session_state.df_completo = df_completo
                        st.session_state.variaveis_quantitativas = variaveis_quant
                        st.session_state.atletas_selecionados = sorted(df_completo['Nome'].unique())
                        
                        # Selecionar primeira variável por padrão
                        if variaveis_quant and st.session_state.variavel_selecionada is None:
                            st.session_state.variavel_selecionada = variaveis_quant[0]
                        
                        st.success(f"✅ Arquivo carregado! {len(variaveis_quant)} variáveis identificadas.")
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
        
        # Determinar o índice atual
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
        
        # Mostrar estatísticas básicas da variável
        df_temp = st.session_state.df_completo[variavel_selecionada].dropna()
        if not df_temp.empty:
            st.caption(f"📊 {len(df_temp)} observações | Média: {df_temp.mean():.2f} | Desvio: {df_temp.std():.2f}")
    
    # --- FILTRO POR ATLETA ---
    if st.session_state.df_completo is not None:
        st.markdown("---")
        st.header("🔍 Filtro por Atleta")
        
        lista_atletas = sorted(st.session_state.df_completo['Nome'].unique())
        
        # Inicializar ou atualizar atletas_selecionados se estiver vazio
        if not st.session_state.atletas_selecionados:
            st.session_state.atletas_selecionados = lista_atletas.copy()
        
        # Checkbox para selecionar todos
        selecionar_todos = st.checkbox(
            "Selecionar todos os atletas",
            value=len(st.session_state.atletas_selecionados) == len(lista_atletas),
            key="selecionar_todos"
        )
        
        if selecionar_todos:
            st.session_state.atletas_selecionados = lista_atletas.copy()
            st.info(f"✅ {len(lista_atletas)} atletas selecionados")
        else:
            # Multiselect para seleção individual
            atletas_sel = st.multiselect(
                "Selecione os atletas:",
                options=lista_atletas,
                default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else lista_atletas[:1],
                key="multiselect_ativos"
            )
            
            # Atualizar session state com a seleção atual
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
    
    # --- BOTÃO PROCESSAR ---
    # Inicializar variável de controle
    pode_processar = True
    
    # Validação pré-processamento
    if st.session_state.df_completo is not None:
        if not st.session_state.atletas_selecionados:
            st.error("❌ Selecione pelo menos um atleta antes de processar")
            pode_processar = False
        if not st.session_state.variavel_selecionada:
            st.error("❌ Selecione uma variável para análise")
            pode_processar = False
    else:
        pode_processar = False
    
    # Botão processar (desabilitado se não puder processar)
    process_button = st.button(
        "🔄 Processar Análise", 
        type="primary", 
        use_container_width=True,
        disabled=not pode_processar
    )

# --- ÁREA PRINCIPAL ---
if process_button and st.session_state.df_completo is not None and st.session_state.atletas_selecionados and st.session_state.variavel_selecionada:
    
    df_completo = st.session_state.df_completo
    atletas_selecionados = st.session_state.atletas_selecionados
    variavel_analise = st.session_state.variavel_selecionada
    
    # Aplicar filtros
    df_filtrado = df_completo[df_completo['Nome'].isin(atletas_selecionados)].copy()
    df_filtrado = df_filtrado.dropna(subset=[variavel_analise])
    
    if df_filtrado.empty:
        st.warning("⚠️ Nenhum dado encontrado para os atletas e variável selecionados")
    else:
        # Título da análise
        st.header(f"📊 Análise de Normalidade: **{variavel_analise}**")
        st.caption(f"🎯 {len(atletas_selecionados)} atleta(s) | {len(df_filtrado)} observações totais")
        
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
        
        # --- TABELA RESUMO POR ATLETA ---
        st.subheader("🏃 Resumo por Atleta")
        
        resumo_atletas = []
        
        for nome in atletas_selecionados:
            dados_atleta = df_filtrado[df_filtrado['Nome'] == nome]
            
            if not dados_atleta.empty:
                idx_max = dados_atleta[variavel_analise].idxmax()
                valor_max = dados_atleta.loc[idx_max, variavel_analise]
                minuto_max = dados_atleta.loc[idx_max, 'Minuto']
                
                idx_min = dados_atleta[variavel_analise].idxmin()
                valor_min = dados_atleta.loc[idx_min, variavel_analise]
                minuto_min = dados_atleta.loc[idx_min, 'Minuto']
                
                amplitude = valor_max - valor_min
                
                resumo_atletas.append({
                    'Atleta': nome,
                    f'Máx {variavel_analise}': valor_max,
                    'Minuto do Máx': minuto_max,
                    f'Mín {variavel_analise}': valor_min,
                    'Minuto do Mín': minuto_min,
                    'Amplitude': amplitude,
                    'Média': dados_atleta[variavel_analise].mean(),
                    'Nº Amostras': len(dados_atleta)
                })
        
        if resumo_atletas:
            df_resumo = pd.DataFrame(resumo_atletas)
            df_resumo = df_resumo.sort_values('Atleta').reset_index(drop=True)
            
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
        
        # --- GRÁFICO DE LINHA DO TEMPO ---
        st.subheader("⏱️ Evolução Temporal dos Valores")
        
        # Ordenar por minuto
        df_tempo = df_filtrado.copy()
        df_tempo = df_tempo.sort_values('Minuto').reset_index(drop=True)
        
        # Calcular média e limiar de 80%
        media_valor = df_tempo[variavel_analise].mean()
        limiar_80 = df_tempo[variavel_analise].max() * 0.8
        
        # Criar gráfico
        fig_tempo, ax_tempo = plt.subplots(figsize=(14, 6))
        
        # Definir cores baseadas no limiar de 80%
        cores = ['red' if valor > limiar_80 else 'steelblue' for valor in df_tempo[variavel_analise]]
        
        # Plotar barras
        bars = ax_tempo.bar(
            range(len(df_tempo)),
            df_tempo[variavel_analise],
            color=cores,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Configurar eixo X com minutos
        ax_tempo.set_xticks(range(len(df_tempo)))
        ax_tempo.set_xticklabels(
            df_tempo['Minuto'], 
            rotation=45, 
            ha='right',
            fontsize=8
        )
        
        # Adicionar linha média tracejada em preto
        ax_tempo.axhline(
            y=media_valor,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'Média: {media_valor:.2f}'
        )
        
        # Adicionar linha do limiar
        ax_tempo.axhline(
            y=limiar_80,
            color='orange',
            linestyle=':',
            linewidth=1,
            alpha=0.5,
            label=f'80% do Máx: {limiar_80:.2f}'
        )
        
        # Títulos e labels
        ax_tempo.set_title(f"Evolução Temporal - {variavel_analise} - {len(atletas_selecionados)} atleta(s)", fontsize=14, fontweight='bold')
        ax_tempo.set_xlabel("Minuto", fontsize=12)
        ax_tempo.set_ylabel(variavel_analise, fontsize=12)
        
        # Legenda
        ax_tempo.legend(loc='upper right')
        
        # Grid para melhor legibilidade
        ax_tempo.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Exibir gráfico
        st.pyplot(fig_tempo)
        plt.close(fig_tempo)
        
        # Legenda explicativa
        st.caption(
            "🔵 Barras azuis: valores ≤ 80% do máximo | "
            "🔴 Barras vermelhas: valores > 80% do máximo | "
            "⚫ Linha tracejada preta: média | "
            "🟠 Linha pontilhada laranja: 80% do valor máximo"
        )
        
        # --- DADOS BRUTOS (EXPANSÍVEL) ---
        with st.expander("📋 Visualizar dados brutos filtrados"):
            st.dataframe(df_filtrado, use_container_width=True)

elif not process_button:
    if st.session_state.df_completo is None:
        st.info("👈 **Passo 1:** Faça upload de um arquivo CSV para começar")
        st.markdown("""
        ### 📋 Formato esperado do arquivo:
        
        **Primeira coluna:** Identificação no formato `Nome-Minuto`  
        **Demais colunas:** Variáveis numéricas para análise
        
        **Exemplo:**
        ```
        Identificacao,Potencia,Frequencia,VO2Max
        Joao-00:30,250,145,45.2
        Joao-01:00,245,148,44.8
        Maria-00:30,230,152,42.1
        ```
        """)
    else:
        st.info("👈 **Passo 2:** Selecione a variável, os atletas e clique em 'Processar Análise'")
        
        # Mostrar preview dos dados carregados
        with st.expander("📋 Preview dos dados carregados"):
            st.dataframe(st.session_state.df_completo.head(10), use_container_width=True)
            st.caption(f"**Variáveis disponíveis:** {', '.join(st.session_state.variaveis_quantitativas)}")