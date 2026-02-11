import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide")
st.title("Teste de Normalidade dos Dados")

# Inicializar session state para manter os dados entre interações
if 'df_completo' not in st.session_state:
    st.session_state.df_completo = None
if 'atletas_selecionados' not in st.session_state:
    st.session_state.atletas_selecionados = []
if 'periodos_selecionados' not in st.session_state:
    st.session_state.periodos_selecionados = []

with st.sidebar:
    upload_file = st.file_uploader(
        "Escolha o arquivo:", 
        type=['csv'],
        accept_multiple_files=False
    )
    
    n_classes = st.slider(
        "Número de classes (faixas):", 
        min_value=3, 
        max_value=20, 
        value=5,
        key="n_classes_hist"
    )
    
    # Processar arquivo quando enviado
    if upload_file is not None:
        try:
            data = pd.read_csv(upload_file)
            
            if data.shape[1] >= 2 and not data.empty and not data.iloc[:, 1].isnull().all():
                # Processar dados da primeira coluna
                primeira_coluna = data.iloc[:, 0].astype(str)
                
                # EXTRAIR NOME: tudo antes do primeiro "-"
                nomes = primeira_coluna.str.split('-').str[0].str.strip()
                
                # EXTRAIR PERÍODO: entre o primeiro "-" e os últimos 13 caracteres
                apos_primeiro_hifen = primeira_coluna.str.split('-', 1).str[1]
                periodo = apos_primeiro_hifen.str[:-13].str.strip()
                
                # EXTRAIR MINUTO: últimos 13 caracteres
                minutos = primeira_coluna.str[-13:].str.strip()
                
                # Dados de teste (segunda coluna)
                dados_teste = data.iloc[:, 1].dropna().reset_index(drop=True)
                
                # Garantir que todos os vetores tenham o mesmo tamanho
                nomes = nomes.reset_index(drop=True)
                periodo = periodo.reset_index(drop=True)
                minutos = minutos.reset_index(drop=True)
                
                if len(nomes) == len(periodo) == len(minutos) == len(dados_teste):
                    df_completo = pd.DataFrame({
                        'Nome': nomes,
                        'Periodo': periodo,
                        'Minuto': minutos,
                        'Valor': dados_teste
                    })
                    
                    # Remover linhas com nome ou período vazio
                    df_completo = df_completo[
                        (df_completo['Nome'].str.len() > 0) & 
                        (df_completo['Periodo'].str.len() > 0)
                    ].reset_index(drop=True)
                    
                    if not df_completo.empty:
                        st.session_state.df_completo = df_completo
                        st.session_state.atletas_selecionados = sorted(df_completo['Nome'].unique())
                        st.session_state.periodos_selecionados = sorted(df_completo['Periodo'].unique())
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {str(e)}")
    
    # --- FILTRO POR ATLETA ---
    if st.session_state.df_completo is not None:
        st.markdown("---")
        st.subheader("🔍 Filtro por Atleta")
        
        lista_atletas = sorted(st.session_state.df_completo['Nome'].unique())
        
        selecionar_todos_atletas = st.checkbox(
            "Selecionar todos os atletas",
            value=True,
            key="selecionar_todos_atletas"
        )
        
        if selecionar_todos_atletas:
            st.session_state.atletas_selecionados = lista_atletas
            # Mostrar apenas um texto informativo quando "selecionar todos" está ativo
            st.info(f"✅ {len(lista_atletas)} atletas selecionados")
        else:
            atletas_sel = st.multiselect(
                "Selecione os atletas:",
                options=lista_atletas,
                default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else [],key="multiselect_atletas")
            st.session_state.atletas_selecionados = atletas_sel if atletas_sel else []
        
        if st.session_state.atletas_selecionados:
            st.caption(f"✅ {len(st.session_state.atletas_selecionados)} atletas selecionados")
    
    # --- FILTRO POR PERÍODO ---
    if st.session_state.df_completo is not None:
        st.markdown("---")
        st.subheader("📅 Filtro por Período")
        
        lista_periodos = sorted(st.session_state.df_completo['Periodo'].unique())
        
        selecionar_todos_periodos = st.checkbox(
            "Selecionar todos os períodos",
            value=True,
            key="selecionar_todos_periodos"
        )
        
        if selecionar_todos_periodos:
            st.session_state.periodos_selecionados = lista_periodos
            st.info(f"✅ {len(lista_periodos)} períodos selecionados")
        else:
            periodos_sel = st.multiselect(
                "Selecione os períodos:",
                options=lista_periodos,
                default=st.session_state.periodos_selecionados if st.session_state.periodos_selecionados else [],
                key="multiselect_periodos"
            )
            st.session_state.periodos_selecionados = periodos_sel if periodos_sel else []
        
        if st.session_state.periodos_selecionados:
            st.caption(f"✅ {len(st.session_state.periodos_selecionados)} períodos selecionados")
    
    # --- ÚNICO BOTÃO PROCESSAR ---
    # NÃO HÁ NENHUM OUTRO BOTÃO ANTES DESTA LINHA
    botao_desabilitado = True
    mostrar_warning_atleta = False
    mostrar_warning_periodo = False
    
    if st.session_state.df_completo is not None:
        if len(st.session_state.atletas_selecionados) > 0 and len(st.session_state.periodos_selecionados) > 0:
            botao_desabilitado = False
        else:
            if not st.session_state.atletas_selecionados:
                mostrar_warning_atleta = True
            if not st.session_state.periodos_selecionados:
                mostrar_warning_periodo = True
    
    if mostrar_warning_atleta:
        st.warning("⚠️ Selecione pelo menos um atleta")
    if mostrar_warning_periodo:
        st.warning("⚠️ Selecione pelo menos um período")
    
    # ÚNICA DEFINIÇÃO DO BOTÃO
    processar = st.button(
        "Processar", 
        type="primary", 
        use_container_width=True,
        disabled=botao_desabilitado
    )

# --- ÁREA PRINCIPAL ---
if processar and st.session_state.df_completo is not None:
    df_completo = st.session_state.df_completo
    atletas_selecionados = st.session_state.atletas_selecionados
    periodos_selecionados = st.session_state.periodos_selecionados
    
    # Aplicar filtros de atleta E período
    df_filtrado = df_completo[
        df_completo['Nome'].isin(atletas_selecionados) & 
        df_completo['Periodo'].isin(periodos_selecionados)
    ].reset_index(drop=True)
    
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os atletas e períodos selecionados")
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
            ax_hist.set_title(f"Histograma - {len(atletas_selecionados)} atleta(s), {len(periodos_selecionados)} período(s)")
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
            ax_qq.set_title(f"QQ Plot - {len(atletas_selecionados)} atleta(s), {len(periodos_selecionados)} período(s)")
            st.pyplot(fig_qq)
            plt.close(fig_qq)
        
        # --- TABELA DE FREQUÊNCIA ---
        st.subheader("📊 Tabela de Frequência - Intervalos Fixos")
        
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
        
        # --- TABELA RESUMO POR ATLETA E PERÍODO ---
        st.subheader("🏃 Resumo por Atleta e Período")
        
        resumo_atletas = []
        
        for nome in atletas_selecionados:
            for periodo in periodos_selecionados:
                dados_filtro = df_filtrado[
                    (df_filtrado['Nome'] == nome) & 
                    (df_filtrado['Periodo'] == periodo)
                ]
                
                if not dados_filtro.empty:
                    idx_max = dados_filtro['Valor'].idxmax()
                    valor_max = dados_filtro.loc[idx_max, 'Valor']
                    minuto_max = dados_filtro.loc[idx_max, 'Minuto']
                    
                    idx_min = dados_filtro['Valor'].idxmin()
                    valor_min = dados_filtro.loc[idx_min, 'Valor']
                    minuto_min = dados_filtro.loc[idx_min, 'Minuto']
                    
                    amplitude = valor_max - valor_min
                    
                    resumo_atletas.append({
                        'Atleta': nome,
                        'Período': periodo,
                        'Valor Máximo': valor_max,
                        'Minuto do Máximo': minuto_max,
                        'Valor Mínimo': valor_min,
                        'Minuto do Mínimo': minuto_min,
                        'Amplitude (Máx - Mín)': amplitude,
                        'Nº Amostras': len(dados_filtro)
                    })
        
        if resumo_atletas:
            df_resumo = pd.DataFrame(resumo_atletas)
            df_resumo = df_resumo.sort_values(['Atleta', 'Período']).reset_index(drop=True)
            
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
        st.write(f"**Atletas:** {len(atletas_selecionados)} | **Períodos:** {len(periodos_selecionados)}")
        
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
        
        # --- GRÁFICO DE LINHA DO TEMPO ---
        st.subheader("⏱️ Evolução Temporal dos Valores")
        
        # Ordenar por minuto
        df_tempo = df_filtrado.copy()
        df_tempo = df_tempo.sort_values('Minuto').reset_index(drop=True)
        
        # Calcular média e limiar de 80%
        media_valor = df_tempo['Valor'].mean()
        max_valor = df_tempo['Valor'].max() if not df_tempo['Valor'].empty else 0
        limiar_80 = max_valor * 0.8
        limiar_70 = max_valor * 0.7
        
        def get_color(valor):
            if valor > limiar_80:
                return 'orange'
            elif valor > limiar_70:
                return 'gold'
            else:
                return 'steelblue'
        
        cores = [get_color(valor) for valor in df_tempo['Valor']] if not df_tempo.empty else []
        
        fig_tempo, ax_tempo = plt.subplots(figsize=(12, 6))
        
        if not df_tempo.empty:
            ax_tempo.bar(
                range(len(df_tempo)),
                df_tempo['Valor'],
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
                linewidth=1.5,
                alpha=0.7,
                label=f'80% do Máx: {limiar_80:.2f}'
            )
            
            ax_tempo.axhline(
                y=limiar_70,
                color='gold',
                linestyle=':',
                linewidth=1.5,
                alpha=0.7,
                label=f'70% do Máx: {limiar_70:.2f}'
            )
            
            ax_tempo.set_title(f"Evolução Temporal - {len(atletas_selecionados)} atleta(s), {len(periodos_selecionados)} período(s)", fontsize=14)
            ax_tempo.set_xlabel("Minuto", fontsize=12)
            ax_tempo.set_ylabel("Valor", fontsize=12)
            
            ax_tempo.legend(loc='upper right')
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
            "🟠 Linha pontilhada laranja: 80% do máximo | "
            "🟡 Linha pontilhada amarela: 70% do máximo"
        )

# --- MENSAGENS INICIAIS ---
elif st.session_state.df_completo is not None:
    if not st.session_state.atletas_selecionados or not st.session_state.periodos_selecionados:
        st.info("👈 Selecione pelo menos um atleta e um período, depois clique em 'Processar'")
    else:
        st.info("👈 Clique em 'Processar' para gerar as análises")
else:
    st.info("👈 Faça upload de um arquivo CSV para começar")