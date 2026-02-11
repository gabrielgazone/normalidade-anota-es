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
        value=5
    )
    
    # Processar arquivo quando enviado
    if upload_file is not None:
        try:
            data = pd.read_csv(upload_file)
            
            if data.shape[1] >= 2 and not data.empty and not data.iloc[:, 1].isnull().all():
                # Processar dados
                dados_teste = data.iloc[:, 1].dropna()
                primeira_coluna = data.iloc[:, 0].astype(str)
                
                nomes = primeira_coluna.str.split('-').str[0].str.strip()
                minutos = primeira_coluna.str[-13:].str.strip()
                
                if len(nomes) == len(minutos) == len(dados_teste):
                    df_completo = pd.DataFrame({
                        'Nome': nomes.reset_index(drop=True),
                        'Minuto': minutos.reset_index(drop=True),
                        'Valor': dados_teste.reset_index(drop=True)
                    })
                    
                    df_completo = df_completo[df_completo['Nome'].str.len() > 0]
                    
                    if not df_completo.empty:
                        st.session_state.df_completo = df_completo
                        st.session_state.atletas_selecionados = sorted(df_completo['Nome'].unique())
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {str(e)}")
    
    # --- FILTRO POR ATLETA ---
    if st.session_state.df_completo is not None:
        st.markdown("---")
        st.subheader("🔍 Filtro por Atleta")
        
        lista_atletas = sorted(st.session_state.df_completo['Nome'].unique())

        selecionar_todos = st.checkbox(
            "Selecionar todos os atletas",
            value=True,
            key="selecionar_todos"
        )
        
        if selecionar_todos:
            st.session_state.atletas_selecionados = lista_atletas
            # Mostrar apenas um texto informativo quando "selecionar todos" está ativo
            st.info(f"✅ {len(lista_atletas)} atletas selecionados")
        else:
            atletas_sel = st.multiselect(
                "Selecione os atletas:",
                options=lista_atletas,
                default=st.session_state.atletas_selecionados if st.session_state.atletas_selecionados else [],
                key="multiselect_ativos"
            )
            if atletas_sel:
                st.session_state.atletas_selecionados = atletas_sel
            else:
                st.session_state.atletas_selecionados = []
                st.warning("Selecione pelo menos um atleta")
        
        if st.session_state.atletas_selecionados:
            st.caption(f"✅ {len(st.session_state.atletas_selecionados)} atletas selecionados")
    
    process_button = st.button("Processar", type="primary", use_container_width=True)
    if st.session_state.df_completo is not None and not st.session_state.atletas_selecionados:
        st.error("Selecione pelo menos um atleta antes de processar")
        process_button = False

# --- ÁREA PRINCIPAL ---
if process_button and st.session_state.df_completo is not None and st.session_state.atletas_selecionados:
    df_completo = st.session_state.df_completo
    atletas_selecionados = st.session_state.atletas_selecionados
    
    # Aplicar filtro
    df_filtrado = df_completo[df_completo['Nome'].isin(atletas_selecionados)]
    
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os atletas selecionados")
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
        
        # --- GRÁFICO DE LINHA DO TEMPO ---
        st.subheader("⏱️ Evolução Temporal dos Valores")
        
        # Ordenar por minuto
        df_tempo = df_filtrado.copy()
        df_tempo = df_tempo.sort_values('Minuto').reset_index(drop=True)
        
        # Calcular média e limiar de 80%
        media_valor = df_tempo['Valor'].mean()
        limiar_80 = df_tempo['Valor'].max() * 0.8
        
        # Criar gráfico
        fig_tempo, ax_tempo = plt.subplots(figsize=(12, 6))
        
        # Definir cores baseadas no limiar de 80%
        cores = ['red' if valor > limiar_80 else 'steelblue' for valor in df_tempo['Valor']]
        
        # Plotar barras
        bars = ax_tempo.bar(
            range(len(df_tempo)),
            df_tempo['Valor'],
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
        
        # Adicionar linha do limiar (opcional - para referência)
        ax_tempo.axhline(
            y=limiar_80,
            color='orange',
            linestyle=':',
            linewidth=1,
            alpha=0.5,
            label=f'80% do Máx: {limiar_80:.2f}'
        )
        
        # Títulos e labels
        ax_tempo.set_title(f"Evolução Temporal - {len(atletas_selecionados)} atleta(s)", fontsize=14)
        ax_tempo.set_xlabel("Minuto", fontsize=12)
        ax_tempo.set_ylabel("Valor", fontsize=12)
        
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

elif not process_button:
    if st.session_state.df_completo is None:
        st.info("👈 Faça upload de um arquivo CSV para começar")
    else:
        st.info("👈 Selecione os atletas e clique em 'Processar'")