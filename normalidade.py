import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide")
st.title("Teste de Normalidade dos Dados")

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
    
    process_button = st.button("Processar")

if process_button and upload_file is not None:
    try:
        data = pd.read_csv(upload_file)
        
        if data.shape[1] < 2:
            st.error("O arquivo precisa ter pelo menos 2 colunas")
        elif data.empty or data.iloc[:, 1].isnull().all():
            st.error("O arquivo está vazio ou a segunda coluna não tem dados válidos")
        else:
            # --- PROCESSAMENTO DOS DADOS ---
            dados_teste = data.iloc[:, 1].dropna()
            n_amostras = len(dados_teste)
            
            primeira_coluna = data.iloc[:, 0].astype(str)
            
            nomes = primeira_coluna.str.split('-').str[0].str.strip()
            minutos = primeira_coluna.str[-13:].str.strip()
            
            if len(nomes) != len(minutos) or len(nomes) != len(dados_teste):
                st.error("Erro na extração dos dados. Verifique o formato do arquivo.")
            else:
                df_completo = pd.DataFrame({
                    'Nome': nomes.reset_index(drop=True),
                    'Minuto': minutos.reset_index(drop=True),
                    'Valor': dados_teste.reset_index(drop=True)
                })
                
                df_completo = df_completo[df_completo['Nome'].str.len() > 0]
                
                if df_completo.empty:
                    st.error("Não foi possível extrair nomes válidos da primeira coluna")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.hist(
                            df_completo['Valor'], 
                            bins=n_classes,
                            color='blue', 
                            alpha=0.7, 
                            rwidth=0.85
                        )
                        ax_hist.set_title("Histograma")
                        ax_hist.set_xlabel("Valores")
                        ax_hist.set_ylabel("Frequência")
                        st.pyplot(fig_hist)
                        plt.close(fig_hist)
                    
                    with col2:
                        fig_qq, ax_qq = plt.subplots()
                        stats.probplot(
                            df_completo['Valor'], 
                            dist='norm', 
                            plot=ax_qq
                        )
                        ax_qq.set_title("QQ Plot")
                        st.pyplot(fig_qq)
                        plt.close(fig_qq)
                    
                    st.subheader("📊 Tabela de Frequência")
                    
                    minimo = df_completo['Valor'].min()
                    maximo = df_completo['Valor'].max()
                    amplitude_total = maximo - minimo
                    largura_classe = amplitude_total / n_classes if amplitude_total > 0 else 1
                    
                    limites = [minimo + i * largura_classe for i in range(n_classes + 1)]
                    
                    rotulos = []
                    for i in range(n_classes):
                        inicio = limites[i]
                        fim = limites[i + 1]
                        rotulos.append(f"[{inicio:.2f} - {fim:.2f})")
                    
                    categorias = pd.cut(
                        df_completo['Valor'], 
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
                        freq_table['Frequência'] / len(df_completo) * 100
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
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mínimo", f"{minimo:.2f}")
                    with col2:
                        st.metric("Máximo", f"{maximo:.2f}")
                    with col3:
                        st.metric("Amplitude", f"{amplitude_total:.2f}")
                    with col4:
                        st.metric("Nº de Classes", n_classes)
                    
                    st.subheader("🏃 Resumo por Atleta")
                    
                    resumo_atletas = []
                    
                    for nome in df_completo['Nome'].unique():
                        dados_atleta = df_completo[df_completo['Nome'] == nome]
                        
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
                                'Amplitude (Máx - Mín)': amplitude
                            })
                    
                    if resumo_atletas:
                        df_resumo = pd.DataFrame(resumo_atletas)
                        df_resumo = df_resumo.sort_values('Atleta').reset_index(drop=True)
                        
                        st.dataframe(
                            df_resumo.style.format({
                                'Valor Máximo': '{:.2f}',
                                'Valor Mínimo': '{:.2f}',
                                'Amplitude (Máx - Mín)': '{:.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.warning("Não foi possível gerar resumo dos atletas")
                    
                    st.subheader("🧪 Resultado do Teste de Normalidade")
                    st.write(f"**Tamanho da amostra:** {len(df_completo)}")
                    
                    if len(df_completo) < 3:
                        st.error("Amostra muito pequena (n < 3). Teste não aplicável.")
                    elif len(df_completo) > 5000:
                        st.warning("Amostra grande demais para Shapiro-Wilk. Usando teste D'Agostino-Pearson.")
                        k2, p_value = stats.normaltest(df_completo['Valor'])
                        if p_value < 0.0001:
                            st.write(f"**Valor de p:** {p_value:.2e} (notação científica)")
                        else:
                            st.write(f"**Valor de p:** {p_value:.5f}")
                        if p_value > 0.05:
                            st.success("Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
                        else:
                            st.warning("Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
                    else:
                        shapiro_test = stats.shapiro(df_completo['Valor'])
                        p_valor = shapiro_test.pvalue
                        if p_valor < 0.0001:
                            st.write(f"**Valor de p:** {p_valor:.2e} (notação científica)")
                        else:
                            st.write(f"**Valor de p:** {p_valor:.5f}")
                        if p_valor > 0.05:
                            st.success("Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
                        else:
                            st.warning("Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
                    
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
else:
    if not process_button and upload_file is None:
        st.info("👈 Faça upload de um arquivo CSV e clique em 'Processar' para começar")
    elif upload_file is None:
        st.warning("Por favor, faça upload de um arquivo CSV primeiro.")