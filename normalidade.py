import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide")
st.title("Teste de Normalidade dos Dados")

with st.sidebar:
    upload_file = st.file_uploader("Escolha o arquivo:", type=['csv'],
                                   accept_multiple_files=False)
    
    # Adicionar controle para número de classes
    n_classes = st.slider("Número de classes (faixas):", min_value=3, max_value=20, value=5)
    process_button = st.button("Processar")

if process_button and upload_file is not None: 
    try:
        data = pd.read_csv(upload_file)
        
        if data.empty or data.iloc[:,0].isnull().all():
            st.error("O arquivo está vazio ou a primeira coluna não tem dados válidos")
        else:
            # Pegar dados da primeira coluna
            dados_teste = data.iloc[:,0].dropna()
            n_amostras = len(dados_teste)
            
            # Gráficos
            col1, col2 = st.columns(2)
            with col1:
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(dados_teste, bins=n_classes,  # Usar o número de classes selecionado
                            color='blue', alpha=0.7, rwidth=0.85)
                ax_hist.set_title("Histograma")
                ax_hist.set_xlabel("Valores")
                ax_hist.set_ylabel("Frequência")
                st.pyplot(fig_hist)
            with col2:
                fig_qq, ax_qq = plt.subplots()
                stats.probplot(dados_teste, dist='norm', plot=ax_qq)
                ax_qq.set_title("QQ Plot")
                st.pyplot(fig_qq)
            
            # --- TABELA DE FREQUÊNCIA ---
            st.subheader("📊 Tabela de Frequência")
            
            # Calcular estatísticas para as classes
            minimo = dados_teste.min()
            maximo = dados_teste.max()
            amplitude = maximo - minimo
            largura_classe = amplitude / n_classes
            
            # Criar os limites das classes
            limites = [minimo + i * largura_classe for i in range(n_classes + 1)]
            
            # Criar rótulos para as classes
            rotulos = []
            for i in range(n_classes):
                inicio = limites[i]
                fim = limites[i + 1]
                rotulos.append(f"[{inicio:.2f} - {fim:.2f})")
            
            # Categorizar os dados
            categorias = pd.cut(dados_teste, bins=limites, labels=rotulos, include_lowest=True, right=False)
            
            # Criar tabela de frequência
            freq_table = pd.DataFrame({
                'Faixa de Valores': rotulos,
                'Frequência': [0] * n_classes
            })
            
            # Preencher frequências
            contagens = categorias.value_counts().sort_index()
            for i, rotulo in enumerate(rotulos):
                if rotulo in contagens.index:
                    freq_table.loc[i, 'Frequência'] = contagens[rotulo]
            
            # Adicionar colunas de percentual
            freq_table['Percentual (%)'] = (freq_table['Frequência'] / n_amostras * 100).round(2)
            freq_table['Frequência Acumulada'] = freq_table['Frequência'].cumsum()
            freq_table['Percentual Acumulado (%)'] = freq_table['Percentual (%)'].cumsum()
            
            # Exibir tabela
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
            
            # Estatísticas descritivas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mínimo", f"{minimo:.2f}")
            with col2:
                st.metric("Máximo", f"{maximo:.2f}")
            with col3:
                st.metric("Amplitude", f"{amplitude:.2f}")
            with col4:
                st.metric("Nº de Classes", n_classes)
            
            # --- TESTE DE NORMALIDADE ---
            st.subheader("🧪 Resultado do Teste de Normalidade")
            st.write(f"**Tamanho da amostra:** {n_amostras}")
            
            if n_amostras < 3:
                st.error("Amostra muito pequena (n < 3). Teste não aplicável.")
            elif n_amostras > 5000:
                st.warning("Amostra grande demais para Shapiro-Wilk. Usando teste D'Agostino-Pearson.")
                k2, p_value = stats.normaltest(dados_teste)
                if p_value < 0.0001:
                    st.write(f"**Valor de p:** {p_value:.2e} (notação científica)")
                else:
                    st.write(f"**Valor de p:** {p_value:.5f}")
                if p_value > 0.05:
                    st.success("Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
                else:
                    st.warning("Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
            else:
                shapiro_test = stats.shapiro(dados_teste)
                if shapiro_test.pvalue < 0.0001:
                    st.write(f"**Valor de p:** {shapiro_test.pvalue:.2e} (notação científica)")
                else:
                    st.write(f"**Valor de p:** {shapiro_test.pvalue:.5f}")
                if shapiro_test.pvalue > 0.05:
                    st.success("Não existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
                else:
                    st.warning("Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados")
                    
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    if not process_button:
        st.info("👈 Faça upload de um arquivo CSV e clique em 'Processar' para começar")
    elif upload_file is None:
        st.warning("Por favor, faça upload de um arquivo CSV primeiro.")