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
if 'ordem_personalizada' not in st.session_state:
    st.session_state.ordem_personalizada = []

# --- FUNÇÕES AUXILIARES ---
def interpretar_teste(p_valor, nome_teste):
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

def plotar_grafico_temporal(df, variavel, ordem):
    """Função separada para plotar o gráfico temporal"""
    df_tempo = df.copy()
    
    if ordem == "⏫ Minuto (Crescente)":
        df_tempo = df_tempo.sort_values('Minuto')
    elif ordem == "⏬ Minuto (Decrescente)":
        df_tempo = df_tempo.sort_values('Minuto', ascending=False)
    elif ordem == "📋 Período (A-Z)":
        df_tempo = df_tempo.sort_values(['Período', 'Minuto'])
    elif ordem == "📋 Período (Z-A)":
        df_tempo = df_tempo.sort_values(['Período', 'Minuto'], ascending=[False, True])
    elif ordem == "🎯 Ordem Personalizada":
        if st.session_state.ordem_personalizada:
            ordem_map = {p: i for i, p in enumerate(st.session_state.ordem_personalizada)}
            df_tempo['ordem_temp'] = df_tempo['Período'].map(ordem_map)
            df_tempo = df_tempo.sort_values(['ordem_temp', 'Minuto'])
            df_tempo = df_tempo.drop('ordem_temp', axis=1)
    
    df_tempo = df_tempo.reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    media_valor = df_tempo[variavel].mean()
    limiar_80 = df_tempo[variavel].max() * 0.8
    cores = ['red' if v > limiar_80 else 'steelblue' for v in df_tempo[variavel]]
    
    ax.bar(range(len(df_tempo)), df_tempo[variavel], color=cores, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(df_tempo)))
    ax.set_xticklabels(df_tempo['Minuto'], rotation=45, ha='right', fontsize=8)
    ax.axhline(y=media_valor, color='black', linestyle='--', linewidth=1.5, label=f'Média: {media_valor:.2f}')
    ax.axhline(y=limiar_80, color='orange', linestyle=':', linewidth=1, alpha=0.5, label=f'80% do Máx: {limiar_80:.2f}')
    ax.set_title(f"Evolução Temporal - {variavel}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Minuto", fontsize=12)
    ax.set_ylabel(variavel, fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Upload dos Dados")
    upload_file = st.file_uploader(
        "Escolha o arquivo CSV:", 
        type=['csv'],
        accept_multiple_files=False,
        help="Formato: Primeira coluna = Identificação (Nome-Período-Minuto)"
    )
    
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
                else:
                    st.error("❌ Nenhuma variável numérica válida encontrada")
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
    
    # --- FILTRO POR PERÍODO ---
    if st.session_state.df_completo is not None and st.session_state.todos_periodos:
        st.markdown("---")
        st.header("📅 Filtro por Período")
        
        lista_periodos = st.session_state.todos_periodos
        
        selecionar_todos_periodos = st.checkbox(
            "Selecionar todos os períodos",
            value=len(st.session_state.periodos_selecionados) == len(lista_periodos) if lista_periodos else True,
            key="selecionar_todos_periodos"
        )
        
        if selecionar_todos_periodos:
            st.session_state.periodos_selecionados = lista_periodos.copy()
            st.session_state.ordem_personalizada = lista_periodos.copy()
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
            else:
                st.session_state.periodos_selecionados = []
    
    # --- FILTRO POR ATLETA ---
    if st.session_state.df_completo is not None:
        st.markdown("---")
        st.header("🔍 Filtro por Atleta")
        
        df_temp_atletas = st.session_state.df_completo.copy()
        if st.session_state.periodos_selecionados:
            df_temp_atletas = df_temp_atletas[df_temp_atletas['Período'].isin(st.session_state.periodos_selecionados)]
        
        lista_atletas = sorted(df_temp_atletas['Nome'].unique())
        
        if lista_atletas:
            st.session_state.atletas_selecionados = lista_atletas.copy()
        
        st.info(f"✅ {len(lista_atletas)} atletas disponíveis")
    
    # --- CONFIGURAÇÕES DO GRÁFICO ---
    st.markdown("---")
    st.header("⚙️ Configurações")
    
    n_classes = st.slider("Número de classes no histograma:", 3, 20, 5)
    
    # ============= NOVO SISTEMA DE ORDENAÇÃO COM BOTÕES DE ARRASTAR =============
    st.markdown("---")
    st.header("🎯 Ordem Personalizada")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬆️ Mover para Cima", use_container_width=True):
            st.session_state.mover_direcao = "cima"
    with col2:
        if st.button("⬇️ Mover para Baixo", use_container_width=True):
            st.session_state.mover_direcao = "baixo"
    
    if st.session_state.periodos_selecionados:
        # Mostrar lista de períodos com seleção
        st.markdown("**Selecione um período para mover:**")
        
        # Criar selectbox com os períodos
        periodo_selecionado = st.selectbox(
            "Período",
            options=st.session_state.ordem_personalizada,
            key="periodo_para_mover"
        )
        
        # Aplicar movimento
        if 'mover_direcao' in st.session_state:
            if st.session_state.mover_direcao == "cima":
                idx = st.session_state.ordem_personalizada.index(periodo_selecionado)
                if idx > 0:
                    st.session_state.ordem_personalizada[idx], st.session_state.ordem_personalizada[idx-1] = \
                    st.session_state.ordem_personalizada[idx-1], st.session_state.ordem_personalizada[idx]
                st.session_state.mover_direcao = None
                st.rerun()
            elif st.session_state.mover_direcao == "baixo":
                idx = st.session_state.ordem_personalizada.index(periodo_selecionado)
                if idx < len(st.session_state.ordem_personalizada) - 1:
                    st.session_state.ordem_personalizada[idx], st.session_state.ordem_personalizada[idx+1] = \
                    st.session_state.ordem_personalizada[idx+1], st.session_state.ordem_personalizada[idx]
                st.session_state.mover_direcao = None
                st.rerun()
        
        # Mostrar ordem atual
        st.markdown("**Ordem atual:**")
        for i, p in enumerate(st.session_state.ordem_personalizada):
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{i+1}. {p}")
    
    # ============= BOTÃO PROCESSAR =============
    st.markdown("---")
    
    pode_processar = st.session_state.df_completo is not None and \
                    st.session_state.periodos_selecionados and \
                    st.session_state.variavel_selecionada
    
    process_button = st.button(
        "🔄 Processar Análise", 
        type="primary", 
        use_container_width=True,
        disabled=not pode_processar
    )

# --- ÁREA PRINCIPAL ---
if process_button and st.session_state.df_completo is not None and st.session_state.periodos_selecionados and st.session_state.variavel_selecionada:
    
    df_completo = st.session_state.df_completo
    atletas_selecionados = st.session_state.atletas_selecionados
    periodos_selecionados = st.session_state.periodos_selecionados
    variavel_analise = st.session_state.variavel_selecionada
    
    df_filtrado = df_completo[
        df_completo['Nome'].isin(atletas_selecionados) & 
        df_completo['Período'].isin(periodos_selecionados)
    ].copy()
    
    df_filtrado = df_filtrado.dropna(subset=[variavel_analise])
    
    if not df_filtrado.empty:
        st.header(f"📊 Análise de Normalidade: **{variavel_analise}**")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Períodos", f"{len(periodos_selecionados)}")
        with col_f2:
            st.metric("Atletas", f"{len(atletas_selecionados)}")
        with col_f3:
            st.metric("Observações", f"{len(df_filtrado)}")
        
        # GRÁFICOS
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            ax_hist.hist(df_filtrado[variavel_analise], bins=n_classes, color='steelblue', alpha=0.7, 
                        rwidth=0.85, edgecolor='black', linewidth=0.5)
            ax_hist.set_title(f"Histograma - {variavel_analise}", fontsize=14, fontweight='bold')
            ax_hist.set_xlabel(variavel_analise, fontsize=12)
            ax_hist.set_ylabel("Frequência", fontsize=12)
            ax_hist.grid(axis='y', alpha=0.3, linestyle='--')
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        
        with col2:
            fig_qq, ax_qq = plt.subplots(figsize=(8, 5))
            stats.probplot(df_filtrado[variavel_analise], dist='norm', plot=ax_qq)
            ax_qq.set_title(f"QQ Plot - {variavel_analise}", fontsize=14, fontweight='bold')
            ax_qq.set_xlabel("Quantis Teóricos", fontsize=12)
            ax_qq.set_ylabel("Quantis Observados", fontsize=12)
            ax_qq.grid(alpha=0.3, linestyle='--')
            st.pyplot(fig_qq)
            plt.close(fig_qq)
        
        # GRÁFICO TEMPORAL COM ORDEM PERSONALIZADA
        st.subheader("⏱️ Evolução Temporal dos Valores")
        
        # Botões para alternar entre modos de ordenação
        col_o1, col_o2, col_o3, col_o4, col_o5 = st.columns(5)
        
        with col_o1:
            if st.button("⏫ Minuto ↑", use_container_width=True):
                st.session_state.ordem_temporal = "⏫ Minuto (Crescente)"
                st.rerun()
        with col_o2:
            if st.button("⏬ Minuto ↓", use_container_width=True):
                st.session_state.ordem_temporal = "⏬ Minuto (Decrescente)"
                st.rerun()
        with col_o3:
            if st.button("📋 A-Z", use_container_width=True):
                st.session_state.ordem_temporal = "📋 Período (A-Z)"
                st.rerun()
        with col_o4:
            if st.button("📋 Z-A", use_container_width=True):
                st.session_state.ordem_temporal = "📋 Período (Z-A)"
                st.rerun()
        with col_o5:
            if st.button("🎯 Personalizada", use_container_width=True):
                st.session_state.ordem_temporal = "🎯 Ordem Personalizada"
                st.rerun()
        
        # Definir ordem padrão se não existir
        if 'ordem_temporal' not in st.session_state:
            st.session_state.ordem_temporal = "⏫ Minuto (Crescente)"
        
        # Plotar gráfico
        fig_tempo = plotar_grafico_temporal(df_filtrado, variavel_analise, st.session_state.ordem_temporal)
        st.pyplot(fig_tempo)
        plt.close(fig_tempo)
        
        st.caption(f"**Ordenação atual:** {st.session_state.ordem_temporal}")
        
        # TESTE DE NORMALIDADE
        st.subheader("🧪 Resultado do Teste de Normalidade")
        
        dados_teste = df_filtrado[variavel_analise].dropna()
        n_amostra = len(dados_teste)
        
        st.write(f"**Tamanho da amostra:** {n_amostra}")
        
        if n_amostra < 3:
            st.error("❌ Amostra muito pequena (n < 3)")
        elif n_amostra > 5000:
            try:
                k2, p_value = stats.normaltest(dados_teste)
                interpretar_teste(p_value, "D'Agostino-Pearson")
            except:
                st.warning("⚠️ Erro no teste de normalidade")
        else:
            try:
                shapiro_test = stats.shapiro(dados_teste)
                interpretar_teste(shapiro_test.pvalue, "Shapiro-Wilk")
            except:
                st.warning("⚠️ Erro no teste Shapiro-Wilk")

elif not process_button:
    if st.session_state.df_completo is None:
        st.info("👈 **Passo 1:** Faça upload de um arquivo CSV para começar")
    else:
        st.info("👈 **Passo 2:** Selecione os filtros e clique em 'Processar Análise'")