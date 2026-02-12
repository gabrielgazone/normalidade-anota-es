import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Teste de Normalidade dos Dados", layout="wide")
st.title("📊 Teste de Normalidade dos Dados - Múltiplas Variáveis")

# ---------------- SESSION STATE ----------------
for key, default in {
    "df_completo": None,
    "variaveis_quantitativas": [],
    "variavel_selecionada": None,
    "atletas_selecionados": [],
    "periodos_selecionados": [],
    "todos_periodos": [],
    "ordem_personalizada": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- FUNÇÕES ----------------
def interpretar_teste(p_valor, nome_teste):
    st.write(f"**Teste utilizado:** {nome_teste}")
    st.write(f"**Valor de p:** {p_valor:.5f}" if p_valor >= 0.0001 else f"**Valor de p:** {p_valor:.2e}")
    if p_valor > 0.05:
        st.success("✅ Não há evidências para rejeitar normalidade")
    else:
        st.warning("⚠️ Evidências para rejeitar normalidade")

def extrair_periodo(texto):
    try:
        texto = str(texto)
        primeiro_hifen = texto.find('-')
        if primeiro_hifen == -1 or len(texto) < 13:
            return ""
        return texto[primeiro_hifen + 1:-13].strip()
    except:
        return ""

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Upload dos Dados")
    upload_file = st.file_uploader("Escolha o arquivo CSV:", type=["csv"])

    if upload_file:
        data = pd.read_csv(upload_file)

        primeira_coluna = data.iloc[:, 0].astype(str)
        nomes = primeira_coluna.str.split("-").str[0].str.strip()
        minutos = primeira_coluna.str[-13:].str.strip()
        periodos = primeira_coluna.apply(extrair_periodo)

        variaveis = []
        for col in data.columns[1:]:
            if pd.to_numeric(data[col], errors="coerce").dropna().empty is False:
                variaveis.append(col)

        df = pd.DataFrame({
            "Nome": nomes,
            "Período": periodos,
            "Minuto": minutos
        })

        for col in variaveis:
            df[col] = pd.to_numeric(data[col], errors="coerce")

        st.session_state.df_completo = df
        st.session_state.variaveis_quantitativas = variaveis
        st.session_state.todos_periodos = sorted(df["Período"].unique())
        st.session_state.periodos_selecionados = st.session_state.todos_periodos.copy()
        st.session_state.atletas_selecionados = sorted(df["Nome"].unique())
        st.session_state.ordem_personalizada = st.session_state.todos_periodos.copy()

    if st.session_state.df_completo is not None:

        st.markdown("---")
        st.header("📈 Variável")

        st.session_state.variavel_selecionada = st.selectbox(
            "Escolha a variável:",
            st.session_state.variaveis_quantitativas
        )

        st.markdown("---")
        st.header("📅 Períodos")

        st.session_state.periodos_selecionados = st.multiselect(
            "Selecione:",
            st.session_state.todos_periodos,
            default=st.session_state.periodos_selecionados
        )

        st.markdown("---")
        st.header("🔍 Atletas")

        atletas_disponiveis = sorted(
            st.session_state.df_completo[
                st.session_state.df_completo["Período"].isin(st.session_state.periodos_selecionados)
            ]["Nome"].unique()
        )

        st.session_state.atletas_selecionados = st.multiselect(
            "Selecione:",
            atletas_disponiveis,
            default=atletas_disponiveis
        )

        st.markdown("---")
        st.header("🔄 Ordenação do Gráfico")

        ordem_opcao = st.radio(
            "Ordem do eixo X:",
            ["⏫ Minuto (Crescente)",
             "⏬ Minuto (Decrescente)",
             "📋 Período (A-Z)",
             "📋 Período (Z-A)",
             "🎯 Ordem Personalizada"]
        )

        if ordem_opcao == "🎯 Ordem Personalizada":
            st.session_state.ordem_personalizada = st.multiselect(
                "Defina a ordem:",
                st.session_state.periodos_selecionados,
                default=st.session_state.ordem_personalizada
            )

        process_button = st.button("🔄 Processar Análise", use_container_width=True)

# ---------------- ÁREA PRINCIPAL ----------------
if process_button and st.session_state.df_completo is not None:

    df = st.session_state.df_completo

    df_filtrado = df[
        (df["Nome"].isin(st.session_state.atletas_selecionados)) &
        (df["Período"].isin(st.session_state.periodos_selecionados))
    ].dropna(subset=[st.session_state.variavel_selecionada])

    if df_filtrado.empty:
        st.warning("⚠️ Nenhum dado encontrado.")
        st.stop()

    variavel = st.session_state.variavel_selecionada

    # ---------------- ORDENAR CORRETAMENTE ----------------
    df_tempo = df_filtrado.copy()

    # Extrair primeiro tempo do intervalo (ex: 00:00-01:00 → 00:00)
    df_tempo["Minuto_ord"] = (
        df_tempo["Minuto"]
        .str.extract(r"(\d+:\d+)")[0]
    )

    df_tempo["Minuto_ord"] = pd.to_timedelta(df_tempo["Minuto_ord"], errors="coerce")

    if ordem_opcao == "⏫ Minuto (Crescente)":
        df_tempo = df_tempo.sort_values("Minuto_ord")

    elif ordem_opcao == "⏬ Minuto (Decrescente)":
        df_tempo = df_tempo.sort_values("Minuto_ord", ascending=False)

    elif ordem_opcao == "📋 Período (A-Z)":
        df_tempo = df_tempo.sort_values(["Período", "Minuto_ord"])

    elif ordem_opcao == "📋 Período (Z-A)":
        df_tempo = df_tempo.sort_values(["Período", "Minuto_ord"], ascending=[False, True])

    elif ordem_opcao == "🎯 Ordem Personalizada":
        ordem_map = {p: i for i, p in enumerate(st.session_state.ordem_personalizada)}
        df_tempo["ordem_temp"] = df_tempo["Período"].map(ordem_map)
        df_tempo = df_tempo.sort_values(["ordem_temp", "Minuto_ord"])
        df_tempo = df_tempo.drop(columns=["ordem_temp"])

    df_tempo = df_tempo.reset_index(drop=True)

    # ---------------- GRÁFICO ----------------
    st.subheader("⏱️ Evolução Temporal")

    media = df_tempo[variavel].mean()
    limiar = df_tempo[variavel].max() * 0.8

    fig, ax = plt.subplots(figsize=(14, 6))

    cores = ["red" if v > limiar else "steelblue" for v in df_tempo[variavel]]

    ax.bar(range(len(df_tempo)), df_tempo[variavel], color=cores)
    ax.axhline(media, linestyle="--", color="black", label="Média")
    ax.axhline(limiar, linestyle=":", color="orange", label="80% Máx")

    ax.set_xticks(range(len(df_tempo)))
    ax.set_xticklabels(df_tempo["Minuto"], rotation=45, ha="right")

    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- TESTE NORMALIDADE ----------------
    st.subheader("🧪 Teste de Normalidade")

    dados = df_filtrado[variavel]

    if len(dados) < 3:
        st.error("Amostra muito pequena.")
    else:
        stat, p = stats.shapiro(dados)
        interpretar_teste(p, "Shapiro-Wilk")
