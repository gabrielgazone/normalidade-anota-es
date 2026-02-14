import streamlit as st
import sys

st.set_page_config(page_title="TESTE", layout="wide")

st.write("### Teste de Importações")
st.write("✅ Streamlit OK")

try:
    import pandas as pd
    st.write("✅ pandas OK")
except Exception as e:
    st.write("❌ pandas ERRO:", str(e))

try:
    import numpy as np
    st.write("✅ numpy OK")
except Exception as e:
    st.write("❌ numpy ERRO:", str(e))

try:
    import plotly.express as px
    st.write("✅ plotly OK")
except Exception as e:
    st.write("❌ plotly ERRO:", str(e))

try:
    import scipy.stats as stats
    st.write("✅ scipy OK")
except Exception as e:
    st.write("❌ scipy ERRO:", str(e))

st.write("---")
st.write("✅ Todos os testes concluídos!")