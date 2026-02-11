<!-- Copilot / AI agent instructions for this repository -->
# Quick AI Agent Guide — normalidade-anota-es

Purpose: help AI coding agents become productive quickly with this small Streamlit app.

- **Quick start (dev)**
  - Create a venv and install deps: `pip install -r requirements.txt`
  - Run the app: `streamlit run normalidade.py`

- **Primary files**
  - `normalidade.py` — single-file Streamlit app (main entrypoint).
  - `requirements.txt` — pinned dependencies (Streamlit, pandas, numpy, matplotlib, scipy).
  - `readme.md` — user README (outdated reference to `app.py`; use `normalidade.py`).

- **Data format & parsing details (critical)**
  - The app accepts a CSV (`.csv`) via the sidebar uploader.
  - It expects at least 2 columns. Column 0 contains a combined string with the athlete name and a timestamp/minute; column 1 contains numeric values. The code parses:
    - `nome = first_col.split('-')[0].strip()` — athlete name is everything before the first `-`.
    - `minuto = first_col[-13:].strip()` — minute is extracted as the last 13 characters of column 0.
  - If these assumptions are violated the upload will be ignored or produce errors; prefer samples following the existing CSV naming pattern.

- **UI & state patterns to preserve**
  - Uses `st.session_state` keys: `df_completo` and `atletas_selecionados`. Keep these keys stable when modifying UI flow.
  - The sidebar contains the CSV uploader, `n_classes` slider (3–20) and the `Processar` button. The button is disabled until a dataframe is loaded and at least one athlete is selected.
  - Selecting "Selecionar todos os atletas" writes to `st.session_state.atletas_selecionados` (assignment logic lives in the sidebar block).

- **Analysis and numeric behavior (important for correctness)**
  - Histogram bins follow the `n_classes` slider and are computed via `numpy` ranges and `pandas.cut` with `right=False`.
  - Normality testing:
    - If n < 3: test skipped (too small).
    - If n > 5000: uses `scipy.stats.normaltest` (D'Agostino-Pearson).
    - Otherwise: uses `scipy.stats.shapiro` (Shapiro–Wilk).
  - Outputs use `st.metric`, `st.dataframe` and `st.pyplot` (matplotlib figures). Keep these output primitives when changing visuals.

- **Conventions & patterns**
  - Single-file app: prefer in-file edits for small features. If splitting into modules, preserve session_state key names and import order.
  - Plots are created with matplotlib, then displayed via `st.pyplot(fig)` and immediately `plt.close(fig)`.
  - Frequency table uses `pandas.cut` and explicitly constructs labels; tests/logic depend on label ordering — keep that approach or replicate exact label generation if refactoring.

- **Debugging & developer workflow**
  - Typical debug run: `streamlit run normalidade.py` — observe console output for exceptions. Add `st.write()` or temporary `print()` for quick runtime inspection.
  - Repro steps should use small CSV samples that match the expected `Nome-Minuto` pattern to validate parsing.

- **Notes for edits / pull requests**
  - Update `readme.md` if you change the entrypoint name (`app.py` → `normalidade.py`).
  - Avoid renaming `st.session_state` keys without performing a search-and-replace across `normalidade.py` and any new modules.
  - If adding tests, provide a minimal CSV fixture demonstrating the expected first-column format.

If anything in this guide is unclear or you want more examples (sample CSV, small unit test, or refactor suggestions), tell me which part to expand.
