import streamlit as st
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import plotly.graph_objects as go

# ===================== RTPE CSV: wykres wg rodziny (M/Q/Y/W) z osią czasu z sufiksu =====================
st.markdown("<hr>", unsafe_allow_html=True)
st.header("RTPE — rodziny instrumentów (M/Q/Y/W) z osią czasu z kodu kontraktu")


# DEFAULT_RTPE_CSV = "/Users/michal/Downloads/RTPE_Contracts_abr_sample_CSV.csv"
DEFAULT_RTPE_CSV = (
    "https://raw.githubusercontent.com/michael-ostaszewski/ParsymoniaMVP_EnergyDataVisualizer/"
    "main/RTPE_Contracts_abr_sample_CSV.csv"
)

# DATA_PATH = Path(__file__).parent / "RTPE_Contracts_abr_sample_CSV.csv"
# df = pd.read_csv(DATA_PATH, sep=";", dtype=str, encoding="utf-8", engine="python")

@st.cache_data(show_spinner=False)
def _load_rtpe_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path, sep=";", dtype=str, encoding="utf-8", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    # Zapewnij kolumnę 'instrument' (3. kolumna z pliku)
    if "instrument" not in df.columns and len(df.columns) >= 3:
        df.rename(columns={df.columns[2]: "instrument"}, inplace=True)

    # Parsowanie daty obrotu (nie używana jako oś X, ale może się przydać)
    if "data obrotu" in df.columns:
        df["data obrotu"] = pd.to_datetime(df["data obrotu"], errors="coerce")

    # # Normalizacja liczb (polskie przecinki)
    # numeric_like = [
    #     "DKR","kurs min.","kurs maks.","najlepsza oferta kupna","najlepsza oferta sprzedaży",
    #     "wol. obrotu","liczba kontraktów","liczba transakcji","wartość obrotu",
    #     "liczba otwartych pozycji","liczba kontraktów od początku"
    # ]

    # --- DODAJ do listy numeric_like, żeby parsować nowe pole ---
    # (umieść to w miejscu, gdzie definiujesz numeric_like)
    numeric_like = [
        "DKR", "kurs min.", "kurs maks.", "najlepsza oferta kupna", "najlepsza oferta sprzedaży",
        "wol. obrotu", "liczba kontraktów", "liczba transakcji", "wartość obrotu",
        "liczba otwartych pozycji", "liczba kontraktów od początku", "LOP [MWh]"
    ]

    for col in numeric_like:
        if col in df.columns:
            s = (
                df[col].astype(str)
                      .str.replace("\u00A0", "", regex=False)  # twarde spacje
                      .str.replace(" ", "", regex=False)
                      .str.replace(",", ".", regex=False)
                      .str.replace(r"[^0-9\.\-]", "", regex=True)
            )
            df[col] = pd.to_numeric(s, errors="coerce")
    return df

def _yy_to_year(yy: int) -> int:
    # Uwzględniamy lata 2000–2099; w danych masz 21–25 itd.
    return 2000 + yy

def _first_day_of_quarter(year: int, q: int) -> datetime:
    month = (int(q) - 1) * 3 + 1
    return datetime(year, month, 1)

def _parse_instrument_to_period(instr: str):
    """
    Zwraca dict:
      family: np. 'BASE_M', 'BASE_Q', 'BASE_Y', 'PEAK5_Y', 'BASE_W'
      tenor:  'M'/'Q'/'Y'/'W'
      period_dt: datetime (początek miesiąca/kwartału/roku lub poniedziałek ISO dla tygodnia)
      period_label: etykieta do tooltips/osi
    Jeśli nie parsuje — zwraca None.
    """
    if not isinstance(instr, str):
        return None

    # Wzorzec: PREFIKS_(M|Q|Y|W)-a(-b opcjonalnie)
    m = re.match(r'^([A-Za-z0-9]+)_(M|Q|Y|W)-(\d{1,2})(?:-(\d{2}))?$', instr)
    if not m:
        return None
    prefix, tenor, a, b = m.groups()
    family = f"{prefix}_{tenor}"

    try:
        if tenor == "M":
            # M-<mm>-<yy>
            month = int(a)
            year = _yy_to_year(int(b))
            period_dt = datetime(year, month, 1)
            period_label = f"{year}-{month:02d}"
        elif tenor == "Q":
            # Q-<q>-<yy>
            q = int(a)
            year = _yy_to_year(int(b))
            period_dt = _first_day_of_quarter(year, q)
            period_label = f"Q{q} {year}"
        elif tenor == "Y":
            # Y-<yy>
            year = _yy_to_year(int(a))
            period_dt = datetime(year, 1, 1)
            period_label = f"{year}"
        elif tenor == "W":
            # W-<week>-<yy>  → poniedziałek ISO tego tygodnia
            week = int(a)
            year = _yy_to_year(int(b))
            # próba przez ISO format (G/V/u)
            try:
                period_dt = pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u")
            except Exception:
                # fallback: od 4 stycznia znajdź właściwy poniedziałek
                d = datetime(year, 1, 4)
                period_dt = d + pd.to_timedelta((week-1)*7 - (d.isoweekday()-1), unit="D")
            period_label = f"{year}-W{week:02d}"
        else:
            return None
    except Exception:
        return None

    return {"family": family, "tenor": tenor, "period_dt": period_dt, "period_label": period_label}

@st.cache_data(show_spinner=False)
def _with_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "instrument" not in df.columns:
        return df.copy()
    out = df.copy()
    parsed = out["instrument"].apply(_parse_instrument_to_period)
    out["__family__"]      = parsed.apply(lambda x: x["family"] if x else None)
    out["__tenor__"]       = parsed.apply(lambda x: x["tenor"] if x else None)
    out["__period_dt__"]   = parsed.apply(lambda x: x["period_dt"] if x else None)
    out["__period_lbl__"]  = parsed.apply(lambda x: x["period_label"] if x else None)
    return out

# ── UI: plik ────────────────────────────────────────────────────────────
st.caption("Domyślnie użyjemy: `/Users/michal/Downloads/RTPE_Contracts_abr_sample_CSV.csv`. Możesz też wgrać inny plik CSV.")
upl_csv = st.file_uploader("Wgraj plik CSV (opcjonalnie)", type=["csv"], key="rtpe_csv_uploader_family")

file_source = upl_csv if upl_csv is not None else DEFAULT_RTPE_CSV
if file_source is None:
    st.warning("Nie znaleziono domyślnego pliku i nic nie wgrano.")
else:
    df_raw = _load_rtpe_csv(file_source)
    if df_raw.empty:
        st.info("Plik wczytany, ale nie zawiera danych.")
    else:
        dfp = _with_period_columns(df_raw)
        # odfiltruj tylko poprawnie sparsowane symbole
        dfp = dfp.dropna(subset=["__family__", "__period_dt__"])

        if dfp.empty:
            st.info("Brak kontraktów w formacie *_M-*, *_Q-*, *_Y-* lub *_W-* do narysowania.")
        else:

            # ---------- POMOCNICZE: wyznaczanie okresów z data obrotu ----------
            def _week_monday(ts: pd.Timestamp) -> pd.Timestamp:
                # Poniedziałek tygodnia ISO
                # (poprawka: użyj wartości liczbowej dnia tygodnia)
                return ts - pd.to_timedelta(ts.dayofweek, unit="D")


            def _quarter_start(ts: pd.Timestamp) -> pd.Timestamp:
                first_month = (int((ts.month - 1) / 3) * 3) + 1
                return pd.Timestamp(year=ts.year, month=first_month, day=1)


            def _year_start(ts: pd.Timestamp) -> pd.Timestamp:
                return pd.Timestamp(year=ts.year, month=1, day=1)


            # ---------- UI: ustawienia wykresu + nowa oś X ----------
            families = sorted(dfp["__family__"].dropna().unique().tolist())
            default_fams = [f for f in families if f.endswith("_Y")] or families[:3]

            st.markdown("**Ustawienia wykresu**")
            colL, colR = st.columns([2, 1])
            with colL:
                selected_fams = st.multiselect(
                    "Wybierz rodziny instrumentów (linia = rodzina):",
                    options=families,
                    default=default_fams,
                    help="Np. BASE_M, BASE_Q, BASE_Y. Filtr dotyczy linii na wykresie."
                )
            with colR:
                metric_candidates = [
                    c for c in [
                        "DKR", "kurs min.", "kurs maks.", "najlepsza oferta kupna", "najlepsza oferta sprzedaży",
                        "wol. obrotu", "liczba kontraktów", "liczba transakcji", "wartość obrotu",
                        "liczba otwartych pozycji", "liczba kontraktów od początku", "LOP [MWh]"
                    ] if c in dfp.columns
                ]
                default_metric = "DKR" if "DKR" in metric_candidates else (
                    metric_candidates[0] if metric_candidates else None)
                y_metric = st.selectbox("Metryka (oś Y):", options=metric_candidates,
                                        index=metric_candidates.index(default_metric) if default_metric else 0)

            colA, colB, colC = st.columns(3)
            with colA:
                use_markers = st.checkbox("Punkty na wykresie", value=True)
            with colB:
                show_minmax = st.checkbox("Pokaż min/max (dla metryk cenowych)",
                                          value=("kurs min." in dfp.columns and "kurs maks." in dfp.columns))
            with colC:
                show_table = st.checkbox("Pokaż tabelę danych", value=False)

            # --- NOWE: wybór źródła osi X + agregacja ---
            colX1, colX2 = st.columns([2, 1])
            with colX1:
                x_axis_mode = st.radio(
                    "Źródło osi X:",
                    ["Okres z kodu kontraktu (tak jak było)", "Data obrotu (agregowana)"],
                    index=1,
                    help="Gdy wybierzesz 'Data obrotu (agregowana)', punkty zostaną zgrupowane po dacie wg wybranej granulacji."
                )
            with colX2:
                if x_axis_mode == "Data obrotu (agregowana)":
                    granularity = st.selectbox("Granulacja daty:", ["Dzień", "Tydzień (ISO)", "Kwartał", "Rok"],
                                               index=0)
                    agg_fn_name = st.selectbox("Agregacja:", ["średnia", "ostatnia", "suma", "min", "max"], index=0)
                else:
                    granularity = None
                    agg_fn_name = None

            # Dane do wykresu (po rodzinach)
            dplot = dfp[dfp["__family__"].isin(selected_fams)].copy()

            # --- wyznacz kolumnę x_dt zależnie od wyboru osi X ---
            if x_axis_mode == "Okres z kodu kontraktu (tak jak było)":
                # jak wcześniej – używamy __period_dt__ z sufiksu instrumentu
                dplot["x_dt"] = dplot["__period_dt__"]
                x_title = "Okres (z sufiksu instrumentu)"
            else:
                # opieramy się na data obrotu
                if "data obrotu" not in dplot.columns or dplot["data obrotu"].isna().all():
                    st.warning("Brak poprawnych wartości w kolumnie 'data obrotu' – nie mogę zagregować po dacie. "
                               "Przełączam na oś z kodu kontraktu.")
                    dplot["x_dt"] = dplot["__period_dt__"]
                    x_title = "Okres (z sufiksu instrumentu)"
                else:
                    # Upewnij się, że to Timestamp
                    dplot["data obrotu"] = pd.to_datetime(dplot["data obrotu"], errors="coerce")
                    if granularity == "Dzień":
                        dplot["x_dt"] = dplot["data obrotu"].dt.normalize()
                    elif granularity == "Tydzień (ISO)":
                        dplot["x_dt"] = dplot["data obrotu"].apply(lambda x: _week_monday(x) if pd.notna(x) else pd.NaT)
                    elif granularity == "Kwartał":
                        dplot["x_dt"] = dplot["data obrotu"].apply(
                            lambda x: _quarter_start(x) if pd.notna(x) else pd.NaT)
                    elif granularity == "Rok":
                        dplot["x_dt"] = dplot["data obrotu"].apply(lambda x: _year_start(x) if pd.notna(x) else pd.NaT)
                    else:
                        dplot["x_dt"] = dplot["data obrotu"].dt.normalize()
                    x_title = f"Data obrotu — agregacja: {granularity}"

            # wybierz kolumny i usuń NaN
            needed_cols = ["instrument", "__family__", "__tenor__", "x_dt", "__period_dt__", y_metric]
            if show_minmax and "kurs min." in dplot.columns and "kurs maks." in dplot.columns:
                needed_cols += ["kurs min.", "kurs maks."]
            dplot = dplot[[c for c in needed_cols if c in dplot.columns]].dropna(subset=["x_dt"])


            # --- jeśli oś X to data obrotu, agregujemy po (family, x_dt) ---
            def _agg_fn(series: pd.Series):
                if agg_fn_name == "średnia":
                    return series.mean(skipna=True)
                if agg_fn_name == "ostatnia":
                    # ostatnia nie-NaN po sortowaniu po x_dt
                    s = series.dropna()
                    return s.iloc[-1] if len(s) else float("nan")
                if agg_fn_name == "suma":
                    return series.sum(skipna=True)
                if agg_fn_name == "min":
                    return series.min(skipna=True)
                if agg_fn_name == "max":
                    return series.max(skipna=True)
                return series.mean(skipna=True)


            if x_axis_mode == "Data obrotu (agregowana)":
                # posortuj przed „ostatnią”
                dplot = dplot.sort_values(["__family__", "x_dt"])
                agg_dict = {y_metric: _agg_fn}
                if show_minmax and "kurs min." in dplot.columns and "kurs maks." in dplot.columns:
                    # dla min/max użyjemy odpowiednio min i max niezależnie od wyboru metryki
                    agg_dict["kurs min."] = "min"
                    agg_dict["kurs maks."] = "max"

                dplot = (
                    dplot.groupby(["__family__", "x_dt"], as_index=False)
                    .agg(agg_dict)
                )
            else:
                # bez agregacji – po prostu sortuj po x
                dplot = dplot.sort_values(["__family__", "x_dt"])

            # --- rysunek ---
            if dplot.empty or not selected_fams:
                st.info("Wybierz co najmniej jedną rodzinę oraz metrykę.")
            else:
                fig = go.Figure()
                for fam in sorted(dplot["__family__"].unique()):
                    sub = dplot[dplot["__family__"] == fam]
                    fig.add_trace(go.Scatter(
                        x=sub["x_dt"],
                        y=sub[y_metric],
                        mode="lines+markers" if use_markers else "lines",
                        name=fam,
                        # hovertemplate="Rodzina: " + fam + "<br>Okres/Daty: %{x|%Y-%m-%d}<br>" + f"{y_metric}: %{y}<extra></extra>",
                        hovertemplate="Rodzina: " + fam + "<br>Okres/Daty: %{x|%Y-%m-%d}<br>" + y_metric + ": %{y}<extra></extra>",

                    ))

                    # Opcjonalne min/max – działa dla obu trybów osi
                    if show_minmax and {"kurs min.", "kurs maks."}.issubset(sub.columns) and y_metric in ["DKR",
                                                                                                          "kurs min.",
                                                                                                          "kurs maks."]:
                        fig.add_trace(go.Scatter(
                            x=sub["x_dt"], y=sub["kurs maks."],
                            mode="lines", name=f"{fam} — kurs maks.", line=dict(dash="dot"),
                            hovertemplate="Max: %{y}<extra></extra>", showlegend=True
                        ))
                        fig.add_trace(go.Scatter(
                            x=sub["x_dt"], y=sub["kurs min."],
                            mode="lines", name=f"{fam} — kurs min.", line=dict(dash="dot"),
                            hovertemplate="Min: %{y}<extra></extra>", showlegend=True
                        ))

                fig.update_layout(
                    title=f"RTPE — {y_metric} wg rodzin (oś X: {'kod kontraktu' if x_axis_mode.startswith('Okres') else 'data obrotu'})",
                    xaxis_title=x_title,
                    yaxis_title=y_metric,
                    hovermode="x unified",
                    legend_title="Rodzina"
                )
                st.plotly_chart(fig, use_container_width=True)

                if show_table:
                    # Tabela wynikowa po agregacji (jeśli wybrano)
                    table_df = dplot.copy().rename(columns={"__family__": "Rodzina", "x_dt": "Oś X"})
                    st.dataframe(table_df.reset_index(drop=True), use_container_width=True)

            with st.expander("Informacje o danych / parsowaniu"):
                st.write("• Linie reprezentują **rodziny** (np. `BASE_M`, `BASE_Q`, `BASE_Y`).")
                st.write("• Oś X możesz przełączyć między:")
                st.write("  - **Okresem z sufiksu instrumentu** (`*_M-mm-yy`, `*_Q-q-yy`, `*_Y-yy`, `*_W-ww-yy`).")
                st.write(
                    "  - **Datą obrotu** z możliwością agregacji do: dnia, tygodnia ISO (poniedziałek), kwartału, roku.")
                st.write("• Przy agregacji wybierasz funkcję: średnia / ostatnia / suma / min / max.")

#             # ── UI: wybór rodzin (BASE_M, BASE_Q, BASE_Y, PEAK5_Y, ...) ──
#             families = sorted(dfp["__family__"].dropna().unique().tolist())
#             default_fams = [f for f in families if f.endswith("_Y")] or families[:3]
#
#             st.markdown("**Ustawienia wykresu**")
#             colL, colR = st.columns([2,1])
#             with colL:
#                 selected_fams = st.multiselect(
#                     "Wybierz rodziny instrumentów (linia = rodzina):",
#                     options=families,
#                     default=default_fams,
#                     help="Np. BASE_M, BASE_Q, BASE_Y. Oś X to okres wyliczony z sufiksu kontraktu."
#                 )
#             with colR:
#                 metric_candidates = [
#                     c for c in [
#                         "DKR","kurs min.","kurs maks.","najlepsza oferta kupna","najlepsza oferta sprzedaży",
#                         "wol. obrotu","liczba kontraktów","liczba transakcji","wartość obrotu",
#                         "liczba otwartych pozycji","liczba kontraktów od początku"
#                     ] if c in dfp.columns
#                 ]
#                 default_metric = "DKR" if "DKR" in metric_candidates else (metric_candidates[0] if metric_candidates else None)
#                 y_metric = st.selectbox("Metryka (oś Y):", options=metric_candidates, index=metric_candidates.index(default_metric) if default_metric else 0)
#
#             colA, colB, colC = st.columns(3)
#             with colA:
#                 use_markers = st.checkbox("Punkty na wykresie", value=True)
#             with colB:
#                 show_minmax = st.checkbox("Pokaż min/max (jeśli Y to cena)", value=("kurs min." in dfp.columns and "kurs maks." in dfp.columns))
#             with colC:
#                 show_table = st.checkbox("Pokaż tabelę danych", value=False)
#
#             # Dane do wykresu
#             dplot = dfp[dfp["__family__"].isin(selected_fams)].copy()
#             # wybierz potrzebne kolumny i usuń NaN w metryce lub czasie
#             needed_cols = ["instrument","__family__","__tenor__","__period_dt__","__period_lbl__", y_metric]
#             dplot = dplot[needed_cols].dropna(subset=["__period_dt__", y_metric])
#
#             if dplot.empty or not selected_fams:
#                 st.info("Wybierz co najmniej jedną rodzinę oraz metrykę.")
#             else:
#                 # posortuj po czasie
#                 dplot = dplot.sort_values("__period_dt__")
#
#                 # Rysunek: jedna linia na rodzinę
#                 fig = go.Figure()
#                 for fam in sorted(dplot["__family__"].unique()):
#                     sub = dplot[dplot["__family__"] == fam]
#                     fig.add_trace(go.Scatter(
#                         x=sub["__period_dt__"],
#                         y=sub[y_metric],
#                         mode="lines+markers" if use_markers else "lines",
#                         name=fam,
#                         hovertemplate=("Rodzina: %{text}<br>Okres: %{x|%Y-%m-%d}<br>"
#                                        + f"{y_metric}: "+"%{y}<extra></extra>"),
#                         text=[fam]*len(sub)
#                     ))
#
#                     # Opcjonalny zakres min/max (tylko gdy to ma sens)
#                     if show_minmax and ("kurs min." in dplot.columns) and ("kurs maks." in dplot.columns) and y_metric in ["DKR","kurs min.","kurs maks."]:
#                         sub2 = dfp[(dfp["__family__"] == fam)].dropna(subset=["__period_dt__"])
#                         sub2 = sub2.sort_values("__period_dt__")
#                         if "kurs min." in sub2.columns and "kurs maks." in sub2.columns:
#                             fig.add_trace(go.Scatter(
#                                 x=sub2["__period_dt__"], y=sub2["kurs maks."],
#                                 mode="lines", name=f"{fam} — kurs maks.", line=dict(dash="dot"),
#                                 hovertemplate="Max: %{y}<extra></extra>", showlegend=True
#                             ))
#                             fig.add_trace(go.Scatter(
#                                 x=sub2["__period_dt__"], y=sub2["kurs min."],
#                                 mode="lines", name=f"{fam} — kurs min.", line=dict(dash="dot"),
#                                 hovertemplate="Min: %{y}<extra></extra>", showlegend=True
#                             ))
#                             # (jeśli chcesz „band”, możemy dodać fill='tonexty' – daj znać)
#
#                 fig.update_layout(
#                     title=f"RTPE — {y_metric} wg rodzin (oś X z kodu kontraktu)",
#                     xaxis_title="Okres (z sufiksu instrumentu)",
#                     yaxis_title=y_metric,
#                     hovermode="x unified",
#                     legend_title="Rodzina"
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#
#                 if show_table:
#                     st.dataframe(
#                         dplot.rename(columns={
#                             "__family__":"Rodzina", "__tenor__":"Tenor",
#                             "__period_dt__":"Okres (data)", "__period_lbl__":"Okres (etykieta)"
#                         })[["Rodzina","Tenor","Okres (data)","Okres (etykieta)","instrument", y_metric]].reset_index(drop=True),
#                         use_container_width=True
#                     )
#
#             with st.expander("Informacje o danych / parsowaniu"):
#                 st.write("• Linie reprezentują **rodziny** (np. `BASE_M`, `BASE_Q`, `BASE_Y`).")
#                 st.write("• Oś X to **okres** wyciągnięty z sufiksu kontraktu:")
#                 st.write("  - `*_M-mm-yy` → pierwszy dzień miesiąca")
#                 st.write("  - `*_Q-q-yy` → pierwszy dzień kwartału")
#                 st.write("  - `*_Y-yy` → 1 stycznia danego roku")
#                 st.write("  - `*_W-ww-yy` → poniedziałek ISO tygodnia")
# # =================== /RTPE CSV: koniec sekcji =====================
