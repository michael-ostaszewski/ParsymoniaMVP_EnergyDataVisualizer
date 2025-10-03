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

    # Normalizacja liczb (polskie przecinki)
    numeric_like = [
        "DKR","kurs min.","kurs maks.","najlepsza oferta kupna","najlepsza oferta sprzedaży",
        "wol. obrotu","liczba kontraktów","liczba transakcji","wartość obrotu",
        "liczba otwartych pozycji","liczba kontraktów od początku"
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
            # ── UI: wybór rodzin (BASE_M, BASE_Q, BASE_Y, PEAK5_Y, ...) ──
            families = sorted(dfp["__family__"].dropna().unique().tolist())
            default_fams = [f for f in families if f.endswith("_Y")] or families[:3]

            st.markdown("**Ustawienia wykresu**")
            colL, colR = st.columns([2,1])
            with colL:
                selected_fams = st.multiselect(
                    "Wybierz rodziny instrumentów (linia = rodzina):",
                    options=families,
                    default=default_fams,
                    help="Np. BASE_M, BASE_Q, BASE_Y. Oś X to okres wyliczony z sufiksu kontraktu."
                )
            with colR:
                metric_candidates = [
                    c for c in [
                        "DKR","kurs min.","kurs maks.","najlepsza oferta kupna","najlepsza oferta sprzedaży",
                        "wol. obrotu","liczba kontraktów","liczba transakcji","wartość obrotu",
                        "liczba otwartych pozycji","liczba kontraktów od początku"
                    ] if c in dfp.columns
                ]
                default_metric = "DKR" if "DKR" in metric_candidates else (metric_candidates[0] if metric_candidates else None)
                y_metric = st.selectbox("Metryka (oś Y):", options=metric_candidates, index=metric_candidates.index(default_metric) if default_metric else 0)

            colA, colB, colC = st.columns(3)
            with colA:
                use_markers = st.checkbox("Punkty na wykresie", value=True)
            with colB:
                show_minmax = st.checkbox("Pokaż min/max (jeśli Y to cena)", value=("kurs min." in dfp.columns and "kurs maks." in dfp.columns))
            with colC:
                show_table = st.checkbox("Pokaż tabelę danych", value=False)

            # Dane do wykresu
            dplot = dfp[dfp["__family__"].isin(selected_fams)].copy()
            # wybierz potrzebne kolumny i usuń NaN w metryce lub czasie
            needed_cols = ["instrument","__family__","__tenor__","__period_dt__","__period_lbl__", y_metric]
            dplot = dplot[needed_cols].dropna(subset=["__period_dt__", y_metric])

            if dplot.empty or not selected_fams:
                st.info("Wybierz co najmniej jedną rodzinę oraz metrykę.")
            else:
                # posortuj po czasie
                dplot = dplot.sort_values("__period_dt__")

                # Rysunek: jedna linia na rodzinę
                fig = go.Figure()
                for fam in sorted(dplot["__family__"].unique()):
                    sub = dplot[dplot["__family__"] == fam]
                    fig.add_trace(go.Scatter(
                        x=sub["__period_dt__"],
                        y=sub[y_metric],
                        mode="lines+markers" if use_markers else "lines",
                        name=fam,
                        hovertemplate=("Rodzina: %{text}<br>Okres: %{x|%Y-%m-%d}<br>"
                                       + f"{y_metric}: "+"%{y}<extra></extra>"),
                        text=[fam]*len(sub)
                    ))

                    # Opcjonalny zakres min/max (tylko gdy to ma sens)
                    if show_minmax and ("kurs min." in dplot.columns) and ("kurs maks." in dplot.columns) and y_metric in ["DKR","kurs min.","kurs maks."]:
                        sub2 = dfp[(dfp["__family__"] == fam)].dropna(subset=["__period_dt__"])
                        sub2 = sub2.sort_values("__period_dt__")
                        if "kurs min." in sub2.columns and "kurs maks." in sub2.columns:
                            fig.add_trace(go.Scatter(
                                x=sub2["__period_dt__"], y=sub2["kurs maks."],
                                mode="lines", name=f"{fam} — kurs maks.", line=dict(dash="dot"),
                                hovertemplate="Max: %{y}<extra></extra>", showlegend=True
                            ))
                            fig.add_trace(go.Scatter(
                                x=sub2["__period_dt__"], y=sub2["kurs min."],
                                mode="lines", name=f"{fam} — kurs min.", line=dict(dash="dot"),
                                hovertemplate="Min: %{y}<extra></extra>", showlegend=True
                            ))
                            # (jeśli chcesz „band”, możemy dodać fill='tonexty' – daj znać)

                fig.update_layout(
                    title=f"RTPE — {y_metric} wg rodzin (oś X z kodu kontraktu)",
                    xaxis_title="Okres (z sufiksu instrumentu)",
                    yaxis_title=y_metric,
                    hovermode="x unified",
                    legend_title="Rodzina"
                )
                st.plotly_chart(fig, use_container_width=True)

                if show_table:
                    st.dataframe(
                        dplot.rename(columns={
                            "__family__":"Rodzina", "__tenor__":"Tenor",
                            "__period_dt__":"Okres (data)", "__period_lbl__":"Okres (etykieta)"
                        })[["Rodzina","Tenor","Okres (data)","Okres (etykieta)","instrument", y_metric]].reset_index(drop=True),
                        use_container_width=True
                    )

            with st.expander("Informacje o danych / parsowaniu"):
                st.write("• Linie reprezentują **rodziny** (np. `BASE_M`, `BASE_Q`, `BASE_Y`).")
                st.write("• Oś X to **okres** wyciągnięty z sufiksu kontraktu:")
                st.write("  - `*_M-mm-yy` → pierwszy dzień miesiąca")
                st.write("  - `*_Q-q-yy` → pierwszy dzień kwartału")
                st.write("  - `*_Y-yy` → 1 stycznia danego roku")
                st.write("  - `*_W-ww-yy` → poniedziałek ISO tygodnia")
# =================== /RTPE CSV: koniec sekcji =====================