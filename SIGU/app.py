# Interfaz grafica de SIGU

import io
import numpy as np
import pandas as pd
import backend as be
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SIGU – Triage Urgencias", page_icon="🏥", layout="wide")


# ------------------------------------------------------------------ datos (cache)
@st.cache_data
def datos():
    return be.cargar_datos(), be.cargar_llegadas(), be.cargar_calidad()


@st.cache_resource
def entrenar(ips_sel: str):
    """Entrena la regresión de Poisson una sola vez por IPS seleccionada."""
    _, lleg_todas, _ = datos()
    l = lleg_todas if ips_sel == "Todas" else lleg_todas[lleg_todas["Ips"] == ips_sel]
    return be.modelo_regresion_poisson(l)


try:
    df, llegadas, calidad = datos()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


def tarjeta_prueba(r: dict):
    """Muestra H0/H1, estadístico, p y decisión de una prueba."""
    st.markdown(f"**H₀:** {r['h0']}  \n**H₁:** {r['h1']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estadístico", f"{r['estadistico']:.3f}")
    c2.metric("Grados de libertad" if "gl" in r else "n muestra", r.get("gl", r.get("n")))
    c3.metric("Valor p", be.formato_p(r["p"]))
    if r["p"] < r["alpha"]:
        st.error(f"**{r['decision']}** (p < α = {r['alpha']})")
    else:
        st.success(f"**{r['decision']}** (p ≥ α = {r['alpha']})")


# ------------------------------------------------------------------ barra lateral
st.sidebar.header("Filtros")
triages_disp = [t for t in be.TRIAGE_ORDEN if t in df["Triage"].unique()]
triages = st.sidebar.multiselect("Triage", triages_disp, default=triages_disp)
ips = st.sidebar.selectbox("IPS", ["Todas"] + sorted(df["Ips"].dropna().unique().tolist()))
alpha = st.sidebar.select_slider("Nivel de significancia α", [0.01, 0.05, 0.10], value=0.05)
st.sidebar.caption("El filtro de triage no aplica a las pestañas Poisson, Modelo y Malla "
                   "(las llegadas se cuentan por IPS-hora).")

dff = be.filtrar(df, triages, ips)
lleg = llegadas if ips == "Todas" else llegadas[llegadas["Ips"] == ips]

st.title("🏥 Tiempos de espera y demanda en urgencias")
st.caption("MVP – Analítica Descriptiva II y Probabilidad")

if dff.empty:
    st.warning("No hay registros con los filtros seleccionados.")
    st.stop()

tab_datos, tab_desc, tab_dias, tab_poi, tab_mod, tab_malla = st.tabs(
    ["📋 Datos", "📊 Descriptiva", "📅 χ² por día", "🎲 Poisson",
     "🤖 Regresión Poisson", "👩‍⚕️ Malla de turnos"])

# ------------------------------------------------------------------ 1. Datos
with tab_datos:
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros filtrados", f"{len(dff):,}")
    c2.metric("IPS", dff["Ips"].nunique())
    c3.metric("Periodo", f"{dff['dt_ingreso'].min():%d/%m/%Y} – {dff['dt_ingreso'].max():%d/%m/%Y}")
    st.subheader("Calidad de datos (limpieza)")
    st.dataframe(calidad, hide_index=True, use_container_width=True)
    st.subheader("Vista previa")
    st.dataframe(dff.head(100), use_container_width=True)

# ------------------------------------------------------------------ 2. Descriptiva
with tab_desc:
    espera = dff["tiempo_espera_min"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media (min)", f"{espera.mean():.2f}")
    c2.metric("Mediana (min)", f"{espera.median():.2f}")
    c3.metric("Desv. estándar", f"{espera.std():.2f}" if len(espera) > 1 else "–")
    c4.metric("Asimetría", f"{espera.skew():.2f}" if len(espera) > 2 else "–")

    st.subheader("Estadísticos por triage")
    st.dataframe(be.descriptivos(dff), use_container_width=True)
    if len(dff) < 100:
        st.info("Pocos registros con estos filtros; interprete los resultados con cautela.")

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(espera, bins=50, ax=ax, color="#1f77b4")
        ax.axvline(espera.mean(), color="red", ls="--", label="Media")
        ax.axvline(espera.median(), color="green", ls="--", label="Mediana")
        ax.set(title="Histograma del tiempo de espera", xlabel="Minutos", ylabel="Frecuencia")
        ax.legend()
        st.pyplot(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(6, 4))
        orden = [t for t in be.TRIAGE_ORDEN if t in dff["Triage"].unique()]
        ax.boxplot([dff.loc[dff["Triage"] == t, "tiempo_espera_min"] for t in orden],
                   flierprops={"markersize": 2})
        ax.set_xticks(range(1, len(orden) + 1), orden)
        ax.set(xlabel="Triage", title="Espera por triage", ylabel="Minutos")
        st.pyplot(fig)

    st.subheader("Intervalo de confianza 95 % de la media")
    st.dataframe(be.ic_por_triage(dff).rename(columns={"li": "Límite inf.", "ls": "Límite sup."}),
                 hide_index=True, use_container_width=True)

    st.subheader("Normalidad (Shapiro–Wilk)")
    r = be.prueba_normalidad(espera, alpha)
    if r:
        tarjeta_prueba(r)
        st.caption("Muestra aleatoria de hasta 5 000 registros (semilla fija).")
    else:
        st.info("No hay suficientes datos para la prueba.")

# ------------------------------------------------------------------ 3. χ² uniforme días
with tab_dias:
    st.subheader("¿Las atenciones se reparten igual entre los días de la semana?")
    r = be.chi2_uniforme_dias(dff, alpha)
    tarjeta_prueba(r)
    c1, c2 = st.columns([3, 2])
    with c1:
        t = r["tabla"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(t.index, t["Observado"], color="#1f77b4", label="Observado")
        ax.plot(t.index, t["Esperado"], "r--o", label="Esperado (uniforme)")
        ax.set(title="Atenciones por día: observado vs esperado", ylabel="Atenciones")
        ax.legend()
        plt.xticks(rotation=30)
        st.pyplot(fig)
    with c2:
        st.dataframe(r["tabla"], use_container_width=True)
    st.caption("Con muestras grandes, diferencias pequeñas resultan significativas; "
               "compare también visualmente O vs E.")

# ------------------------------------------------------------------ 4. Poisson
with tab_poi:
    st.subheader("Llegadas de pacientes por IPS y hora")
    if lleg.empty:
        st.warning("No hay datos de llegadas para la IPS seleccionada.")
    else:
        lam = lleg["llegadas"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("λ (media llegadas/hora)", f"{lam:.3f}")
        c2.metric("Varianza", f"{lleg['llegadas'].var():.3f}" if len(lleg) > 1 else "–")
        c3.metric("Bloques IPS-hora", f"{len(lleg):,}")
        st.caption("En una Poisson la media y la varianza deberían ser parecidas.")

        st.markdown("#### Calculadora de probabilidades")
        k = st.slider("Número de llegadas en una hora (k)", 0, 20, int(round(lam)))
        pr = be.prob_poisson(lam, k)
        p1, p2, p3 = st.columns(3)
        p1.metric(f"P(X = {k})", f"{pr['P(X = k)']:.4f}")
        p2.metric(f"P(X ≤ {k})", f"{pr['P(X ≤ k)']:.4f}")
        p3.metric(f"P(X ≥ {k})", f"{pr['P(X ≥ k)']:.4f}")
        st.latex(r"P(X=k)=\frac{e^{-\lambda}\lambda^{k}}{k!}")

        st.markdown("#### Prueba χ² de bondad de ajuste")
        r = be.chi2_poisson(lleg, alpha)
        if r is None:
            st.warning("Esta IPS tiene muy pocos registros: no hay suficientes categorías con "
                       "frecuencia esperada ≥ 5 para aplicar la prueba χ². "
                       "Seleccione otra IPS o 'Todas'.")
        else:
            tarjeta_prueba(r)
            c1, c2 = st.columns([3, 2])
            with c1:
                t = r["tabla"]
                x = np.arange(len(t))
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(x - 0.2, t["Observado"], 0.4, label="Observado")
                ax.bar(x + 0.2, t["Esperado"], 0.4, label="Poisson esperado")
                ax.set_xticks(x, t["k (llegadas/hora)"])
                ax.set(title="Llegadas por hora: observado vs Poisson",
                       xlabel="Llegadas en una hora", ylabel="Número de horas")
                ax.legend()
                st.pyplot(fig)
            with c2:
                st.dataframe(r["tabla"], hide_index=True, use_container_width=True)
            st.caption("Celdas con frecuencia esperada < 5 agrupadas en las colas. "
                       "gl = categorías − 1 − 1 (λ estimado).")

# ------------------------------------------------------------------ 5. Regresión Poisson
with tab_mod:
    st.subheader("Modelo de regresión de Poisson para predecir llegadas por hora")
    st.latex(r"\log(\lambda)=\beta_0+\beta_{\text{día}}+\beta_{\text{hora}}"
             r"+\beta_{\text{mes}}+\beta_{\text{IPS}}")
    res = entrenar(ips)
    if res is None:
        st.warning("No hay datos suficientes para entrenar el modelo con esta IPS. "
                   "Seleccione otra IPS o 'Todas'.")
    else:
        st.caption(f"Entrenamiento: {res['n_entrenamiento']:,} bloques IPS-hora hasta "
                   f"{res['corte']:%d/%m/%Y} · Prueba: {res['n_prueba']:,} bloques hasta "
                   f"{res['fin']:%d/%m/%Y} (último 20 % del periodo, datos no vistos).")

        met = res["metricas"]
        mae_m, mae_b = met.loc["MAE", "Regresión Poisson"], met.loc["MAE", "Línea base (promedio)"]
        mejora = (mae_b - mae_m) / mae_b * 100 if mae_b > 0 else 0
        phi = res["phi"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE modelo", f"{mae_m:.3f} pac/h")
        c2.metric("MAE línea base", f"{mae_b:.3f} pac/h")
        c3.metric("Mejora vs base", f"{mejora:.1f} %", delta=f"{mejora:.1f} %")
        c4.metric("Dispersión φ", f"{phi:.2f}")

        # Interpretación de φ
        if phi < 0.8:
            st.info(f"φ = {phi:.2f} < 1: **subdispersión**. Hay menos variación que la que "
                    "supone la Poisson; el percentil de la malla es conservador.")
        elif phi <= 1.5:
            st.success(f"φ = {phi:.2f} ≈ 1: la varianza es similar a la media, "
                       "**el supuesto de Poisson es razonable**.")
        else:
            st.warning(f"φ = {phi:.2f} > 1,5: **sobredispersión**. Los datos varían más de lo "
                       "que supone la Poisson; los errores estándar se subestiman y conviene "
                       "una Binomial Negativa o subir el nivel de servicio de la malla.")

        st.markdown("#### Métricas en el conjunto de prueba")
        st.dataframe(met, use_container_width=True)
        with st.expander("¿Qué significa cada métrica?"):
            st.markdown(
                "- **MAE** (error absoluto medio): en promedio, cuántos pacientes por hora se "
                "equivoca la predicción. Menor es mejor.\n"
                "- **RMSE**: como el MAE pero castiga más los errores grandes (horas pico).\n"
                "- **Devianza Poisson media**: error propio de datos de conteo; menor es mejor.\n"
                "- **D²**: % de la devianza explicada frente a predecir siempre la media "
                "(equivalente al R² para Poisson). 0 = no mejora; 1 = perfecto.\n"
                "- **φ (dispersión de Pearson)**: Σ(y−μ̂)²/μ̂ ÷ (n − p). "
                "En una Poisson ideal vale 1.")

        g1, g2 = st.columns(2)
        with g1:  # Comparación de métricas
            fig, ax = plt.subplots(figsize=(6, 4))
            filas = ["MAE", "RMSE", "Devianza Poisson media"]
            x = np.arange(len(filas))
            ax.bar(x - 0.2, met.loc[filas, "Regresión Poisson"], 0.4, label="Regresión Poisson")
            ax.bar(x + 0.2, met.loc[filas, "Línea base (promedio)"], 0.4,
                   label="Línea base (promedio)", color="#999999")
            for i, v in enumerate(met.loc[filas, "Regresión Poisson"]):
                ax.text(i - 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            for i, v in enumerate(met.loc[filas, "Línea base (promedio)"]):
                ax.text(i + 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, ["MAE", "RMSE", "Devianza"])
            ax.set(title="Errores en prueba (menor es mejor)", ylabel="Valor")
            ax.legend()
            st.pyplot(fig)
        with g2:  # Dispersión: varianza vs media por grupo
            fig, ax = plt.subplots(figsize=(6, 4))
            pred, resid = res["pred_pru"], res["residuos"]
            grupos = np.digitize(pred, np.quantile(pred, np.linspace(0, 1, 11)[1:-1]))
            medias = [pred[grupos == g].mean() for g in np.unique(grupos)]
            varianzas = [(resid[grupos == g] ** 2).mean() for g in np.unique(grupos)]
            lim = max(max(medias), max(varianzas)) * 1.1
            ax.plot([0, lim], [0, lim], "k--", label="Poisson ideal (Var = media, φ = 1)")
            ax.plot([0, lim], [0, lim * phi], "r:", label=f"φ estimado = {phi:.2f}")
            ax.scatter(medias, varianzas, color="#1f77b4", zorder=3, label="Grupos de predicción")
            ax.set(title="Parámetro de dispersión", xlabel="Media predicha (λ̂)",
                   ylabel="Varianza observada (y − λ̂)²")
            ax.legend(fontsize=8)
            st.pyplot(fig)

        st.markdown("#### Predicción vs realidad en el periodo de prueba")
        g3, g4 = st.columns(2)
        with g3:
            pdia = res["por_dia"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(pdia.index, pdia["observado"], label="Observado", color="black")
            ax.plot(pdia.index, pdia["modelo"], label="Regresión Poisson", color="#1f77b4")
            ax.plot(pdia.index, pdia["base"], label="Línea base", color="#999999", ls="--")
            ax.set(title="Pacientes por día", ylabel="Pacientes")
            ax.legend(fontsize=8)
            fig.autofmt_xdate()
            st.pyplot(fig)
        with g4:
            ph = res["por_hora"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ph.index, ph["observado"], "ko-", label="Observado", ms=3)
            ax.plot(ph.index, ph["modelo"], color="#1f77b4", label="Regresión Poisson")
            ax.plot(ph.index, ph["base"], color="#999999", ls="--", label="Línea base")
            ax.set(title="Promedio de llegadas por hora del día", xlabel="Hora",
                   ylabel="Pacientes por hora", xticks=range(0, 24, 2))
            ax.legend(fontsize=8)
            st.pyplot(fig)

# ------------------------------------------------------------------ 6. Malla de turnos
with tab_malla:
    st.subheader(f"Malla de turnos sugerida – {ips if ips != 'Todas' else 'todas las IPS (red)'}")
    if lleg.empty:
        st.warning("No hay datos de llegadas para la IPS seleccionada.")
    else:
        c1, c2, c3 = st.columns(3)
        cap = c1.number_input("Pacientes que atiende un médico por hora", 0.5, 20.0, 3.0, 0.5,
                              help="Supuesto operativo: ajústelo con el dato real de la IPS.")
        nivel = c2.slider("Nivel de servicio (cobertura de la demanda)", 0.50, 0.99, 0.90, 0.01,
                          help="0,90 = el número de médicos alcanza en el 90 % de las horas.")
        fuente = c3.radio("Predicción de λ", ["Promedio histórico", "Regresión Poisson"])

        res_m = entrenar(ips) if fuente == "Regresión Poisson" else None
        if fuente == "Regresión Poisson" and res_m is None:
            st.info("No se pudo entrenar el modelo con esta IPS; se usa el promedio histórico.")
        lam_tabla = (be.lambda_modelo_semana(res_m, lleg) if res_m is not None
                     else be.demanda_por_hora(lleg))
        pacientes, medicos = be.medicos_por_hora(lam_tabla, cap, nivel)
        malla = be.malla_turnos(medicos)

        m1, m2, m3 = st.columns(3)
        m1.metric("Pacientes promedio/semana", f"{lam_tabla.values.sum():,.0f}")
        m2.metric("Hora pico (λ)", f"{lam_tabla.values.max():.2f} pac/h")
        m3.metric("Horas-médico/semana", f"{malla['Horas-médico/día'].sum():,}")

        st.markdown("#### 1. Demanda predicha (λ pacientes por hora)")
        fig, ax = plt.subplots(figsize=(12, 3.8))
        sns.heatmap(lam_tabla, cmap="YlOrRd", ax=ax, cbar_kws={"label": "λ pacientes/hora"})
        ax.set(xlabel="Hora del día", ylabel="")
        st.pyplot(fig)

        st.markdown(f"#### 2. Médicos requeridos por hora (percentil {nivel:.0%} de la Poisson)")
        fig, ax = plt.subplots(figsize=(12, 3.8))
        sns.heatmap(medicos, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set(xlabel="Hora del día", ylabel="")
        st.pyplot(fig)

        st.markdown("#### 3. Malla de turnos (médicos por turno de 6 h)")
        st.dataframe(malla, use_container_width=True)
        
        # Crear un búfer en memoria y escribir el Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            malla.to_excel(writer, index=True, sheet_name='Malla_Turnos')

        # Botón de descarga en Streamlit
        st.download_button(
            label="⬇️ Descargar malla (Excel)",
            data=buffer.getvalue(),
            file_name=f"malla_turnos_{ips}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.caption(
            "Método: λ = promedio histórico por día y hora, o predicción de la regresión de "
            "Poisson para el último mes observado. Pacientes a cubrir = percentil del nivel de "
            "servicio de Poisson(λ). Médicos = ⌈pacientes / capacidad⌉, mínimo 1. "
            "Cada turno toma el máximo de sus horas para cubrir el pico.")
