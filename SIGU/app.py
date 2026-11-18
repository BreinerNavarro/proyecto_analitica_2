import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="SIGU | Gestión Inteligente de Urgencias",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 SIGU — Sistema Inteligente de Gestión de Urgencias")
st.caption("Análisis Probabilístico, Inferencia Estadística y Simulador Predictivo de Tiempos de Atención")

# ---------------------------------------------------------
# 2. CARGA DE DATOS (BUSCA EN ./ Y EN data/)
# ---------------------------------------------------------
@st.cache_data
def cargar_datos():
    if os.path.exists("data/urgencias_limpio.csv"):
        ruta = "data/urgencias_limpio.csv"
    elif os.path.exists("urgencias_limpio.csv"):
        ruta = "urgencias_limpio.csv"
    else:
        st.error("No se encontró 'urgencias_limpio.csv'. Colócalo junto a app.py o dentro de la carpeta data/.")
        st.stop()

    df = pd.read_csv(ruta)
    df["dt_ingreso"] = pd.to_datetime(df["dt_ingreso"])
    df["dt_atencion"] = pd.to_datetime(df["dt_atencion"])
    # Depurar los 172 desfases mínimos de reloj (< 0 min)
    df = df[df["tiempo_espera_min"] >= 0].copy()
    return df

df = cargar_datos()

# ---------------------------------------------------------
# 3. BARRA LATERAL: FILTROS OPERATIVOS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Filtros Operativos")

sedes_ordenadas = sorted(df["Ips"].unique().tolist())
opciones_sede = ["Todas las sedes"] + sedes_ordenadas
idx_default = opciones_sede.index("HOSPITAL LOCAL CIVIL") if "HOSPITAL LOCAL CIVIL" in opciones_sede else 0

sede_sel = st.sidebar.selectbox("IPS / Sede hospitalaria:", options=opciones_sede, index=idx_default)

franja_sel = st.sidebar.selectbox(
    "Franja horaria:",
    options=["Todas las franjas"] + sorted(df["franja_horaria"].unique().tolist())
)

triages_disponibles = ["I", "II", "III", "IV", "V"]
triage_sel = st.sidebar.multiselect(
    "Nivel de Triage:",
    options=triages_disponibles,
    default=triages_disponibles
)

# Aplicar filtros
df_filtrado = df[df["Triage"].isin(triage_sel)].copy()
if sede_sel != "Todas las sedes":
    df_filtrado = df_filtrado[df_filtrado["Ips"] == sede_sel]
if franja_sel != "Todas las franjas":
    df_filtrado = df_filtrado[df_filtrado["franja_horaria"] == franja_sel]

st.sidebar.markdown("---")
st.sidebar.metric("Registros analizados", f"{len(df_filtrado):,}")

if len(df_filtrado) < 30:
    st.warning("La selección actual tiene muy pocos registros (< 30). Amplía los filtros en el panel izquierdo.")
    st.stop()

# ---------------------------------------------------------
# 4. ESTRUCTURA DE PESTAÑAS (METODOLOGÍA DEL PROYECTO)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Exploración Descriptiva",
    "🎲 2. Modelación Probabilística",
    "📐 3. Inferencia y Bondad de Ajuste",
    "🤖 4. Regresión y Simulador ML"
])

# =========================================================
# PESTAÑA 1: ESTADÍSTICA DESCRIPTIVA
# =========================================================
with tab1:
    st.subheader(f"Panorama Descriptivo — {sede_sel}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pacientes", f"{len(df_filtrado):,}")
    c2.metric("Espera Promedio", f"{df_filtrado['tiempo_espera_min'].mean():.1f} min")
    c3.metric("Mediana de Espera", f"{df_filtrado['tiempo_espera_min'].median():.1f} min")
    c4.metric("Desviación Estándar", f"{df_filtrado['tiempo_espera_min'].std():.1f} min")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Distribución del Tiempo de Espera (Histograma)**")
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.histplot(df_filtrado["tiempo_espera_min"], bins=40, kde=True, color="#1f77b4", ax=ax1)
        ax1.axvline(df_filtrado["tiempo_espera_min"].mean(), color="red", linestyle="--", label="Media")
        ax1.axvline(df_filtrado["tiempo_espera_min"].median(), color="green", linestyle="-.", label="Mediana")
        ax1.set_xlabel("Tiempo de espera (minutos)")
        ax1.set_ylabel("Frecuencia de pacientes")
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_g2:
        st.markdown("**Tiempo de Espera por Nivel de Triage (Boxplot)**")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df_filtrado,
            x="Triage",
            y="tiempo_espera_min",
            order=["I", "II", "III", "IV", "V"],
            hue="Triage",
            palette="Blues",
            legend=False,
            ax=ax2
        )
        ax2.set_xlabel("Nivel de Triage")
        ax2.set_ylabel("Tiempo de espera (minutos)")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("**Demanda Acumulada de Ingresos por Hora del Día**")
    demanda_hora = df_filtrado.groupby("hora_dia").size().reset_index(name="pacientes")
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    sns.barplot(data=demanda_hora, x="hora_dia", y="pacientes", color="#2b5c8f", ax=ax3)
    ax3.set_xlabel("Hora del día (0 a 23)")
    ax3.set_ylabel("Total de ingresos")
    st.pyplot(fig3)
    plt.close(fig3)

# =========================================================
# PESTAÑA 2: MODELACIÓN PROBABILÍSTICA
# =========================================================
with tab2:
    st.subheader("Ajuste de Distribuciones y Probabilidades de Congestión")

    # Construir serie horaria solo sobre meses activos (evita el hueco 2020-2021)
    meses_activos = set(df_filtrado["anio_mes"].unique())
    rango_h = pd.date_range(
        df_filtrado["dt_ingreso"].min().floor("h"),
        df_filtrado["dt_ingreso"].max().ceil("h"),
        freq="h"
    )
    rango_h = rango_h[rango_h.to_period("M").astype(str).isin(meses_activos)]

    if franja_sel != "Todas las franjas":
        horas_validas = df_filtrado["hora_dia"].unique()
        rango_h = rango_h[rango_h.hour.isin(horas_validas)]

    serie_horas = df_filtrado.set_index("dt_ingreso").resample("h").size().reindex(rango_h, fill_value=0)
    lam = serie_horas.mean()

    col_p1, col_p2 = st.columns(2)

    # --- A. POISSON ---
    with col_p1:
        st.markdown("### 1. Llegadas por Hora — Poisson (λ)")
        umbral_llegadas = st.slider("Umbral crítico de pacientes/hora (k):", 1, 25, 5)
        prob_poisson = 1 - stats.poisson.cdf(umbral_llegadas, mu=lam)

        m_p1, m_p2 = st.columns(2)
        m_p1.metric("Tasa promedio (λ)", f"{lam:.2f} pac/hora")
        m_p2.metric(f"Riesgo P(X > {umbral_llegadas} pac/h)", f"{prob_poisson * 100:.2f}%")

        k_vals = np.arange(0, max(12, int(serie_horas.max()) + 1))
        pmf_teorica = stats.poisson.pmf(k_vals, mu=lam)
        freq_obs = serie_horas.value_counts(normalize=True).reindex(k_vals, fill_value=0)

        fig_p, ax_p = plt.subplots(figsize=(6, 3.8))
        ax_p.bar(k_vals - 0.15, freq_obs.values, width=0.35, label="Observado", color="#7fb3d5")
        ax_p.bar(k_vals + 0.15, pmf_teorica, width=0.35, label=f"Poisson (λ={lam:.2f})", color="#1f618d")
        ax_p.set_xlabel("Pacientes que ingresan en una hora")
        ax_p.set_ylabel("Probabilidad")
        ax_p.legend()
        st.pyplot(fig_p)
        plt.close(fig_p)

    # --- B. NORMAL VS LOG-NORMAL ---
    with col_p2:
        st.markdown("### 2. Tiempo de Espera — Normal vs. Log-normal")
        espera_pos = df_filtrado[df_filtrado["tiempo_espera_min"] > 0]["tiempo_espera_min"]

        mu_norm, std_norm = stats.norm.fit(espera_pos)
        shape_log, loc_log, scale_log = stats.lognorm.fit(espera_pos, floc=0)

        umbral_min = st.slider("Tiempo máximo de espera tolerado (min):", 15, 180, 60)
        prob_norm = 1 - stats.norm.cdf(umbral_min, loc=mu_norm, scale=std_norm)
        prob_log = 1 - stats.lognorm.cdf(umbral_min, s=shape_log, loc=loc_log, scale=scale_log)

        m_e1, m_e2 = st.columns(2)
        m_e1.metric(f"Normal: P(X > {umbral_min} min)", f"{prob_norm * 100:.2f}%")
        m_e2.metric(f"Log-normal: P(X > {umbral_min} min)", f"{prob_log * 100:.2f}%")

        x_vals = np.linspace(0.1, np.percentile(espera_pos, 99), 200)
        fig_e, ax_e = plt.subplots(figsize=(6, 3.8))
        sns.histplot(espera_pos, bins=35, stat="density", color="#d5dbdb", label="Datos reales", ax=ax_e)
        ax_e.plot(x_vals, stats.norm.pdf(x_vals, mu_norm, std_norm), "r--", lw=2, label="Normal")
        ax_e.plot(x_vals, stats.lognorm.pdf(x_vals, shape_log, loc=0, scale=scale_log), "g-", lw=2, label="Log-normal")
        ax_e.axvline(umbral_min, color="black", linestyle=":", label=f"Umbral ({umbral_min} min)")
        ax_e.set_xlabel("Tiempo de espera (minutos)")
        ax_e.legend()
        st.pyplot(fig_e)
        plt.close(fig_e)

    # --- C. EXPONENCIAL ---
    st.markdown("---")
    st.markdown("### 3. Tiempo entre Llegadas — Distribución Exponencial (1/λ)")
    tel_pos = df_filtrado[
        (df_filtrado["tiempo_entre_llegadas_min"] > 0) &
        (df_filtrado["tiempo_entre_llegadas_min"] <= 1440)
    ]["tiempo_entre_llegadas_min"].dropna()

    if len(tel_pos) > 10:
        _, scale_exp = stats.expon.fit(tel_pos, floc=0)
        min_rapido = st.slider("Intervalo entre llegadas consecutivas (minutos):", 1, 30, 5)
        prob_exp = stats.expon.cdf(min_rapido, loc=0, scale=scale_exp)

        ce1, ce2 = st.columns(2)
        ce1.metric("Tiempo promedio entre llegadas (1/λ)", f"{scale_exp:.2f} min")
        ce2.metric(f"Probabilidad de llegada rápida P(X < {min_rapido} min)", f"{prob_exp * 100:.2f}%")

# =========================================================
# PESTAÑA 3: INFERENCIA Y BONDAD DE AJUSTE
# =========================================================
with tab3:
    st.subheader("Estimación por Intervalos, Pruebas de Hipótesis y Bondad de Ajuste")

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("### 1. Intervalo de Confianza (95%) y Prueba t-Student")
        n_muestra = st.slider("Tamaño de muestra aleatoria (n):", 30, min(1000, len(df_filtrado)), 150)
        meta_inst = st.number_input("Meta máxima institucional de espera (μ₀ en min):", value=45.0, step=5.0)

        muestra = df_filtrado["tiempo_espera_min"].sample(n=n_muestra, random_state=42)
        media_m = muestra.mean()
        sem_m = stats.sem(muestra)
        ic_inf, ic_sup = stats.t.interval(0.95, df=n_muestra - 1, loc=media_m, scale=sem_m)

        # Prueba unilateral H1: mu > meta_inst
        t_stat, p_val_bi = stats.ttest_1samp(muestra, popmean=meta_inst)
        p_val_uni = p_val_bi / 2 if t_stat > 0 else 1 - (p_val_bi / 2)

        st.write(f"**Media muestral:** `{media_m:.2f} min`")
        st.write(f"**IC 95% para la media:** `[{ic_inf:.2f} min , {ic_sup:.2f} min]`")
        st.write(f"**Estadístico t:** `{t_stat:.4f}` | **Valor p (unilateral):** `{p_val_uni:.4e}`")

        if p_val_uni < 0.05:
            st.error(f"Se rechaza H₀ (α = 0.05): El tiempo promedio supera estadísticamente la meta de {meta_inst} min.")
        else:
            st.success(f"No se rechaza H₀ (α = 0.05): El tiempo promedio cumple estadísticamente con la meta de {meta_inst} min.")

    with col_i2:
        st.markdown("### 2. Bondad de Ajuste Chi-Cuadrado (χ²) para Poisson")
        obs_counts = serie_horas.value_counts().sort_index()
        vals = obs_counts.index.values
        f_obs = obs_counts.values
        f_exp = stats.poisson.pmf(vals, mu=lam) * len(serie_horas)

        valid = f_exp >= 5
        if valid.sum() >= 2:
            f_obs_g = np.append(f_obs[valid], f_obs[~valid].sum())
            f_exp_g = np.append(f_exp[valid], f_exp[~valid].sum())
            f_exp_g = f_exp_g * (f_obs_g.sum() / f_exp_g.sum())
            chi2_stat, p_chi2 = stats.chisquare(f_obs_g, f_exp_g, ddof=1)

            st.write(f"**Estadístico χ²:** `{chi2_stat:.2f}`")
            st.write(f"**Valor p:** `{p_chi2:.4e}`")
            if p_chi2 < 0.05:
                st.warning("p < 0.05: Al mezclar distintas franjas horarias o días, λ varía (no es estacionario). Filtra por una franja horaria específica en la barra lateral para evaluar el supuesto de tasa constante.")
            else:
                st.success("p ≥ 0.05: Las llegadas se ajustan estadísticamente a una distribución de Poisson.")
        else:
            st.info("No hay suficientes clases con frecuencia esperada ≥ 5 para ejecutar χ².")

# =========================================================
# PESTAÑA 4: REGRESIÓN Y SIMULADOR PREDICTIVO
# =========================================================
@st.cache_resource
def entrenar_modelo_rf(df_datos):
    df_ml = df_datos.copy()
    df_ml["fecha_hora"] = df_ml["dt_ingreso"].dt.floor("h")
    carga_h = df_ml.groupby("fecha_hora").size().rename("carga_horaria")
    df_sup = df_ml.merge(carga_h, on="fecha_hora")

    X = pd.get_dummies(df_sup[["hora_dia", "carga_horaria", "Triage"]], drop_first=False)
    y = df_sup["tiempo_espera_min"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return modelo, X.columns.tolist(), mae, r2

with tab4:
    st.subheader("Análisis Bivariado y Simulador Predictivo de Dotación")

    df_reg = df_filtrado.copy()
    df_reg["fecha_hora"] = df_reg["dt_ingreso"].dt.floor("h")
    resumen_horario = df_reg.groupby("fecha_hora").agg(
        llegadas_hora=("tiempo_espera_min", "count"),
        espera_promedio=("tiempo_espera_min", "mean")
    ).reset_index()

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("### 1. Correlación y Regresión Lineal Simple")
        if len(resumen_horario) > 5:
            r_val, p_r = stats.pearsonr(resumen_horario["llegadas_hora"], resumen_horario["espera_promedio"])
            reg = LinearRegression()
            reg.fit(resumen_horario[["llegadas_hora"]], resumen_horario["espera_promedio"])

            st.write(f"**Correlación de Pearson (r):** `{r_val:.4f}` (valor p: `{p_r:.4e}`)")
            st.write(f"**Recta estimada:** `Espera = {reg.intercept_:.2f} + {reg.coef_[0]:.2f} × Llegadas/hora`")

            fig_r, ax_r = plt.subplots(figsize=(6, 3.8))
            sns.regplot(
                data=resumen_horario.sample(min(1500, len(resumen_horario)), random_state=42),
                x="llegadas_hora",
                y="espera_promedio",
                scatter_kws={"alpha": 0.3, "color": "#2b5c8f"},
                line_kws={"color": "red"},
                ax=ax_r
            )
            ax_r.set_xlabel("Pacientes que ingresan en la hora")
            ax_r.set_ylabel("Tiempo de espera promedio (min)")
            st.pyplot(fig_r)
            plt.close(fig_r)

    with col_r2:
        st.markdown("### 2. Simulador Predictivo de Personal (Machine Learning)")
        modelo_rf, cols_modelo, mae, r2 = entrenar_modelo_rf(df_filtrado)
        st.caption(f"Validación Random Forest — **MAE:** {mae:.1f} min | **R²:** {r2:.2f}")

        sim_hora = st.slider("Hora del turno a simular (0 - 23):", 0, 23, 10)
        sim_llegadas = st.slider("Demanda esperada (pacientes/hora):", 1, 20, 6)
        sim_triage = st.selectbox("Nivel de Triage:", ["I", "II", "III", "IV", "V"], index=2)
        sim_medicos = st.slider("Médicos asignados al turno:", 1, 8, 2)

        fila_sim = pd.DataFrame(0, index=[0], columns=cols_modelo)
        fila_sim["hora_dia"] = sim_hora
        # Ajuste operativo: mayor dotación médica distribuye la carga efectiva por profesional
        fila_sim["carga_horaria"] = sim_llegadas * (2.0 / sim_medicos)
        col_triage = f"Triage_{sim_triage}"
        if col_triage in fila_sim.columns:
            fila_sim[col_triage] = 1

        espera_estimada = modelo_rf.predict(fila_sim)[0]

        st.metric(
            label=f"Tiempo de espera proyectado ({sim_medicos} médico/s)",
            value=f"{espera_estimada:.1f} min",
            delta=f"{espera_estimada - 60:.1f} min vs umbral crítico (60 min)",
            delta_color="inverse"
        )