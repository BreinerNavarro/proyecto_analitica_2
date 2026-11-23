# Logica de SIGU

# ___________ Librerías y constantes ___________
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Rutas relativas a este archivo: funcionan en cualquier PC / carpeta
RAIZ = Path(__file__).resolve().parent.parent          # Project_Analitica_II
CARPETA_DATOS = RAIZ / "data"
RUTA_CRUDO = CARPETA_DATOS / "Clasificación_en_Triage_Urgencias_20261003.csv"

ANIO = 2022                 # Año de análisis 
FORMATO_FECHA = "%Y %b %d %I:%M:%S %p"
MAX_ESPERA_MIN = None       # Ej. 360 si decides recortar registros extremos; None = no recortar
MAX_SALTO_LLEGADAS = 1440   # Saltos > 24 h entre llegadas se consideran huecos de registro
SEMILLA = 20261003

MAPA_DIAS = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
}
DIAS_ORDEN = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
TRIAGE_ORDEN = ["I", "II", "III", "IV", "V"]


# =====================================================================
#  LIMPIEZA 
# =====================================================================
def cargar_crudo(ruta: Path = RUTA_CRUDO) -> pd.DataFrame:
    """Lee el CSV original tal como viene de la fuente."""
    if not ruta.exists():
        # Busca el crudo aunque el nombre tenga otra codificación de la tilde
        candidatos = list(CARPETA_DATOS.glob("Clasificaci*Triage*.csv"))
        if not candidatos:
            raise FileNotFoundError(f"No se encontró el CSV crudo en {CARPETA_DATOS}")
        ruta = candidatos[0]
    return pd.read_csv(ruta, encoding="latin-1")


def asignar_franja(hora: int) -> str:
    if 6 <= hora < 12:
        return "Mañana (06:00 - 11:59)"
    elif 12 <= hora < 18:
        return "Tarde (12:00 - 17:59)"
    elif 18 <= hora < 24:
        return "Noche (18:00 - 23:59)"
    return "Madrugada (00:00 - 05:59)"

# Devuelve el DataFrame limpio y  (paso, registros restantes)
def limpiar(data: pd.DataFrame, anio: int = ANIO) -> tuple[pd.DataFrame, list]:
    log = [("Registros iniciales", len(data))]
    df = data.copy()

    # Nombres de columnas sin espacios
    df.columns = df.columns.str.strip()

    # Duplicados
    df = df.drop_duplicates()
    log.append(("Sin duplicados", len(df)))

    # Fechas completas (Hora_Ingre / Hora_Atencion traen año 1900, se ignoran)
    df["dt_ingreso"] = pd.to_datetime(df["Fecha_Ing"], format=FORMATO_FECHA, errors="coerce")
    df["dt_atencion"] = pd.to_datetime(df["Fecha_Atencion"], format=FORMATO_FECHA, errors="coerce")
    df = df.dropna(subset=["dt_ingreso", "dt_atencion"])
    log.append(("Con fechas válidas", len(df)))

    # Solo el año de análisis
    df = df[df["dt_ingreso"].dt.year == anio]
    log.append((f"Solo año {anio}", len(df)))

    # Variable principal: tiempo de espera (min) y eliminación de negativos
    df["tiempo_espera_min"] = (df["dt_atencion"] - df["dt_ingreso"]).dt.total_seconds() / 60.0
    df = df[df["tiempo_espera_min"] >= 0]
    log.append(("Sin esperas negativas", len(df)))

    if MAX_ESPERA_MIN is not None:
        df = df[df["tiempo_espera_min"] <= MAX_ESPERA_MIN]
        log.append((f"Espera <= {MAX_ESPERA_MIN} min", len(df)))

    # Categóricas
    df["Triage"] = df["Triage"].astype(str).str.strip().str.upper()
    df["Nom_Admini"] = df["Nom_Admini"].fillna("NO REPORTADA")

    # Variables de tiempo (días en español sin depender del idioma del sistema)
    df["hora_dia"] = df["dt_ingreso"].dt.hour
    df["dia_semana"] = df["dt_ingreso"].dt.day_name().map(MAPA_DIAS)
    df["anio_mes"] = df["dt_ingreso"].dt.to_period("M").astype(str)
    df["franja_horaria"] = df["hora_dia"].apply(asignar_franja)

    # Tiempo entre llegadas por IPS (sin rellenar: los NaN se quedan como NaN)
    df = df.sort_values(["Ips", "dt_ingreso"]).reset_index(drop=True)
    tel = df.groupby("Ips")["dt_ingreso"].diff().dt.total_seconds() / 60.0
    df["tiempo_entre_llegadas_min"] = tel.where(tel <= MAX_SALTO_LLEGADAS)

    # Redondeo y columnas finales
    df["tiempo_espera_min"] = df["tiempo_espera_min"].round(2)
    df["tiempo_entre_llegadas_min"] = df["tiempo_entre_llegadas_min"].round(2)
    columnas_finales = [
        "dt_ingreso", "dt_atencion", "anio_mes", "dia_semana", "hora_dia",
        "franja_horaria", "Triage", "Ips", "Red", "Nom_Admini",
        "tiempo_espera_min", "tiempo_entre_llegadas_min",
    ]
    return df[columnas_finales].copy(), log


def tabla_calidad_desde_log(log: list) -> pd.DataFrame:
    tabla = pd.DataFrame(log, columns=["Paso", "Registros"])
    tabla["Eliminados"] = (-tabla["Registros"].diff()).fillna(0).astype(int)
    return tabla


def llegadas_por_ips_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta ingresos por IPS en cada hora, incluyendo las horas con 0 llegadas
    entre el primer y el último registro de cada IPS."""
    tmp = df.assign(hora=df["dt_ingreso"].dt.floor("h"))
    partes = []
    for ips, g in tmp.groupby("Ips"):
        conteo = g.groupby("hora").size()
        rango = pd.date_range(conteo.index.min(), conteo.index.max(), freq="h")
        conteo = conteo.reindex(rango, fill_value=0)
        partes.append(pd.DataFrame({"Ips": ips, "hora": rango, "llegadas": conteo.values}))
    return pd.concat(partes, ignore_index=True)


def generar_archivos_limpios() -> None:
    """Ejecuta toda la limpieza y guarda los 3 CSV en /data."""
    data = cargar_crudo()
    print("Crudo:", data.shape)
    data_limpio, log = limpiar(data)
    calidad = tabla_calidad_desde_log(log)
    llegadas = llegadas_por_ips_hora(data_limpio)

    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    data_limpio.to_csv(CARPETA_DATOS / "urgencias_limpio.csv", index=False, encoding="utf-8-sig")
    llegadas.to_csv(CARPETA_DATOS / "llegadas_ips_hora.csv", index=False, encoding="utf-8-sig")
    calidad.to_csv(CARPETA_DATOS / "tabla_calidad.csv", index=False, encoding="utf-8-sig")

    print(calidad.to_string(index=False))
    print(f"Registros limpios: {len(data_limpio):,}")
    print(f"λ (media llegadas/hora): {llegadas['llegadas'].mean():.3f} | "
          f"varianza: {llegadas['llegadas'].var():.3f}")
    print("Archivos guardados en", CARPETA_DATOS)


# =====================================================================
# FUNCIONES QUE USA app.py
# =====================================================================
def _verificar(nombre: str) -> Path:
    ruta = CARPETA_DATOS / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}. Ejecuta primero:  python SIGU/backend.py")
    return ruta


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(_verificar("urgencias_limpio.csv"),
                     parse_dates=["dt_ingreso", "dt_atencion"], encoding="utf-8-sig")
    df["Triage"] = df["Triage"].astype(str)
    df["Ips"] = df["Ips"].astype(str)
    return df


def cargar_llegadas() -> pd.DataFrame:
    df = pd.read_csv(_verificar("llegadas_ips_hora.csv"), parse_dates=["hora"], encoding="utf-8-sig")
    df["Ips"] = df["Ips"].astype(str)
    return df


def cargar_calidad() -> pd.DataFrame:
    return pd.read_csv(_verificar("tabla_calidad.csv"), encoding="utf-8-sig")


def filtrar(df: pd.DataFrame, triages=None, ips=None) -> pd.DataFrame:
    if triages:
        df = df[df["Triage"].isin(triages)]
    if ips and ips != "Todas":
        df = df[df["Ips"] == ips]
    return df


# ___________ Descriptiva ___________
def descriptivos(df: pd.DataFrame, col: str = "tiempo_espera_min") -> pd.DataFrame:
    """Estadísticos de la espera por triage + fila Total."""
    def _resumen(s):
        return pd.Series({
            "n": s.count(), "media": s.mean(), "mediana": s.median(), "de": s.std(),
            "mín": s.min(), "Q1": s.quantile(.25), "Q3": s.quantile(.75),
            "máx": s.max(), "asimetría": s.skew(),
        })
    tabla = df.groupby("Triage")[col].apply(_resumen).unstack()
    tabla = tabla.reindex([t for t in TRIAGE_ORDEN if t in tabla.index])
    tabla.loc["Total"] = _resumen(df[col])
    tabla["n"] = tabla["n"].astype(int)
    return tabla.round(2)


def ic_media(serie: pd.Series, confianza: float = 0.95) -> dict:
    s = serie.dropna()
    n, media = len(s), s.mean()
    if n < 2:
        return {"n": n, "media": media, "li": np.nan, "ls": np.nan}
    em = stats.t.ppf((1 + confianza) / 2, n - 1) * s.std(ddof=1) / np.sqrt(n)
    return {"n": n, "media": media, "li": media - em, "ls": media + em}


def ic_por_triage(df: pd.DataFrame, confianza: float = 0.95) -> pd.DataFrame:
    filas = []
    for t in [t for t in TRIAGE_ORDEN if t in df["Triage"].unique()]:
        filas.append({"Triage": t, **ic_media(df.loc[df["Triage"] == t, "tiempo_espera_min"], confianza)})
    filas.append({"Triage": "Total", **ic_media(df["tiempo_espera_min"], confianza)})
    return pd.DataFrame(filas).round(2)


# ___________ Pruebas de hipótesis ___________
def _decision(p: float, alpha: float) -> str:
    return "Se rechaza H₀" if p < alpha else "No se rechaza H₀"


def formato_p(p: float) -> str:
    return "p < 0,001" if p < 0.001 else f"p = {p:.4f}"


def prueba_normalidad(serie: pd.Series, alpha: float = 0.05, n_muestra: int = 5000):
    s = serie.dropna()
    if len(s) < 3:
        return None
    m = s.sample(min(n_muestra, len(s)), random_state=SEMILLA)
    w, p = stats.shapiro(m)
    return {
        "h0": "El tiempo de espera sigue una distribución normal.",
        "h1": "El tiempo de espera no sigue una distribución normal.",
        "estadistico": w, "p": p, "alpha": alpha, "n": len(m),
        "decision": _decision(p, alpha),
    }


def chi2_uniforme_dias(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """χ² de bondad de ajuste: ¿atenciones uniformes entre los 7 días?"""
    obs = df["dia_semana"].value_counts().reindex(DIAS_ORDEN, fill_value=0)
    esp = pd.Series(obs.sum() / 7, index=DIAS_ORDEN)
    chi2, p = stats.chisquare(obs.values, esp.values)
    tabla = pd.DataFrame({"Observado": obs, "Esperado": esp.round(1)})
    tabla["(O-E)²/E"] = ((obs - esp) ** 2 / esp).round(2)
    return {
        "h0": "Las atenciones se distribuyen de forma uniforme entre los 7 días (1/7 cada día).",
        "h1": "Las atenciones no se distribuyen de forma uniforme entre los días.",
        "estadistico": chi2, "gl": 6, "p": p, "alpha": alpha,
        "decision": _decision(p, alpha), "tabla": tabla,
    }


# ___________ Poisson ___________
def prob_poisson(lam: float, k: int) -> dict:
    return {
        "P(X = k)": stats.poisson.pmf(k, lam),
        "P(X ≤ k)": stats.poisson.cdf(k, lam),
        "P(X ≥ k)": stats.poisson.sf(k - 1, lam),
    }


def chi2_poisson(llegadas: pd.DataFrame, alpha: float = 0.05):
    """χ² de bondad de ajuste a Poisson(λ), agrupando celdas con E < 5.
    Devuelve None si no hay datos suficientes (p. ej. IPS con muy pocos registros)."""
    x = llegadas["llegadas"].dropna().astype(int)
    n = len(x)
    if n == 0 or x.mean() == 0:
        return None
    lam = x.mean()
    ks = np.arange(0, int(x.max()) + 1)
    e = n * stats.poisson.pmf(ks, lam)
    validos = np.where(e >= 5)[0]
    # Se necesitan al menos 3 categorías para tener gl = categorías - 2 >= 1
    if len(validos) < 2 or (validos.max() - validos.min() + 1) < 3:
        return None
    k_ini, k_fin = int(validos.min()), int(validos.max())

    etiquetas = [f"≤ {k_ini}" if k_ini > 0 else "0"]
    obs = [(x <= k_ini).sum()]
    esp = [n * stats.poisson.cdf(k_ini, lam)]
    for k in range(k_ini + 1, k_fin):
        etiquetas.append(str(k))
        obs.append((x == k).sum())
        esp.append(n * stats.poisson.pmf(k, lam))
    etiquetas.append(f"≥ {k_fin}")
    obs.append((x >= k_fin).sum())
    esp.append(n * stats.poisson.sf(k_fin - 1, lam))

    obs, esp = np.array(obs, float), np.array(esp, float)
    chi2, p = stats.chisquare(obs, esp, ddof=1)        # ddof=1 porque se estimó λ
    tabla = pd.DataFrame({"k (llegadas/hora)": etiquetas,
                          "Observado": obs.astype(int), "Esperado": esp.round(1)})
    tabla["(O-E)²/E"] = ((obs - esp) ** 2 / esp).round(2)
    return {
        "h0": f"El número de llegadas por IPS-hora sigue una distribución Poisson (λ = {lam:.3f}).",
        "h1": "El número de llegadas por IPS-hora no sigue una distribución Poisson.",
        "estadistico": chi2, "gl": len(obs) - 2, "p": p, "alpha": alpha,
        "decision": _decision(p, alpha), "tabla": tabla,
        "lambda": lam, "varianza": x.var(), "n": n,
    }


# ___________ Regresión de Poisson (modelo de ML) ___________
VARIABLES_MODELO = ["dia_semana", "hora_dia", "mes", "Ips"]


def _features(llegadas: pd.DataFrame) -> pd.DataFrame:
    """Variables explicativas a partir de la fecha-hora de cada bloque IPS-hora."""
    return pd.DataFrame({
        "dia_semana": llegadas["hora"].dt.day_name().map(MAPA_DIAS),
        "hora_dia": llegadas["hora"].dt.hour,
        "mes": llegadas["hora"].dt.month,
        "Ips": llegadas["Ips"].astype(str),
    }, index=llegadas.index)


def modelo_regresion_poisson(llegadas: pd.DataFrame, prop_prueba: float = 0.2,
                             alpha_reg: float = 1e-4):
    """Entrena una regresión de Poisson (GLM, enlace log):
        log(λ) = β0 + β·[día semana] + β·[hora] + β·[mes] + β·[IPS]
    División temporal: el último `prop_prueba` del periodo es el conjunto de prueba
    (simula predecir semanas futuras). Compara contra la línea base
    (promedio histórico por día×hora del entrenamiento).
    Devuelve None si no hay datos suficientes."""
    d = llegadas.dropna(subset=["llegadas"]).sort_values("hora").reset_index(drop=True)
    if len(d) < 200 or d["llegadas"].sum() == 0:
        return None

    corte = d["hora"].min() + (d["hora"].max() - d["hora"].min()) * (1 - prop_prueba)
    ent, pru = d[d["hora"] <= corte], d[d["hora"] > corte]
    if len(pru) < 24 or len(ent) < 100:
        return None

    X_ent, X_pru = _features(ent), _features(pru)
    y_ent, y_pru = ent["llegadas"].values, pru["llegadas"].values

    # Todas las variables como categóricas (one-hot); categorías nuevas se ignoran
    modelo = Pipeline([
        ("onehot", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), VARIABLES_MODELO)])),
        ("poisson", PoissonRegressor(alpha=alpha_reg, max_iter=1000)),
    ])
    modelo.fit(X_ent, y_ent)
    pred_ent = np.clip(modelo.predict(X_ent), 1e-9, None)
    pred_pru = np.clip(modelo.predict(X_pru), 1e-9, None)

    # Línea base: promedio de entrenamiento por (IPS, día, hora)
    base_tab = (X_ent.assign(y=y_ent)
                .groupby(["Ips", "dia_semana", "hora_dia"])["y"].mean())
    base_pru = (X_pru.join(base_tab, on=["Ips", "dia_semana", "hora_dia"])["y"]
                .fillna(y_ent.mean()).clip(lower=1e-9).values)

    # Parámetro de dispersión φ (Pearson) en entrenamiento: ≈1 Poisson OK, >1 sobredispersión
    n_param = modelo.named_steps["onehot"].transform(X_ent).shape[1] + 1
    gl = max(len(y_ent) - n_param, 1)
    phi = float(np.sum((y_ent - pred_ent) ** 2 / pred_ent) / gl)

    def _metricas(y, p):
        return {"MAE": mean_absolute_error(y, p),
                "RMSE": float(np.sqrt(mean_squared_error(y, p))),
                "Devianza Poisson media": mean_poisson_deviance(y, p)}

    m_modelo, m_base = _metricas(y_pru, pred_pru), _metricas(y_pru, base_pru)
    # D² = % de devianza explicada frente a predecir siempre la media (pseudo-R²)
    dev_nula = mean_poisson_deviance(y_pru, np.full_like(pred_pru, y_ent.mean(), dtype=float))
    m_modelo["D² (devianza explicada)"] = 1 - m_modelo["Devianza Poisson media"] / dev_nula
    m_base["D² (devianza explicada)"] = 1 - m_base["Devianza Poisson media"] / dev_nula

    metricas = pd.DataFrame({"Regresión Poisson": m_modelo, "Línea base (promedio)": m_base})

    # Datos para gráficas
    res = pru[["hora"]].assign(observado=y_pru, modelo=pred_pru, base=base_pru)
    por_dia = res.set_index("hora").resample("D")[["observado", "modelo", "base"]].sum()
    por_hora = (res.assign(hora_dia=res["hora"].dt.hour)
                   .groupby("hora_dia")[["observado", "modelo", "base"]].mean())

    return {
        "modelo": modelo, "metricas": metricas.round(4), "phi": phi,
        "n_entrenamiento": len(ent), "n_prueba": len(pru), "n_parametros": n_param,
        "corte": corte, "fin": d["hora"].max(),
        "por_dia": por_dia, "por_hora": por_hora,
        "residuos": y_pru - pred_pru, "pred_pru": pred_pru,
    }


def lambda_modelo_semana(resultado: dict, llegadas: pd.DataFrame) -> pd.DataFrame:
    """λ predicho por la regresión para cada día×hora (sumando IPS), usando el último mes
    observado. Sirve como alternativa a `demanda_por_hora` para la malla."""
    mes = int(llegadas["hora"].max().month)
    filas = [{"dia_semana": d, "hora_dia": h, "mes": mes, "Ips": i}
             for i in llegadas["Ips"].astype(str).unique() for d in DIAS_ORDEN for h in range(24)]
    X = pd.DataFrame(filas)
    X["lam"] = resultado["modelo"].predict(X)
    return (X.groupby(["dia_semana", "hora_dia"])["lam"].sum()
             .unstack("hora_dia").reindex(index=DIAS_ORDEN, columns=range(24)).fillna(0))


# ___________ Malla de turnos ___________
TURNOS = {  # nombre: horas que cubre (turnos de 6 h, iguales a las franjas horarias)
    "Madrugada (00-06)": range(0, 6),
    "Mañana (06-12)": range(6, 12),
    "Tarde (12-18)": range(12, 18),
    "Noche (18-24)": range(18, 24),
}


def demanda_por_hora(llegadas: pd.DataFrame) -> pd.DataFrame:
    """Predicción de pacientes por día de semana y hora: λ = promedio histórico
    de llegadas en esa franja (incluye horas con 0). Si hay varias IPS, se suman
    los λ de cada IPS (la suma de Poisson independientes también es Poisson)."""
    t = llegadas.assign(dia_semana=llegadas["hora"].dt.day_name().map(MAPA_DIAS),
                        hora_dia=llegadas["hora"].dt.hour)
    lam = (t.groupby(["Ips", "dia_semana", "hora_dia"])["llegadas"].mean()
             .groupby(["dia_semana", "hora_dia"]).sum())
    tabla = lam.unstack("hora_dia").reindex(index=DIAS_ORDEN, columns=range(24))
    return tabla.fillna(0)


def medicos_por_hora(lam: pd.DataFrame, pacientes_medico_hora: float,
                     nivel_servicio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pacientes a cubrir = percentil `nivel_servicio` de la Poisson(λ)
    (así se cubre la demanda en ese % de las horas, no solo el promedio).
    Médicos = techo(pacientes a cubrir / capacidad por médico), mínimo 1."""
    pacientes = pd.DataFrame(stats.poisson.ppf(nivel_servicio, lam.values),
                             index=lam.index, columns=lam.columns).fillna(0)
    medicos = np.ceil(pacientes / pacientes_medico_hora).clip(lower=1).astype(int)
    return pacientes.astype(int), medicos


def malla_turnos(medicos_hora: pd.DataFrame) -> pd.DataFrame:
    """Médicos por día y turno = máximo requerido dentro de las horas del turno
    (cubre la hora pico del turno)."""
    malla = pd.DataFrame({turno: medicos_hora[list(horas)].max(axis=1)
                          for turno, horas in TURNOS.items()})
    malla["Horas-médico/día"] = (malla[list(TURNOS)] * 6).sum(axis=1)
    return malla


# =====================================================================
# Solo se ejecuta con:  python SIGU/backend.py
# =====================================================================
if __name__ == "__main__":
    generar_archivos_limpios()
    df = cargar_datos()
    assert (df["tiempo_espera_min"] >= 0).all()
    assert df["dia_semana"].isin(DIAS_ORDEN).all()
    print("OK – dataset listo para la app:", df.shape)
