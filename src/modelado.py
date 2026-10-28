"""Modelos, validacion hacia adelante y metricas del notebook 03.

Validacion walk-forward: para evaluar el anio T se entrena con todos los titulos de la ventana
anteriores a T y se predice T. Imita el uso real (predecir lanzamientos futuros con lo ya
lanzado) y evita las dos fugas del K-fold aleatorio: mezclar epocas y separar a los "hermanos"
de un mismo titulo (el mismo juego en varias plataformas) entre entrenamiento y prueba.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OBJETIVO = "exito"
ANIO_INICIO_ENTRENAMIENTO = 1995
SEMILLA = 0

PARAMS_HGB = dict(max_depth=5, learning_rate=0.03, max_iter=150, min_samples_leaf=60,
                  l2_regularization=1.0, random_state=SEMILLA)


# ------------------------------------------------------------------------------ modelos
# Todas las funciones comparten la firma  f(entreno, prueba, cat, num) -> probabilidades.

def ajustar_hgb(entreno, cat, num, **params):
    """Ajusta un gradient boosting con histogramas. Devuelve (modelo, preparar).

    Las categorias se fijan con las del entrenamiento: un valor nuevo en prueba pasa a nulo en
    lugar de cambiar la codificacion.
    """
    categorias = {c: sorted(entreno[c].dropna().unique()) for c in cat}

    def preparar(d):
        X = d[cat + num].copy()
        for c in cat:
            X[c] = pd.Categorical(X[c], categories=categorias[c])
        return X

    modelo = HistGradientBoostingClassifier(categorical_features="from_dtype", **{**PARAMS_HGB, **params})
    modelo.fit(preparar(entreno), entreno[OBJETIVO])
    return modelo, preparar


def hgb(entreno, prueba, cat, num, **params):
    modelo, preparar = ajustar_hgb(entreno, cat, num, **params)
    return modelo.predict_proba(preparar(prueba))[:, 1]


def _matriz_lineal(d, cat, num):
    """Rellena nulos con 0 y agrega un indicador `_na` por cada numerica con nulos."""
    out = d[cat + num].copy()
    for c in num:
        if out[c].isna().any():
            out[c + "_na"] = out[c].isna().astype(int)
            out[c] = out[c].fillna(0)
    return out


def logistica(entreno, prueba, cat, num, C=0.3):
    a, b = _matriz_lineal(entreno, cat, num), _matriz_lineal(prueba, cat, num)
    b = b.reindex(columns=a.columns, fill_value=0)
    num_cols = [c for c in a.columns if c not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
                             ("num", StandardScaler(), num_cols)])
    modelo = make_pipeline(pre, LogisticRegression(C=C, max_iter=3000))
    modelo.fit(a, entreno[OBJETIVO])
    return modelo.predict_proba(b)[:, 1]


def bosque(entreno, prueba, cat, num, **params):
    a, b = entreno[cat + num].fillna(-1), prueba[cat + num].fillna(-1)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat)],
                            remainder="passthrough")
    modelo = make_pipeline(pre, RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                       n_jobs=-1, random_state=SEMILLA, **params))
    modelo.fit(a, entreno[OBJETIVO])
    return modelo.predict_proba(b)[:, 1]


def azar(entreno, prueba, cat=None, num=None):
    """Baseline sin informacion: la tasa de exito del entrenamiento para todos."""
    return np.full(len(prueba), entreno[OBJETIVO].mean())


def regla_del_analista(entreno, prueba, cat=None, num=None):
    """Baseline de negocio: 'apuesta por el publisher con mejor historial de exitos'.

    Puntaje = proporcion de exitos previos del publisher (sin historial: tasa global).
    """
    return prueba["pub_exito_prev"].fillna(entreno[OBJETIVO].mean()).to_numpy()


# ------------------------------------------------------------------------------ validacion

def _resultado(prueba, p):
    return pd.DataFrame({"anio": prueba["anio"].to_numpy(), "y": prueba[OBJETIVO].to_numpy(), "p": p,
                         "nombre": prueba["nombre"].to_numpy(), "plataforma": prueba["plataforma"].to_numpy()})


def recorrer_anios(base, modelo, cat, num, anios, desde=ANIO_INICIO_ENTRENAMIENTO, **params):
    """Walk-forward: entrena con [desde, T) y predice T para cada T en `anios`."""
    piezas = []
    for t in anios:
        entreno = base[(base["anio"] >= desde) & (base["anio"] < t)]
        prueba = base[base["anio"] == t]
        piezas.append(_resultado(prueba, modelo(entreno, prueba, cat, num, **params)))
    return pd.concat(piezas, ignore_index=True)


def ajustar_y_predecir(base, modelo, cat, num, anio_corte, anios_prueba, desde=ANIO_INICIO_ENTRENAMIENTO, **params):
    """Un solo ajuste con [desde, anio_corte] y prediccion de `anios_prueba` (prueba final)."""
    entreno = base[(base["anio"] >= desde) & (base["anio"] <= anio_corte)]
    prueba = base[base["anio"].isin(anios_prueba)]
    return _resultado(prueba, modelo(entreno, prueba, cat, num, **params))


def validacion_cruzada(base, modelo, cat, num, grupos=None, k=5, semilla=SEMILLA, **params):
    """K-fold aleatorio (grupos=None) o agrupado por `grupos` (GroupKFold). Probabilidades fuera de muestra.

    Es el protocolo 'ingenuo' que el notebook 03 compara con el walk-forward.
    """
    base = base.reset_index(drop=True)
    divisor = KFold(k, shuffle=True, random_state=semilla) if grupos is None else GroupKFold(k)
    p = np.zeros(len(base))
    for tr, te in divisor.split(base, groups=None if grupos is None else base[grupos]):
        p[te] = modelo(base.iloc[tr], base.iloc[te], cat, num, **params)
    return _resultado(base, p)


# ------------------------------------------------------------------------------ metricas

def _orden_por_puntaje(p, semilla=SEMILLA):
    """Indices de mayor a menor puntaje; los empates se resuelven al AZAR (semilla fija).

    Sin esto, un puntaje con muchos empates (el azar, la regla del analista) se desempata por el
    orden de las filas, que en este dataset viene ordenado por ventas: regalaria aciertos.
    """
    p = np.asarray(p, dtype=float)
    desempate = np.random.default_rng(semilla).permutation(len(p))
    return np.lexsort((desempate, -p))          # clave principal: -p; secundaria: orden aleatorio


def precision_en_top(y, p, fraccion=0.10):
    """Proporcion de exitos entre el `fraccion` de titulos con mayor puntaje.

    Los empates en el corte se tratan por su valor ESPERADO (como si se desempataran al azar):
    los titulos por encima del corte entran enteros y los empatados en el corte entran en la
    proporcion que cabe. Asi un puntaje constante da exactamente la tasa base, sin ruido.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    k = max(1, int(round(fraccion * len(y))))
    corte = np.sort(p)[::-1][k - 1]                 # puntaje del k-esimo titulo
    por_encima, empatados = p > corte, p == corte
    plazas = k - int(por_encima.sum())              # plazas que quedan para los empatados en el corte
    return float((y[por_encima].sum() + y[empatados].mean() * plazas) / k)


def metricas(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    top10 = precision_en_top(y, p, 0.10)
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, pc),
        "brier": brier_score_loss(y, pc),
        "tasa_base": float(y.mean()),
        "precision_top10": top10,
        "lift_top10": top10 / float(y.mean()),
    }


def resumen_por_anio(res):
    """Metricas de cada anio evaluado (un fold por anio) y su promedio +/- desviacion."""
    filas = {int(a): metricas(g["y"], g["p"]) for a, g in res.groupby("anio")}
    por_anio = pd.DataFrame(filas).T
    por_anio.index.name = "anio"
    return por_anio


def promedio_de_folds(res, columnas=("auc", "pr_auc", "log_loss", "lift_top10")):
    por_anio = resumen_por_anio(res)[list(columnas)]
    return por_anio.mean(), por_anio.std(ddof=1)


def intervalo_bootstrap(res, funcion, grupos="nombre", repeticiones=1000, semilla=SEMILLA):
    """IC 95% por bootstrap de titulos: remuestrea NOMBRES, no filas.

    Las fichas de un mismo juego en varias plataformas se parecen mucho; remuestrear filas
    sueltas subestimaria la incertidumbre.
    """
    rng = np.random.default_rng(semilla)
    indices = res.groupby(grupos).indices
    claves = list(indices)
    y, p = res["y"].to_numpy(), res["p"].to_numpy()
    valores = []
    for _ in range(repeticiones):
        elegidas = rng.integers(0, len(claves), size=len(claves))
        filas = np.concatenate([indices[claves[i]] for i in elegidas])
        valores.append(funcion(y[filas], p[filas]))
    return np.percentile(valores, [2.5, 97.5])


def diferencia_bootstrap(res_a, res_b, funcion, grupos="nombre", repeticiones=1000, semilla=SEMILLA):
    """IC 95% de funcion(a) - funcion(b) sobre los MISMOS titulos (bootstrap pareado)."""
    assert (res_a["nombre"].to_numpy() == res_b["nombre"].to_numpy()).all(), "los resultados deben cubrir las mismas filas"
    rng = np.random.default_rng(semilla)
    indices = res_a.groupby(grupos).indices
    claves = list(indices)
    y, pa, pb = res_a["y"].to_numpy(), res_a["p"].to_numpy(), res_b["p"].to_numpy()
    dif = []
    for _ in range(repeticiones):
        elegidas = rng.integers(0, len(claves), size=len(claves))
        filas = np.concatenate([indices[claves[i]] for i in elegidas])
        dif.append(funcion(y[filas], pa[filas]) - funcion(y[filas], pb[filas]))
    return float(np.mean(dif)), np.percentile(dif, [2.5, 97.5])


def curva_de_ganancia(y, p, ventas=None):
    """% acumulado de exitos (y de ventas, si se pasan) capturado al ordenar por puntaje."""
    orden = _orden_por_puntaje(p)
    y = np.asarray(y)[orden]
    x = np.arange(1, len(y) + 1) / len(y)
    ganancia = np.cumsum(y) / y.sum()
    if ventas is None:
        return x, ganancia
    v = np.asarray(ventas)[orden]
    return x, ganancia, np.cumsum(v) / v.sum()


def tabla_calibracion(y, p, cortes=10):
    """Promedio predicho vs frecuencia observada por tramos de probabilidad (deciles)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["tramo"] = pd.qcut(d["p"].rank(method="first"), cortes, labels=False)
    return d.groupby("tramo").agg(n=("y", "size"), predicho=("p", "mean"), observado=("y", "mean"))


def importancia_por_grupo(entreno, prueba, cat, num, grupos, repeticiones=20, **params):
    """Importancia por permutacion AGRUPADA: cuanto cae el AUC al barajar todas las columnas
    de un grupo a la vez (p. ej. las 4 variables de historial de franquicia).

    Barajar columnas sueltas subestima a las que estan correlacionadas entre si; barajar el
    grupo completo mide la informacion que aporta el bloque.
    """
    modelo, preparar = ajustar_hgb(entreno, cat, num, **params)
    X, y = preparar(prueba), prueba[OBJETIVO].to_numpy()
    auc_base = roc_auc_score(y, modelo.predict_proba(X)[:, 1])
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for nombre, columnas in grupos.items():
        caidas = []
        for _ in range(repeticiones):
            perm = rng.permutation(len(X))
            X2 = X.copy()
            for c in columnas:
                X2[c] = X[c].iloc[perm].values
            caidas.append(auc_base - roc_auc_score(y, modelo.predict_proba(X2)[:, 1]))
        filas.append({"grupo": nombre, "caida_auc": np.mean(caidas), "desv": np.std(caidas, ddof=1)})
    return auc_base, pd.DataFrame(filas).sort_values("caida_auc", ascending=False).reset_index(drop=True)
