"""Variables del modelo de exito comercial (notebook 03).

Todas las variables estan disponibles ANTES de conocer las ventas del propio titulo:

  - de la ficha:       plataforma, genero, publisher (agrupado), edad de la plataforma,
                       numero de versiones lanzadas el mismo anio
  - de la franquicia:  historial de ventas de entregas anteriores (heuristica de nombre)
  - del publisher:     historial de ventas de sus lanzamientos anteriores

Los historiales son lo delicado. Para un titulo del anio Y solo cuentan titulos de anios
<= Y - 1 - rezago. Con rezago = 0 se usarian las ventas FINALES (acumuladas hasta 2016) de lo
lanzado en Y-1, que en el momento del lanzamiento todavia no se conocian: es una mirada al
futuro. `rezago = 1` exige que la entrega anterior haya tenido al menos un anio completo para
madurar. El notebook 03 mide cuanto cambia el resultado entre ambos.
"""

import numpy as np
import pandas as pd

MIN_TITULOS_PUBLISHER = 40     # publishers con menos titulos se agrupan en "Otros"
REZAGO_MADUREZ = 1             # anios de madurez exigidos a los historiales

CAT = ["plataforma", "genero", "publisher_grp"]
NUM_FICHA = ["edad_plataforma", "n_versiones"]
NUM_FRANQUICIA = ["fr_n_prev", "fr_media_prev", "fr_max_prev", "fr_exito_prev"]
NUM_PUBLISHER = ["pub_n_prev", "pub_media_prev", "pub_exito_prev"]

# bloques acumulativos para la ablacion: cada uno agrega informacion al anterior
BLOQUES = {
    "A. Plataforma y genero": (["plataforma", "genero"], []),
    "B. + Publisher": (CAT, []),
    "C. + Edad de plataforma y versiones": (CAT, NUM_FICHA),
    "D. + Historial de franquicia": (CAT, NUM_FICHA + NUM_FRANQUICIA),
    "E. + Historial del publisher": (CAT, NUM_FICHA + NUM_FRANQUICIA + NUM_PUBLISHER),
}

# Regla (fijada antes de ver la prueba final): un bloque se conserva si mejora el AUC medio de los
# folds de seleccion en al menos esta cantidad. Con ella el modelo final es el bloque D.
MEJORA_MINIMA_AUC = 0.005


def preparar_base(df, min_titulos=MIN_TITULOS_PUBLISHER):
    """Titulos analizables (sin registros parciales ni anio desconocido) y variables de la ficha."""
    b = df.loc[~df["registro_parcial"] & df["anio"].notna()].copy()
    b["anio"] = b["anio"].astype(int)
    b["edad_plataforma"] = b["edad_plataforma"].astype(float)
    b["n_versiones"] = b.groupby(["nombre", "anio"])["nombre"].transform("size").clip(upper=8)
    pub = b["publisher"].fillna("Desconocido")
    conteo = pub.value_counts()
    b["publisher_clave"] = pub
    b["publisher_grp"] = np.where(pub.map(conteo) >= min_titulos, pub, "Otros")
    b.loc[pub == "Desconocido", "publisher_grp"] = "Desconocido"
    b["ventas_log"] = np.log1p(b["ventas_global"])
    return b.reset_index(drop=True)


def historial_previo(b, clave, prefijo, rezago=REZAGO_MADUREZ):
    """Estadisticos de los titulos de la misma `clave` lanzados hasta el anio Y - 1 - rezago.

    Devuelve un DataFrame alineado con `b` y columnas:
      {prefijo}_n_prev      cuantas fichas previas hay
      {prefijo}_media_prev  media de log(1 + ventas) de las fichas previas (NaN si no hay)
      {prefijo}_max_prev    maximo de log(1 + ventas) de las fichas previas
      {prefijo}_exito_prev  proporcion de fichas previas con >= 1 M de copias
    """
    por_anio = (b.groupby([clave, "anio"])
                  .agg(n=("ventas_log", "size"), s=("ventas_log", "sum"),
                       mx=("ventas_log", "max"), h=("exito", "sum"))
                  .reset_index().sort_values([clave, "anio"]))
    for c in ("n", "s", "h"):
        por_anio[c] = por_anio.groupby(clave)[c].cumsum()
    por_anio["mx"] = por_anio.groupby(clave)["mx"].cummax()
    por_anio = por_anio.rename(columns={"anio": "anio_hist"}).sort_values("anio_hist")

    consulta = pd.DataFrame({"idx": b.index, clave: b[clave].to_numpy(),
                             "corte": (b["anio"] - 1 - rezago).to_numpy()}).sort_values("corte")
    m = pd.merge_asof(consulta, por_anio, left_on="corte", right_on="anio_hist", by=clave,
                      direction="backward").sort_values("idx")
    n = m["n"].fillna(0).to_numpy()
    seguro = np.where(n > 0, n, 1)
    out = pd.DataFrame(index=b.index)
    out[f"{prefijo}_n_prev"] = n
    out[f"{prefijo}_media_prev"] = np.where(n > 0, m["s"].to_numpy() / seguro, np.nan)
    out[f"{prefijo}_max_prev"] = np.where(n > 0, m["mx"].to_numpy(), np.nan)
    out[f"{prefijo}_exito_prev"] = np.where(n > 0, m["h"].to_numpy() / seguro, np.nan)
    return out


def agregar_historiales(b, rezago=REZAGO_MADUREZ):
    """Agrega los historiales de franquicia y de publisher a la base preparada."""
    return pd.concat([b,
                      historial_previo(b, "franquicia", "fr", rezago),
                      historial_previo(b, "publisher_clave", "pub", rezago)], axis=1)
