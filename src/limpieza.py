"""Limpieza de vgsales.csv (ventas de videojuegos, 1980-2016).

Cada paso recibe un DataFrame y devuelve (df, info). `info` es un dict con los conteos y
ejemplos que el notebook 01 muestra, asi cada decision queda respaldada con numeros y no
solo con prosa.

La limpieza es NO destructiva: salvo 2 duplicados, ninguna fila se elimina. Los problemas
se marcan con banderas (`anio_invalido`, `registro_parcial`, `publisher_desconocido`) y cada
analisis decide que filas usar.

Uso como script:    python src/limpieza.py
Uso desde notebook: from limpieza import limpiar
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "vgsales.csv"
PROCESSED = RAIZ / "data" / "processed" / "vgsales_clean.csv"

# ------------------------------------------------------------------ parametros del proyecto

ANIO_MAX_OBSERVACION = 2016          # el dataset no tiene datos fiables posteriores a 2016
VENTANA_INICIO = 1995                # ventana de analisis y modelado (ver notebook 01, seccion 3)
VENTANA_FIN = 2014
UMBRAL_EXITO = 1.0                   # "exito" = al menos 1 millon de copias vendidas en el mundo

COLUMNAS = {
    "Rank": "rank_original", "Name": "nombre", "Platform": "plataforma", "Year": "anio",
    "Genre": "genero", "Publisher": "publisher", "NA_Sales": "ventas_na", "EU_Sales": "ventas_eu",
    "JP_Sales": "ventas_jp", "Other_Sales": "ventas_otros", "Global_Sales": "ventas_global",
}
VENTAS_REGIONALES = ["ventas_na", "ventas_eu", "ventas_jp", "ventas_otros"]

# Primer anio de lanzamiento mundial de cada consola (conocimiento de dominio). Solo sirve para
# detectar fechas imposibles: un juego no puede salir en una consola que aun no existe.
# PC no tiene una fecha unica de lanzamiento: 1970 deja la regla sin efecto.
LANZAMIENTO_PLATAFORMA = {
    "2600": 1977, "NES": 1983, "TG16": 1987, "GEN": 1988, "GB": 1989, "SNES": 1990, "GG": 1990,
    "NG": 1990, "SCD": 1991, "3DO": 1993, "SAT": 1994, "PS": 1994, "PCFX": 1994, "N64": 1996,
    "DC": 1998, "WS": 1999, "PS2": 2000, "GBA": 2001, "GC": 2001, "XB": 2001, "DS": 2004,
    "PSP": 2004, "X360": 2005, "Wii": 2006, "PS3": 2006, "3DS": 2011, "PSV": 2011, "WiiU": 2012,
    "PS4": 2013, "XOne": 2013, "PC": 1970,
}

FABRICANTE = {
    "NES": "Nintendo", "SNES": "Nintendo", "N64": "Nintendo", "GC": "Nintendo", "Wii": "Nintendo",
    "WiiU": "Nintendo", "GB": "Nintendo", "GBA": "Nintendo", "DS": "Nintendo", "3DS": "Nintendo",
    "PS": "Sony", "PS2": "Sony", "PS3": "Sony", "PS4": "Sony", "PSP": "Sony", "PSV": "Sony",
    "XB": "Microsoft", "X360": "Microsoft", "XOne": "Microsoft",
    "SAT": "Sega", "DC": "Sega", "GEN": "Sega", "SCD": "Sega", "GG": "Sega",
    "PC": "PC",
    "2600": "Otros", "NG": "Otros", "3DO": "Otros", "TG16": "Otros", "WS": "Otros", "PCFX": "Otros",
}
PORTATILES = {"GB", "GBA", "DS", "3DS", "PSP", "PSV", "GG", "WS"}

# Patron de las fichas con la fuente de la cifra anotada en el nombre: "(JP sales)",
# "(US weekly sales)", "(All region sales)"... Son cifras parciales o estimaciones alternativas.
PATRON_PARCIAL = re.compile(r"\([^)]*\b(?:sales|weekly)\b[^)]*\)", re.IGNORECASE)

# Plataformas que a veces se agregan al final del titulo ("Mario Kart Wii")
_SUFIJOS_PLATAFORMA = {"wii", "ds", "psp", "ps2", "ps3", "ps4", "3ds", "gba", "gbc", "gc", "pc", "xbox", "vita"}
_ROMANOS = r"ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv"


# ------------------------------------------------------------------------------ utilidades

def extraer_franquicia(nombre):
    """Heuristica de franquicia: el titulo sin entregas, anios, plataformas ni anotaciones.

    "Call of Duty: Black Ops II" -> "call of duty";  "FIFA 14" -> "fifa";
    "Mario Kart Wii" -> "mario kart";  "Madden NFL 13" -> "madden nfl".

    Es imperfecta a proposito (sin diccionarios ni fuzzy matching): falla en sagas cuyo nombre
    cambia por edicion (Pokemon Red/Blue, Gold/Silver...). El notebook 03 mide cuanto cuesta.
    """
    completo = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower()
    completo = re.sub(r"\([^)]*\)", " ", completo)                  # anotaciones entre parentesis
    s = re.split(r"\s*(?::| - |/)\s*", completo)[0]                 # lo que va antes de ':', ' - ' o '/'
    s = re.sub(r"[^a-z0-9 ]+", " ", s)                              # puntuacion
    s = re.sub(rf"\b(?:\d+[a-z]*|{_ROMANOS})\b", " ", s)            # entregas, anios, 2k14, 3d, romanos
    palabras = [p for p in s.split() if p not in {"the", "soccer"}]
    while palabras and palabras[-1] in _SUFIJOS_PLATAFORMA:         # "mario kart wii" -> "mario kart"
        palabras.pop()
    if not palabras:                                                # titulos que son solo numeros ("007: ...")
        palabras = re.sub(r"[^a-z0-9 ]+", " ", completo).split()
    return " ".join(palabras)


# ------------------------------------------------------------------------------ pasos

def cargar_crudo(ruta=RAW):
    """Lee el CSV y estandariza nombres de columnas. pandas interpreta 'N/A' como nulo."""
    df = pd.read_csv(ruta).rename(columns=COLUMNAS)
    df["anio"] = df["anio"].astype("Int64")
    return df


def normalizar_publisher(df):
    """'Unknown' (203 filas) es un nulo disfrazado de categoria; ya hay 58 'N/A' reales.

    Se unifican como NaN y se deja la bandera `publisher_desconocido`.
    """
    df = df.copy()
    es_unknown = df["publisher"].eq("Unknown")
    info = {
        "n_na_original": int(df["publisher"].isna().sum()),
        "n_unknown_texto": int(es_unknown.sum()),
    }
    df.loc[es_unknown, "publisher"] = np.nan
    df["publisher_desconocido"] = df["publisher"].isna()
    info["n_total"] = int(df["publisher_desconocido"].sum())
    return df, info


def invalidar_anios_imposibles(df):
    """Anula (NaN) los anios posteriores a 2016 o anteriores al lanzamiento de la consola."""
    df = df.copy()
    lanzamiento = df["plataforma"].map(LANZAMIENTO_PLATAFORMA)
    futuro = (df["anio"] > ANIO_MAX_OBSERVACION).fillna(False).to_numpy(dtype=bool)
    previo = (df["anio"] < lanzamiento).fillna(False).to_numpy(dtype=bool)
    malo = futuro | previo
    info = {
        "n_nulos_originales": int(df["anio"].isna().sum()),
        "n_futuro": int(futuro.sum()),
        "n_previo_a_consola": int(previo.sum()),
        "filas": df.loc[malo, ["nombre", "plataforma", "anio", "ventas_global"]].copy(),
    }
    df["anio_invalido"] = malo
    df.loc[malo, "anio"] = pd.NA
    info["n_nulos_finales"] = int(df["anio"].isna().sum())
    return df, info


def quitar_duplicados(df):
    """Quita los duplicados reales y conserva los homonimos legitimos.

    1) Filas identicas salvo `rank_original` -> se conserva la primera.
    2) Misma (nombre, plataforma, anio) con ventas distintas -> ficha fantasma de ventas
       residuales; se conserva la de mayor `ventas_global`.

    NO son duplicados: Need for Speed: Most Wanted (2005 y 2012 son juegos distintos) ni
    Sonic the Hedgehog en PS3 (2006 vs. sin anio; ambiguo, se conserva y se documenta).
    """
    columnas = [c for c in df.columns if c != "rank_original"]
    identica = df.duplicated(subset=columnas, keep="first")
    sin_identicas = df.loc[~identica]

    orden = sin_identicas.sort_values("ventas_global", ascending=False, kind="stable")
    fantasma_idx = orden.index[orden.duplicated(subset=["nombre", "plataforma", "anio"], keep="first")]
    info = {
        "n_identicas": int(identica.sum()),
        "n_fantasma": int(len(fantasma_idx)),
        "eliminadas": pd.concat([df.loc[identica], sin_identicas.loc[fantasma_idx]])[
            ["rank_original", "nombre", "plataforma", "anio", "ventas_global"]
        ],
        "filas_antes": len(df),
    }
    limpio = sin_identicas.drop(index=fantasma_idx).sort_values("rank_original").reset_index(drop=True)
    info["filas_despues"] = len(limpio)
    return limpio, info


def marcar_registros_parciales(df):
    """Marca las fichas cuyo nombre anota la fuente de la cifra: "(JP sales)", "(weekly)"...

    Su `ventas_global` suele subestimar el total del titulo, y en 82 casos el mismo juego ya
    existe como ficha normal en la misma plataforma (doble conteo). No se borran: se excluyen
    de los analisis por titulo con la bandera.
    """
    df = df.copy()
    parcial = df["nombre"].str.contains(PATRON_PARCIAL, na=False)
    base = df["nombre"].str.replace(r"\s*\([^)]*\)\s*", "", regex=True).str.strip()
    claves_normales = set(zip(base[~parcial], df.loc[~parcial, "plataforma"]))
    con_base = np.array([(b, p) in claves_normales for b, p in zip(base[parcial], df.loc[parcial, "plataforma"])])
    info = {
        "n_parciales": int(parcial.sum()),
        "n_con_ficha_base": int(con_base.sum()),
        "n_sin_ficha_base": int((~con_base).sum()),
        "ventas_global_parciales": float(df.loc[parcial, "ventas_global"].sum()),
        "ventas_global_total": float(df["ventas_global"].sum()),
        "ejemplos": df.loc[parcial, ["nombre", "plataforma", "anio", "ventas_global"]].head(8).copy(),
    }
    df["registro_parcial"] = parcial
    return df, info


def agregar_derivadas(df):
    """Columnas derivadas que usan el EDA y el modelado."""
    df = df.copy()
    lanzamiento = df["plataforma"].map(LANZAMIENTO_PLATAFORMA)
    edad = (df["anio"].astype("Float64") - lanzamiento).astype("Float64")
    df["fabricante"] = df["plataforma"].map(FABRICANTE)
    df["tipo_plataforma"] = np.where(df["plataforma"].isin(PORTATILES), "Portatil",
                                     np.where(df["plataforma"].eq("PC"), "PC", "Sobremesa"))
    df["edad_plataforma"] = edad.mask(df["plataforma"].eq("PC")).astype("Float64")   # PC no tiene "edad"
    df["franquicia"] = df["nombre"].map(extraer_franquicia)
    df["exito"] = (df["ventas_global"] >= UMBRAL_EXITO).astype(int)
    df["en_ventana"] = df["anio"].between(VENTANA_INICIO, VENTANA_FIN).fillna(False).astype(bool)
    return df


ORDEN_COLUMNAS = [
    "rank_original", "nombre", "plataforma", "anio", "genero", "publisher",
    "ventas_na", "ventas_eu", "ventas_jp", "ventas_otros", "ventas_global",
    "fabricante", "tipo_plataforma", "edad_plataforma", "franquicia", "exito", "en_ventana",
    "publisher_desconocido", "anio_invalido", "registro_parcial",
]


def limpiar(ruta=RAW):
    """Pipeline completo. Devuelve (df_limpio, registro) con el `info` de cada paso."""
    df = cargar_crudo(ruta)
    registro = {"filas_crudas": len(df)}
    df, registro["publisher"] = normalizar_publisher(df)
    df, registro["anios"] = invalidar_anios_imposibles(df)
    df, registro["duplicados"] = quitar_duplicados(df)
    df, registro["parciales"] = marcar_registros_parciales(df)
    df = agregar_derivadas(df)
    return df[ORDEN_COLUMNAS], registro


# ------------------------------------------------------------------------------ validacion

def validar(df):
    """Verifica los invariantes del dataset limpio. Devuelve la lista de (descripcion, ok)."""
    suma_regional = df[VENTAS_REGIONALES].sum(axis=1)
    lanzamiento = df["plataforma"].map(LANZAMIENTO_PLATAFORMA)
    sin_rank = [c for c in df.columns if c != "rank_original"]
    clave = df.dropna(subset=["anio"]).duplicated(["nombre", "plataforma", "anio"]).sum()
    pruebas = [
        ("no quedan filas identicas (ignorando rank)", int(df.duplicated(subset=sin_rank).sum()) == 0),
        ("no quedan (nombre, plataforma, anio) repetidos", int(clave) == 0),
        ("rank_original es unico", df["rank_original"].is_unique),
        ("ninguna venta negativa", bool((df[VENTAS_REGIONALES + ["ventas_global"]] >= 0).all().all())),
        ("ventas_global >= 0.01 en todas las filas", bool((df["ventas_global"] >= 0.01 - 1e-9).all())),
        ("ventas_global = suma regional +/- 0.02 (redondeo)", bool(((df["ventas_global"] - suma_regional).abs() <= 0.0201).all())),
        ("ventas_global >= cada venta regional", bool((df[VENTAS_REGIONALES].max(axis=1) <= df["ventas_global"] + 1e-9).all())),
        ("anio dentro de 1980-2016 o nulo", bool(df["anio"].dropna().between(1980, ANIO_MAX_OBSERVACION).all())),
        ("ningun anio anterior al lanzamiento de la consola", not bool((df["anio"] < lanzamiento).fillna(False).any())),
        ("'Unknown' ya no existe como publisher", not bool(df["publisher"].eq("Unknown").any())),
        ("publisher_desconocido coincide con publisher nulo", bool((df["publisher_desconocido"] == df["publisher"].isna()).all())),
        ("12 generos y 31 plataformas, sin nulos", df["genero"].nunique() == 12 and df["plataforma"].nunique() == 31
         and not df[["genero", "plataforma", "nombre"]].isna().any().any()),
        ("toda plataforma tiene fabricante", not df["fabricante"].isna().any()),
        ("exito == (ventas_global >= 1)", bool((df["exito"] == (df["ventas_global"] >= UMBRAL_EXITO)).all())),
        ("las 3 banderas son booleanas", all(df[c].dtype == bool for c in ["publisher_desconocido", "anio_invalido", "registro_parcial"])),
        ("en_ventana implica anio conocido en 1995-2014", bool(df.loc[df["en_ventana"], "anio"].between(VENTANA_INICIO, VENTANA_FIN).all())),
        ("edad_plataforma >= 0 cuando existe", bool((df["edad_plataforma"].dropna() >= 0).all())),
        ("franquicia no vacia salvo casos extremos (<0.5%)", float((df["franquicia"] == "").mean()) < 0.005),
    ]
    return pruebas


def main():
    df, registro = limpiar()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    fallos = [d for d, ok in validar(df) if not ok]
    print(f"{len(df):,} filas x {df.shape[1]} columnas -> {PROCESSED}")
    print("verificaciones fallidas:", fallos if fallos else "ninguna")


if __name__ == "__main__":
    main()
