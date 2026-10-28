"""Tokens de color y estilo de matplotlib para las figuras del proyecto.

Paleta validada con el validador de la guia de visualizacion (superficie #fcfcfb, modo claro):

  - 8 slots categoricos, pares adyacentes:  CVD dE 9.1 (protan), normal 19.6  -> PASS
  - 4 primeros slots (regiones, fabricantes): CVD dE 9.1, normal 22.9          -> PASS
  - 3 primeros slots, todos los pares (curvas): CVD dE 9.2, normal 24.0        -> PASS
    Aqua (2.74:1), amarillo (2.11:1) y magenta (2.62:1) quedan bajo 3:1 de contraste ->
    obligan etiquetas directas o tabla. Cada figura que los usa tiene ambas.

Reglas que siguen todas las figuras: un tono por entidad (el color sigue a la entidad, no a su
posicion), marcas finas, rejilla hairline solida, texto en tinta (nunca en el color de la serie),
sin doble eje, leyenda siempre que hay >= 2 series.

Uso:
    import estilo
    estilo.aplicar()
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------- tokens

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
APAGADO = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
NEUTRO = "#f0efec"          # punto medio de la escala divergente

# categoricas, en orden fijo de slot (nunca ciclar)
AZUL = "#2a78d6"            # slot 1
NARANJA = "#eb6834"         # slot 2
AQUA = "#1baf7a"            # slot 3
AMARILLO = "#eda100"        # slot 4
MAGENTA = "#e87ba4"         # slot 5
VERDE = "#008300"           # slot 6
VIOLETA = "#6250d6"         # slot 7
ROJO = "#e34948"            # slot 8

# enfasis: una serie destacada, el resto en gris
GRIS_FONDO = "#c8c7c0"

# el color sigue a la entidad: la misma region / fabricante tiene el mismo tono en todo el proyecto
COLOR_REGION = {"NA": AZUL, "EU": NARANJA, "JP": AQUA, "Otros": AMARILLO}
COLOR_FABRICANTE = {"Nintendo": AZUL, "Sony": NARANJA, "Microsoft": AQUA, "Sega": AMARILLO,
                    "PC": MAGENTA, "Otros": GRIS_FONDO}

# rampa secuencial azul documentada, pasos 100 -> 700 (magnitud continua)
RAMPA_AZUL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SECUENCIAL = LinearSegmentedColormap.from_list("azul_secuencial", RAMPA_AZUL)


# ------------------------------------------------------------ escala divergente (OKLCH)

def _hex_a_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def _rgb_a_hex(rgb):
    return "#" + "".join(f"{int(round(float(np.clip(c, 0, 1)) * 255)):02x}" for c in rgb)


def _hex_a_oklab(h):
    c = _hex_a_rgb(h)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l = 0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2]
    m = 0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2]
    s = 0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2]
    l, m, s = np.cbrt([l, m, s])
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def _oklab_a_lineal(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                     -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                     -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])


def _oklab_a_hex(lab):
    lin = np.clip(_oklab_a_lineal(lab), 0, 1)
    return _rgb_a_hex(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055))


def _brazo_rojo(hex_azules):
    """Para cada paso azul devuelve un rojo (matiz del slot 8) con la MISMA luminosidad OKLab.

    La paleta documentada solo trae rampa para el azul; el brazo rojo se deriva del slot 8 para que
    ambos brazos tengan pasos equivalentes en luminosidad (requisito de la escala divergente).
    El croma baja lo necesario para que el color quepa en sRGB sin recortar canales.
    """
    _, a_r, b_r = _hex_a_oklab(ROJO)
    matiz = np.arctan2(b_r, a_r)
    salida = []
    for h in hex_azules:
        L = _hex_a_oklab(h)[0]
        croma = float(np.hypot(a_r, b_r))
        lab = np.array([L, croma * np.cos(matiz), croma * np.sin(matiz)])
        while croma > 0.01 and not np.all((_oklab_a_lineal(lab) >= -1e-4) & (_oklab_a_lineal(lab) <= 1 + 1e-4)):
            croma *= 0.97
            lab = np.array([L, croma * np.cos(matiz), croma * np.sin(matiz)])
        salida.append(_oklab_a_hex(lab))
    return salida


# brazo azul documentado (pasos 600, 400, 200) + neutro + brazo rojo equivalente en luminosidad
_AZULES_BRAZO = ["#184f95", "#3987e5", "#9ec5f4"]
_ROJOS_BRAZO = _brazo_rojo(_AZULES_BRAZO)
DIVERGENTE = _AZULES_BRAZO + [NEUTRO] + _ROJOS_BRAZO[::-1]
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list("azul_rojo_divergente", DIVERGENTE)


# ------------------------------------------------------------------------ estilo base

def aplicar():
    """Aplica el estilo base: marcas finas, rejilla hairline solida, ejes recesivos."""
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.dpi": 110,

        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,

        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",      # nunca discontinua
        "axes.axisbelow": True,

        "xtick.color": APAGADO,
        "ytick.color": APAGADO,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TINTA_2,
    })


def titular(ax, titulo, subtitulo=None):
    """Pone titulo y, opcionalmente, una linea de contexto debajo.

    Recibe el titulo como argumento a proposito: `ax.get_title()` lee por defecto el titulo
    CENTRADO, mientras que `axes.titlelocation="left"` hace que `set_title` escriba el IZQUIERDO.
    Releer y reescribir borraria el titulo.
    """
    if subtitulo is None:
        ax.set_title(titulo)
        return
    ax.set_title(titulo, pad=28)
    ax.annotate(subtitulo, xy=(0, 1), xycoords="axes fraction", xytext=(0, 9),
                textcoords="offset points", ha="left", va="bottom", fontsize=9.5, color=TINTA_2)


def rejilla(ax, eje="y"):
    """Deja solo la rejilla del eje indicado ('x', 'y', 'ambos' o None)."""
    ax.grid(False)
    if eje in ("x", "y"):
        ax.grid(True, axis=eje)
    elif eje == "ambos":
        ax.grid(True)


# ------------------------------------------------------------ numeros en formato espanol

def num(valor, decimales=0):
    """1431 -> '1.431' ; 4.4 -> '4,4' ; -42 -> '−42' (punto de miles, coma decimal, menos tipografico)."""
    s = f"{valor:,.{decimales}f}".translate(str.maketrans(",.", ".,"))     # intercambia coma y punto
    return s.replace("-", "−")


def pct(valor, decimales=1):
    """4.4 -> '4,4 %' (el valor ya viene en porcentaje, no en fraccion)."""
    return f"{num(valor, decimales)} %"


def formato_eje(decimales=0, sufijo=""):
    """Formateador de ejes con numeros en espanol."""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: num(v, decimales) + sufijo)


def tinta_sobre(fondo):
    """Devuelve blanco o tinta oscura, el que contraste mejor sobre `fondo` (hex).

    Necesario porque la paleta mezcla slots oscuros (azul) y claros (aqua, amarillo, gris): un
    blanco fijo seria ilegible sobre los claros. Es la unica excepcion a 'el texto nunca lleva el
    color de los datos': un rotulo DENTRO de un relleno.
    """
    r, g, b = _hex_a_rgb(fondo)
    canal = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * canal[0] + 0.7152 * canal[1] + 0.0722 * canal[2]
    return "#ffffff" if 1.05 / (lum + 0.05) >= (lum + 0.05) / 0.05 else TINTA


def barras_h(ax, etiquetas, valores, colores=AZUL, decimales=0, sufijo="", alto=0.55, margen=0.012):
    """Barras horizontales finas con el valor en la punta (arriba -> abajo en el orden dado).

    `colores` puede ser un color (todas las barras iguales, categorias nominales) o una lista
    (enfasis: una barra en azul y el resto en gris). Devuelve las posiciones en y.
    """
    y = np.arange(len(etiquetas))[::-1]
    if isinstance(colores, str):
        colores = [colores] * len(etiquetas)
    ax.barh(y, valores, height=alto, color=colores, edgecolor=SUPERFICIE, linewidth=1.0, zorder=3)
    ax.set_yticks(y, etiquetas)
    ax.tick_params(axis="y", length=0)
    rejilla(ax, "x")
    ax.xaxis.set_major_formatter(formato_eje(decimales))
    tope = max(valores)
    for yi, v in zip(y, valores):
        ax.text(v + tope * margen, yi, num(v, decimales) + sufijo, va="center", ha="left", fontsize=9, color=TINTA_2)
    ax.set_xlim(0, tope * 1.14)
    return y


def guardar(fig, nombre, carpeta=None):
    """Guarda la figura en reports/figures/ y devuelve la ruta."""
    carpeta = Path(carpeta) if carpeta else Path(__file__).resolve().parents[1] / "reports" / "figures"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    fig.savefig(ruta)
    return ruta


if __name__ == "__main__":
    print("divergente:", DIVERGENTE)
    for nombre, hexes in {"azul": _AZULES_BRAZO, "rojo": _ROJOS_BRAZO}.items():
        print(nombre, [round(float(_hex_a_oklab(h)[0]), 3) for h in hexes])
