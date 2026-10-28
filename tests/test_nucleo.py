"""Pruebas de las piezas donde un error pasaria desapercibido: fugas de informacion y empates.

Se ejecutan con:
    python tests/test_nucleo.py        (sin dependencias extra)
    pytest tests                       (si pytest esta instalado)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F
import limpieza as L
import modelado as M


# ------------------------------------------------------------------ metricas con empates

def test_puntaje_constante_da_exactamente_la_tasa_base():
    # bug real corregido: con empates, el orden de las filas (ordenado por ventas) regalaba aciertos
    y = np.r_[np.ones(30), np.zeros(170)]                      # los exitos van PRIMERO, como en el dataset
    assert M.precision_en_top(y, np.full(200, 0.3), 0.10) == y.mean()


def test_puntaje_perfecto_da_precision_1():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(int)
    assert M.precision_en_top(y, y + rng.random(500) * 1e-6, 0.10) == 1.0


def test_empate_parcial_en_el_corte_usa_el_valor_esperado():
    y = np.r_[np.ones(30), np.zeros(20), np.zeros(950)]
    p = np.r_[np.ones(50), np.zeros(950)]                      # 50 empatados arriba (30 exitos) y 950 abajo
    assert M.precision_en_top(y, p, 0.10) == 0.30              # k = 100: entran los 50 y 50 de los 950 (sin exitos)


# ------------------------------------------------------------------ historiales sin mirar al futuro

def _saga(anios, ventas_log, exitos):
    return pd.DataFrame({"franquicia": "x", "anio": anios, "ventas_log": ventas_log, "exito": exitos})


def test_historial_con_rezago_1_solo_ve_hasta_dos_anios_antes():
    b = _saga([2000, 2001, 2002, 2003, 2004], [1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 1, 1, 1])
    h = F.historial_previo(b, "franquicia", "fr", rezago=1)
    assert h["fr_n_prev"].tolist() == [0, 0, 1, 2, 3]          # el titulo de 2004 ve 2000-2002, no 2003
    assert h["fr_media_prev"].iloc[4] == 2.0                   # media de 1, 2 y 3
    assert h["fr_max_prev"].iloc[4] == 3.0
    assert np.isnan(h["fr_media_prev"].iloc[0]) and np.isnan(h["fr_media_prev"].iloc[1])


def test_historial_sin_rezago_ve_el_anio_anterior_pero_nunca_el_propio():
    b = _saga([2000, 2001, 2001, 2002], [1.0, 2.0, 9.0, 3.0], [0, 0, 1, 1])      # dos fichas en 2001 (versiones simultaneas)
    h = F.historial_previo(b, "franquicia", "fr", rezago=0)
    assert h["fr_n_prev"].tolist() == [0, 1, 1, 3]             # las dos de 2001 solo ven 2000, y no se ven entre si
    assert h["fr_exito_prev"].iloc[3] == 1 / 3                 # 2002 ve 2000 (0) y 2001 (0, 1)


def test_historial_no_mezcla_sagas():
    b = pd.concat([_saga([2000, 2005], [1.0, 2.0], [0, 1]), _saga([2003, 2008], [7.0, 8.0], [1, 1]).assign(franquicia="y")], ignore_index=True)
    h = F.historial_previo(b, "franquicia", "fr", rezago=1)
    assert h["fr_n_prev"].tolist() == [0, 1, 0, 1]
    assert h["fr_media_prev"].iloc[3] == 7.0                   # la saga "y" solo ve su propia entrega de 2003


# ------------------------------------------------------------------ franquicias y limpieza

def test_franquicia_agrupa_entregas_de_una_misma_saga():
    f = L.extraer_franquicia
    assert f("FIFA 14") == f("FIFA Soccer 13") == "fifa"
    assert f("Call of Duty: Black Ops II") == f("Call of Duty 3") == "call of duty"
    assert f("Mario Kart Wii") == f("Mario Kart DS") == "mario kart"
    assert f("007: Quantum of Solace") != ""                   # un titulo que es solo un numero no queda vacio


def test_limpieza_pasa_todas_las_verificaciones():
    df, registro = L.limpiar()
    fallos = [d for d, ok in L.validar(df) if not ok]
    assert not fallos, fallos
    assert len(df) == registro["filas_crudas"] - 2             # solo se eliminan 2 duplicados
    assert registro["parciales"]["n_parciales"] == 151


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for nombre, prueba in pruebas:
        prueba()
        print(f"✔ {nombre}")
    print(f"\n{len(pruebas)}/{len(pruebas)} pruebas en verde")
