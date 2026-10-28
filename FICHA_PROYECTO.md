# FICHA DE PROYECTO

## 1. Información general

| | |
|---|---|
| **Nombre del proyecto** | ¿Qué hace que un videojuego venda más de un millón de copias? |
| **Especialidad** | Data Science (nivel Jr.) |
| **Fuente del proyecto** | Dataset público *Video Game Sales* (Kaggle), generado con un *scraper* de vgchartz.com. 16.598 filas × 11 columnas. Copia congelada en `data/raw/vgsales.csv` (SHA-256 `E2076095…5D4EBE`) |
| **Link a la fuente original** | https://www.kaggle.com/datasets/gregorut/videogamesales — **pendiente de verificar** (no se pudo abrir desde la sesión de trabajo; la descripción se contrastó solo con copias secundarias) |
| **Link al proyecto publicado** | **Pendiente** — el proyecto todavía no está publicado |
| **Fecha de cierre** | 2026-10-27 |

## 2. Objetivo

Averiguar qué distingue a los videojuegos que superan el millón de copias y comprobar con qué fiabilidad *real* se puede anticipar un éxito usando solo información disponible antes del lanzamiento. Está pensado para un equipo de catálogo o marketing que debe repartir presupuesto entre muchos lanzamientos y necesita **priorizar** en cuáles apostar.

## 3. Plan de trabajo

1. **Exploración inicial y auditoría de calidad** — revisar cada columna, buscar nulos disfrazados, fechas imposibles, duplicados, registros parciales y relaciones entre columnas que sean trampas (`Global_Sales` = suma regional, `Rank` = objetivo ordenado).
2. **Preparación** — limpieza no destructiva (banderas en lugar de borrar), ventana de análisis 1995–2014 fijada antes de modelar, 18 verificaciones automáticas.
3. **Análisis exploratorio con hipótesis** — concentración del éxito, ciclo de vida de las consolas, géneros, gustos regionales, publisher, saga y versiones simultáneas, con pruebas estadísticas y control de confusión.
4. **Modelado y evaluación** — variables disponibles antes del lanzamiento, validación walk-forward, comparación contra baselines, ablación de variables, prueba final y prueba de estrés.
5. **Conclusiones** — interpretación, análisis de errores, límites y próximos pasos.

## 4. Preguntas clave

**¿Cómo sabes que el modelo no está usando información del futuro?**
Tres salvaguardas: (a) walk-forward, siempre se entrena con años anteriores al evaluado; (b) los historiales de saga exigen un año de rezago de madurez, porque las ventas finales de lo lanzado el año anterior todavía no existían al lanzar; (c) una «escalera de optimismo» mide cuánto inflaría la cifra cada atajo: el mismo modelo da AUC 0,887 con *split* aleatorio y 0,853 con la validación honesta.

**¿Mejora algo respecto a una solución simple?**
Sí, con intervalo de confianza. Frente a «apostar por el publisher con mejor historial» (AUC 0,804) el modelo suma +0,072 de AUC (IC 95 % 0,039–0,104). Frente a una regresión logística no hay diferencia (+0,002; IC −0,006 a +0,011), así que la señal es sobre todo aditiva y el modelo simple sirve igual.

**¿Qué tan fiables son los datos?**
Las ventas son estimaciones redondeadas a 10.000 copias, la cobertura cae un 42 % tras 2011 y los últimos años no han acumulado ventas. Por eso hay una ventana de análisis, una prueba de estrés en 2015–2016 y un apartado de límites. El dataset tampoco cumple su propia descripción («más de 100.000 copias»: el 35 % de las fichas vende menos).

## 5. Qué se hizo y cómo

- **Limpieza** (`src/limpieza.py`): `"Unknown"` en publisher → nulo con bandera (no se imputa: son fichas de bajo perfil); 8 años imposibles → nulo con bandera; 2 duplicados reales eliminados y los homónimos legítimos conservados; 151 registros parciales marcados y excluidos del análisis por título; derivadas (fabricante, edad de la plataforma, clave de saga).
- **Variables** (`src/features.py`): plataforma, género, publisher agrupado (52 categorías), edad de la plataforma, versiones simultáneas e historial de saga (4 variables) construido con rezago de 1 año. El año calendario se descarta porque casi no separa ventas y no se extrapola.
- **Validación** (`src/modelado.py`): walk-forward con selección en 2006–2011 y prueba final en 2012–2014; intervalos por *bootstrap* de **títulos** (no de filas, porque las versiones de un mismo juego son casi gemelas); desempate exacto de empates en las métricas de ranking.
- **Selección de variables** con una regla fijada de antemano (un bloque se conserva si sube el AUC ≥ 0,005).
- **Métricas**: AUC, PR-AUC, precisión en el 10 % superior y *lift*, *log-loss*; no se usa *accuracy* porque «siempre no» ya acierta el 88 %.
- **Pruebas** (`tests/test_nucleo.py`): 8 pruebas sobre las piezas donde un error pasaría desapercibido (empates en las métricas de ranking, historiales de saga que no miran al futuro, detección de sagas y verificaciones de la limpieza).
- **Visualización**: paleta validada para daltonismo, sin doble eje, números en formato español.
- **Stack**: Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, Jupyter.

## 6. Resultados

- **Métrica principal** (prueba 2012–2014, 1.784 fichas, 14,3 % de éxitos): **AUC 0,877** (IC 95 % 0,851–0,902) y **PR-AUC 0,548** frente a 0,143 del azar.
- **Valor práctico:** el 10 % de lanzamientos mejor puntuado contiene un 61 % de éxitos (≈ 4,2× la tasa base) y capta el 42 % de los éxitos; el 20 % capta el 67 %.
- **Hallazgos del análisis:** el 10 % de las fichas vende el 58 % de las ventas; el publisher separa las ventas cuatro veces más que el género (ε² 0,25 frente a 0,06); Nintendo logra un 46 % de éxitos frente al 11 % del resto; la saga con historial multiplica las probabilidades (OR 1,87 → 1,61 ajustado); «salir en 3+ plataformas» baja de OR 1,64 a 1,30 al ajustar por tamaño del publisher; la cuota de Japón cae 5,9 puntos.
- **Prueba de estrés:** 2015 se comporta como los años de prueba (AUC 0,876); 2016 rompe el modelo (AUC 0,789; 4,4 % observado frente a 10,1 % predicho).

## 7. Conclusiones

**Qué se aprendió**
- El éxito se puede anticipar en parte, pero la señal viene sobre todo de *quién publica* y de *la historia de la saga*, no del género.
- La validación importa tanto como el modelo: sin cuidado, la cifra habría sido 0,034 de AUC más alta sin que el modelo valiera más.
- Un modelo más complejo no ayuda cuando el límite está en la información disponible.
- Ordenar bien no es lo mismo que estimar probabilidades bien: el modelo subestima la proporción de éxitos tras 2011.

**Qué se mejoraría con más tiempo**
- Unificar las variantes regionales de un publisher («Sony Computer Entertainment Europe» frente a «Sony Computer Entertainment») y detectar sagas con coincidencia aproximada (*Pokémon*, *Mario*).
- Recalibrar las probabilidades con datos recientes.
- Un modelo por región, empezando por Japón, y una regresión para la magnitud de las ventas.
- Añadir reseñas y fecha exacta de lanzamiento si estuvieran disponibles.

**Qué mencionaría en una entrevista**
- La escalera de optimismo y por qué la cifra honesta es 0,853 y no 0,887.
- El rezago de madurez: un historial construido con ventas finales es una fuga de información sutil que un *split* temporal simple no detecta.
- Que «salir en más plataformas» parecía ayudar y casi la mitad del efecto era confusión con el tamaño del publisher.
- Un error propio corregido: la precisión en el 10 % superior del baseline «azar» salía 1,000 porque los empates se resolvían por el orden de las filas (que estaba ordenado por ventas); se detectó, se corrigió con el valor esperado y quedó cubierto por una prueba (`tests/test_nucleo.py`) que reproduce el fallo.
- El límite honesto: la prueba final no es completamente «virgen».

## 8. Checklist antes de publicar

- [x] README claro (qué hace, cómo ejecutarlo, resultados y límites)
- [x] Archivos organizados, sin sueltos ni versiones viejas
- [x] Sin credenciales ni datos sensibles (dataset público, sin datos personales)
- [ ] Link a la fuente original funcionando — **verificar** (no se pudo abrir desde la sesión de trabajo)
- [ ] Revisar la licencia del dataset antes de redistribuir `data/raw/vgsales.csv`
- [ ] Proyecto publicado y accesible — **pendiente**
- [ ] Link compartido con el coach — **pendiente**
