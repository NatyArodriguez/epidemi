"""Intervenciones térmicas sobre una trayectoria de temperatura ya generada.

Se aplican después de generar la realización base completa (los dos años
calendario de la temporada), así que no hacen sorteos ni reintentos: la
realización intervenida queda emparejada día a día con la base. Versión de
la segunda devolución del director (oct 2026), contrastes C1 y C2.
"""
import numpy as np


def transform(tm, tn, dates, noise=1., amp=1.):
    """Aplica C1 (variabilidad dentro de la década) y C2 (brecha Tmean - Tmin).

    C1 (`noise`): en cada década (días 1-10, 11-20 y 21 a fin de mes) escala
    los desvíos realizados de Tmean respecto de la media de la década y suma
    el mismo cambio diario a Tmin. Conserva la media de cada década y la
    brecha diaria Tmean - Tmin.

    C2 (`amp`): Tmin' = Tmean - amp*(Tmean - Tmin). Tmean queda idéntica.

    No se recorta el resultado (límites físicos); el orden Tmin <= Tmean se
    conserva por construcción.

    Args:
        tm: Tmean diaria de la realización base.
        tn: Tmin diaria de la realización base.
        dates: fechas de cada día (`pd.DatetimeIndex`, mismo largo que `tm`).
        noise: factor de los desvíos dentro de la década (1 = sin cambio).
        amp: factor de la brecha Tmean - Tmin (1 = sin cambio).

    Returns:
        (tmean, tmin) intervenidas (copias; las entradas no se modifican).
    """
    mt = np.array(tm, copy=True)
    nt = np.array(tn, copy=True)
    if noise != 1:
        grupos = dates.year*1000 + dates.month*10 + np.minimum((dates.day - 1)//10, 2)
        for grupo in np.unique(grupos):
            ii = grupos == grupo
            delta = (noise - 1)*(mt[ii] - mt[ii].mean())
            mt[ii] += delta
            nt[ii] += delta
    if amp != 1:
        nt = mt - amp*(mt - nt)
    return mt, nt
