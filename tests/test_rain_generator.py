"""rain_generator (modulo autocontenido) reproduce las funciones de rain_model."""
import calendar

import numpy as np
import pandas as pd

from epidemi.core import rain_generator as rg
from epidemi.core import rain_model as rm


def test_carga_y_markov_iguales():
    a, b = rg.load_daily_rain(), rm.load_daily_rain()
    assert a.equals(b)
    ma, mb = rg.markov_climatology(a, 'Q', 0.), rm.markov_climatology(b, 'Q', 0.)
    assert ma == mb


def test_generador_con_pesos_identico():
    rain = rg.load_daily_rain()
    mk = rg.markov_climatology(rain, 'Q', 0.)
    par = ([500., 100., 10., 300.], [40, 22, 5, 28], [0.38, 0.46, 0.465, 0.41])
    for leap in (False, True):
        a = rg.synthetic_rain_year_markov_weighted(*par, mk, rg.WEIGHTS_ORAN, leap, np.random.default_rng(3))
        b = rm.synthetic_rain_year_markov_weighted(*par, mk, rg.WEIGHTS_ORAN, leap, np.random.default_rng(3))
        assert np.array_equal(a, b)


def test_emparejado_conserva_trimestres():
    rain = rg.load_daily_rain()
    mk = rg.markov_climatology(rain, 'Q', 0.)
    total, dias = np.array([500., 100., 10., 300.]), np.array([40, 22, 5, 28])
    s = rg.synthetic_rain_year_paired(total, dias, [0.38, 0.46, 0.465, 0.41], mk, rg.WEIGHTS_ORAN,
                                      calendar.isleap(2016), [16000, 2016, 0, 2016])
    for q, idx in enumerate(rg.period_day_indices('Q', True)):
        assert np.isclose(s[idx].sum(), total[q]) and (s[idx] > 0).sum() == dias[q]
    # cambiar un trimestre no cambia los otros
    total2 = total.copy(); total2[0] *= 1.4
    s2 = rg.synthetic_rain_year_paired(total2, dias, [0.38, 0.46, 0.465, 0.41], mk, rg.WEIGHTS_ORAN,
                                       True, [16000, 2016, 0, 2016])
    q1 = rg.period_day_indices('Q', True)[0]
    otros = np.setdiff1d(np.arange(366), q1)
    assert np.array_equal(s[otros], s2[otros])


def test_temporada_julio_junio():
    rain = rg.load_daily_rain()
    mk = rg.markov_climatology(rain, 'Q', 0.)
    tabla = rg.aggregate_rain(rain, 'Q', 0.)
    tabla['Alpha'] = 0.4                       # alpha fijo: evita el ajuste (lento)
    s = rg.synthetic_rain_season(2016, realization=3, table=tabla, markov=mk)
    assert s.index[0] == pd.Timestamp('2015-07-01') and s.index[-1] == pd.Timestamp('2016-06-30')
    obs = rain.loc['2015-07-01':'2016-06-30']
    for q in (1, 2, 3, 4):
        a, b = s[s.index.quarter == q], obs[obs.index.quarter == q]
        assert np.isclose(a.sum(), b.sum()) and (a > 0).sum() == (b > 0).sum()


def test_tabla_del_paquete():
    tabla = rg.load_quarterly_table()
    obs = rg.aggregate_rain(rg.load_daily_rain(), 'Q', 0.)
    assert np.allclose(tabla[['TotalRain', 'RainyDays']].to_numpy(), obs[['TotalRain', 'RainyDays']].to_numpy())
    a = rg.synthetic_rain_season(2010, realization=1)
    b = rg.synthetic_rain_season(2010, realization=1, table=tabla,
                                 markov=rg.markov_climatology(rg.load_daily_rain(), 'Q', 0.), weights=rg.WEIGHTS_ORAN)
    assert a.equals(b)
