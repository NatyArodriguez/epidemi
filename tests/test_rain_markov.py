import calendar

import numpy as np
import pytest

from epidemi.core import rain_model as rm


@pytest.fixture(scope='module')
def lluvia():
    return rm.load_daily_rain(2000, 2022)


@pytest.fixture(scope='module')
def clima(lluvia):
    return rm.markov_climatology(lluvia, 'Q', 0.)


def test_climatologia_markov(clima):
    assert len(clima['transitions']) == 4 and len(clima['first_days']) == 4
    for p_ww, p_wd in clima['transitions']:
        assert 0. <= p_wd < p_ww <= 1.          # la lluvia se agrupa: mas probable despues de un dia de lluvia
    assert all(len(f) > 0 for f in clima['first_days'])


@pytest.mark.parametrize('year', [2014, 2016])
def test_conserva_total_y_dias(clima, year):
    total = np.array([400., 80., 10., 300.])
    dias = np.array([30, 15, 3, 25])
    alpha = np.array([0.3, 0.4, 0.5, 0.35])
    leap = calendar.isleap(year)
    s = rm.synthetic_rain_year_markov(total, dias, alpha, clima, leap, rng=1)
    assert len(s) == (366 if leap else 365)
    for q, idx in enumerate(rm.period_day_indices('Q', leap)):
        assert s[idx].sum() == pytest.approx(total[q])
        assert (s[idx] > 0).sum() == dias[q]


def test_reproducible_y_serie(clima):
    args = ([400., 80., 10., 300.], [30, 15, 3, 25], [0.3, 0.4, 0.5, 0.35])
    a = rm.synthetic_rain_series(2010, 2012, *args, rng=7, markov=clima)
    b = rm.synthetic_rain_series(2010, 2012, *args, rng=7, markov=clima)
    assert np.array_equal(a, b) and len(a) == 365 + 365 + 366
    # sin markov, sigue siendo la colocacion uniforme de antes
    u = rm.synthetic_rain_series(2010, 2012, *args, rng=7)
    rng = np.random.default_rng(7)
    v = np.concatenate([rm.synthetic_rain_year(*args, 'Q', calendar.isleap(y), rng) for y in (2010, 2011, 2012)])
    assert np.array_equal(u, v)


def test_markov_agrupa_mas_que_uniforme(clima):
    args = ([400., 80., 10., 300.], [30, 15, 3, 25], [0.3, 0.4, 0.5, 0.35])
    def p_ll_ll(s):
        w = (s > 0).astype(int)
        return ((w[:-1] == 1) & (w[1:] == 1)).sum() / (w[:-1] == 1).sum()
    m = np.mean([p_ll_ll(rm.synthetic_rain_year_markov(*args, clima, False, rng=r)) for r in range(50)])
    u = np.mean([p_ll_ll(rm.synthetic_rain_year(*args, 'Q', False, rng=r)) for r in range(50)])
    assert m > u + 0.1


def test_pesos_mensuales_conserva_y_unos(clima):
    total = np.array([400., 80., 10., 300.])
    dias = np.array([30, 15, 3, 25])
    alpha = np.array([0.3, 0.4, 0.5, 0.35])
    pesos = {'frequency': np.linspace(0.5, 1.5, 12), 'intensity': np.linspace(1.5, 0.5, 12)}
    for leap in (False, True):
        s = rm.synthetic_rain_year_markov_weighted(total, dias, alpha, clima, pesos, leap, rng=3)
        assert len(s) == (366 if leap else 365)
        for q, idx in enumerate(rm.period_day_indices('Q', leap)):
            assert s[idx].sum() == pytest.approx(total[q])
            assert (s[idx] > 0).sum() == dias[q]
    # con pesos 1, mismo agrupamiento que el generador de Markov
    unos = {'frequency': np.ones(12), 'intensity': np.ones(12)}
    def p_ll_ll(s):
        w = (s > 0).astype(int)
        return ((w[:-1] == 1) & (w[1:] == 1)).sum() / (w[:-1] == 1).sum()
    a = np.mean([p_ll_ll(rm.synthetic_rain_year_markov_weighted(total, dias, alpha, clima, unos, False, rng=r)) for r in range(80)])
    b = np.mean([p_ll_ll(rm.synthetic_rain_year_markov(total, dias, alpha, clima, False, rng=r)) for r in range(80)])
    assert abs(a - b) < 0.03


def test_pesos_mensuales_mueven_la_lluvia(clima):
    # mas peso de intensidad en diciembre -> mas lluvia de oct-dic en diciembre
    total, dias, alpha = [400., 80., 10., 300.], [30, 15, 3, 25], [0.3, 0.4, 0.5, 0.35]
    idx = rm.period_day_indices('M', False)
    def frac_dic(v_dic):
        v = np.ones(12); v[11] = v_dic
        pesos = {'frequency': np.ones(12), 'intensity': v}
        return np.mean([rm.synthetic_rain_year_markov_weighted(total, dias, alpha, clima, pesos, False, rng=r)[idx[11]].sum()
                        for r in range(60)]) / 300.
    assert frac_dic(4.) > 1.4 * frac_dic(1.)
