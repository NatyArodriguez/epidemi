"""Generador estocastico de lluvia diaria para Oran (Salta).

Generador vigente de la linea de lluvia: cascada multiplicativa de Valdez et al.
(2017, J. Theor. Biol.) por trimestre, con los dias de lluvia ubicados por una
cadena de Markov de dos estados y pesos mensuales dentro de cada trimestre.

Para cada trimestre se fijan tres parametros: lluvia total, dias de lluvia y
alpha (heterogeneidad de la cascada: cerca de 0, lluvia pareja entre los dias
de lluvia; cerca de 1, concentrada en pocos dias). El generador:
  1. elige los dias de lluvia con una cadena de Markov (persistencia
     r = P(llueve|llovio) - P(llueve|no llovio) del trimestre) cuya
     frecuencia en cada mes es la del trimestre por un peso de frecuencia;
  2. reparte la lluvia total en esos dias con la cascada;
  3. asigna las piezas, de mayor a menor, a los dias con probabilidad
     proporcional a un peso de intensidad del mes.
Conserva exactamente el total, los dias y una cascada por trimestre. Los pesos
(WEIGHTS_ORAN) se calibraron con calibrate_monthly_weights sobre 2000-2022 para
reproducir el reparto observado entre los meses de cada trimestre (abr-jun:
70% en abril; jul-sep: 68% en septiembre; oct-dic: 53% en diciembre).

Modulo autocontenido: solo usa numpy y pandas. El codigo de las funciones es
el mismo de epidemi.core.rain_model (se verifica en tests/test_rain_generator.py).

Datos (en epidemi/data/, o donde se guarden):
  - oran_diaria_98_22.txt: lluvia diaria observada (columnas day y rain), para
    la cadena de Markov (markov_climatology).
  - tabla_lluvia_trimestral.txt: lluvia total, dias de lluvia y alpha por año
    y trimestre 2000-2022, la tabla de los trabajos. Se obtuvo con
    fit_alpha_table(aggregate_rain(rain, 'Q', 0.), alphas=ALPHAS_EXTENDED,
    min_rainy_days=2, rng=1); recalcularla da exactamente la misma tabla.

Uso basico:

    import rain_generator as rg

    rain = rg.load_daily_rain('oran_diaria_98_22.txt')
    markov = rg.markov_climatology(rain, 'Q', 0.)
    tabla = rg.load_quarterly_table('tabla_lluvia_trimestral.txt')

    # Temporada julio-junio (2009 = jul-2008 a jun-2009) con los parametros de
    # sus cuatro trimestres, como el escenario base de los trabajos:
    temporada = rg.synthetic_rain_season(2009, realization=0, table=tabla,
                                         markov=markov)

    # Un año calendario con sus parametros:
    total, dias, alpha = rg.year_parameters(2016, tabla)
    serie = rg.synthetic_rain_year_paired(total, dias, alpha, markov,
                                          rg.WEIGHTS_ORAN, leap_year=True,
                                          seed=[16000, 2016, 0, 2016])

Dentro del paquete epidemi, load_daily_rain() y load_quarterly_table() sin
argumentos leen los archivos de epidemi/data/, y synthetic_rain_season(2009)
usa esos archivos y WEIGHTS_ORAN si no se le pasan table, markov ni weights.

Convencion de semillas de los trabajos (realizaciones emparejadas): cada
trimestre se genera con su propio generador aleatorio,
np.random.default_rng(seed + [trimestre]), asi cambiar los parametros de un
trimestre no cambia la lluvia de los otros (synthetic_rain_year_paired).
"""
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = [[m] for m in range(1, 13)]
QUARTERS = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

# Grilla de alpha usada por Valdez: 0.01, 0.02, ..., 0.90
ALPHAS_VALDEZ = np.arange(1, 91) / 100
# Grilla extendida: 0.01, 0.02, ..., 0.99 (evita ajustes censurados en 0.90)
ALPHAS_EXTENDED = np.arange(1, 100) / 100


def _get_periods(periods):
    """Returns the periods as a list of lists of months and checks that\
        each month belongs to exactly one period.
    """
    if isinstance(periods, str):
        if periods == 'M':
            return MONTHS
        if periods == 'Q':
            return QUARTERS
        raise ValueError(f"periods debe ser 'M', 'Q' o una lista. Recibido: {periods}")

    periods = [list(p) for p in periods]
    months = sorted(m for p in periods for m in p)
    if months != list(range(1, 13)):
        raise ValueError("Cada mes (1-12) debe pertenecer a un unico periodo.")
    return periods


def _month_to_period(periods):
    """Dictionary month -> period number (starting at 1)."""
    return {m: i + 1 for i, p in enumerate(periods) for m in p}


def period_day_indices(periods='Q', leap_year=False):
    """Day-of-year indices (0-based) belonging to each period.

    Args:
        periods: 'M', 'Q' or list of lists of months.
        leap_year (bool): True for a 366 days year.

    Returns:
        list of np_array: one array of indices per period.
    """
    periods = _get_periods(periods)
    month_days = [calendar.monthrange(2000 if leap_year else 2001, m)[1]
                  for m in range(1, 13)]
    month_start = np.concatenate(([0], np.cumsum(month_days)[:-1]))

    return [np.concatenate([np.arange(month_start[m - 1],
                                      month_start[m - 1] + month_days[m - 1])
                            for m in p])
            for p in periods]



# Pesos mensuales (meses calendario 1-12) calibrados con calibrate_monthly_weights
# sobre 2000-2022 (notebooks/lluvia/generador_pesos_mensuales.ipynb). Media 1
# dentro de cada trimestre.
WEIGHTS_ORAN = {
    'frequency': np.array([0.8712842177831747, 0.9190671382379264, 1.2096486439788987,
                           1.3369063332904911, 0.9575910072267504, 0.7055026594827585,
                           1.0493581605298825, 0.5431210273242578, 1.4075208121458596,
                           0.7302370780708412, 0.8911245321401403, 1.3786383897890182]),
    'intensity': np.array([1.1226917799245568, 1.084598748126781, 0.7927094719486618,
                           1.8412732333271284, 0.7037400424327519, 0.4549867242401196,
                           0.0426218077915335, 0.4327253290434693, 2.524652863164998,
                           0.5415294939365919, 1.0238858026882216, 1.4345847033751862]),
}


def load_daily_rain(path=None, start_year=2000, end_year=2022):
    """Daily observed rain from a tab-separated file with columns day and rain\
        (format of oran_diaria_98_22.txt).

    Args:
        path (str, optional): Path of the file. If None, the file of the\
            epidemi package (epidemi/data/oran_diaria_98_22.txt).
        start_year (int): First year (included).
        end_year (int): Last year (included).

    Returns:
        pd.Series: Daily rain [mm] indexed by date.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'data' / 'oran_diaria_98_22.txt'
    data = pd.read_table(path, sep='\t')
    data['day'] = pd.to_datetime(data['day'])
    rain = data.set_index('day')['rain']
    return rain.loc[str(start_year):str(end_year)]


def load_quarterly_table(path=None):
    """Quarterly table of the papers: total rain, rainy days and alpha per year\
        and quarter, 2000-2022 (tabla_lluvia_trimestral.txt).

    It is the output of fit_alpha_table(aggregate_rain(rain, 'Q', 0.),\
        alphas=ALPHAS_EXTENDED, min_rainy_days=2, rng=1); Alpha is nan for\
        quarters without rain or with a single rainy day.

    Args:
        path (str, optional): Path of the file. If None, the file of the\
            epidemi package (epidemi/data/tabla_lluvia_trimestral.txt).

    Returns:
        pd.DataFrame: Columns Year | Period | TotalRain | RainyDays | MeanRain |\
            RainVar | Alpha.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'data' / 'tabla_lluvia_trimestral.txt'
    return pd.read_table(path, sep='\t')

def aggregate_rain(rain, periods='Q', threshold=0.):
    """Total rain, rainy days and variance of the rain on rainy days, for\
        each year and period.

    Args:
        rain (pd.Series): Daily rain indexed by date.
        periods: 'M', 'Q' or list of lists of months. The periods are\
            grouped by calendar year (for [12, 1, 2], December is joined\
            with January and February of the same year).
        threshold (float): Minimum rain [mm] to consider a day as rainy\
            (1.0 mm is the WMO criterion). Days below the threshold are\
            considered dry and their rain is discarded, so TotalRain is the\
            sum over rainy days only. With 0 every day with rain > 0 counts.

    Returns:
        pd.DataFrame: Columns Year | Period | TotalRain | RainyDays |\
            MeanRain | RainVar. RainVar is computed with ddof=0 (as Valdez).
    """
    periods = _get_periods(periods)
    month_to_period = _month_to_period(periods)

    df = pd.DataFrame({
        'Year': rain.index.year,
        'Period': rain.index.month.map(month_to_period),
        'rain': rain.values,
    })

    def _stats(x):
        wet = x[(x > 0) & (x >= threshold)]
        n = len(wet)
        return pd.Series({
            'TotalRain': wet.sum(),
            'RainyDays': n,
            'MeanRain': wet.mean() if n > 0 else 0.,
            'RainVar': wet.var(ddof=0) if n > 0 else 0.,
        })

    table = df.groupby(['Year', 'Period'])['rain'].apply(_stats).unstack()
    table = table.reset_index()
    table['RainyDays'] = table['RainyDays'].astype(int)
    return table


def piecewise(x, alpha):
    """Piecewise function of Valdez: fraction of the interval that goes to\
        one of the two subintervals.

    Args:
        x (np_array): Uniform random numbers in [0, 1).
        alpha (float): Heterogeneity parameter, 0 < alpha < 1.

    Returns:
        np_array: Fractions in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    return np.where(
        x < alpha / 2,
        x * (1 - alpha) / alpha,
        np.where(
            x < 1 - alpha / 2,
            0.5 + (x - 0.5) * alpha / (1 - alpha),
            1 - (1 - x) * (1 - alpha) / alpha,
        ),
    )


def cascade(total_rain, alpha, n_pieces, n_sim=1, rng=None):
    """Splits the total rain in n_pieces by the fracturing process of Valdez.

    In each step all the subintervals of the previous step are split in two.
    If n_pieces is not a power of 2, in the last step only some subintervals\
        of the previous step, randomly chosen and without repetition, are\
        split (as in the paper and the original Fortran code).

    Args:
        total_rain (float): Amount of rain to split.
        alpha (float): Heterogeneity parameter, 0 < alpha < 1.
        n_pieces (int): Number of rainy days.
        n_sim (int): Number of independent realizations.
        rng (np.random.Generator, optional): Random generator. If None, a new\
            one is created (different result in each call).

    Returns:
        np_array: Shape (n_sim, n_pieces). Each row sums total_rain.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha debe estar entre 0 y 1. Recibido: {alpha}")

    rng = np.random.default_rng(rng)
    n_pieces = int(n_pieces)

    L = np.full((n_sim, 1), float(total_rain))
    if n_pieces <= 0:
        return np.empty((n_sim, 0))

    # Pasos completos: se parten todos los subintervalos
    while 2 * L.shape[1] <= n_pieces:
        f = piecewise(rng.random(L.shape), alpha)
        L = np.concatenate((L * f, L * (1 - f)), axis=1)

    # Ultimo paso: se parten 'extra' subintervalos distintos del paso anterior
    extra = n_pieces - L.shape[1]
    if extra > 0:
        idx = np.argsort(rng.random(L.shape), axis=1)[:, :extra]
        selected = np.take_along_axis(L, idx, axis=1)
        f = piecewise(rng.random(selected.shape), alpha)
        np.put_along_axis(L, idx, selected * f, axis=1)
        L = np.concatenate((L, selected * (1 - f)), axis=1)

    return L


def cascade_variance(total_rain, alpha, n_pieces, n_sim=10000, rng=None):
    """Mean over realizations of the variance (ddof=0) of the pieces."""
    return cascade(total_rain, alpha, n_pieces, n_sim, rng).var(axis=1).mean()


def fit_alpha(total_rain, rainy_days, rain_var, alphas=ALPHAS_VALDEZ,
              n_sim=10000, min_rainy_days=2, rng=None):
    """Alpha whose cascade reproduces the observed variance of the rain on\
        rainy days (the first of the grid with the smallest difference).

    Args:
        total_rain (float): Observed total rain of the period.
        rainy_days (int): Observed rainy days of the period.
        rain_var (float): Observed variance (ddof=0) on rainy days.
        alphas (np_array): Grid of alphas to test.
        n_sim (int): Realizations per alpha (Valdez used 100000).
        min_rainy_days (int): Minimum rainy days to attempt the fit (default\
            2; with a single rainy day the variance is 0 for any alpha, so\
            the fit would be meaningless).
        rng (np.random.Generator, optional): Random generator.

    Returns:
        float: Fitted alpha, or nan if there was no rain or fewer than\
            min_rainy_days rainy days.
    """
    if total_rain <= 0 or rainy_days <= 0 or rainy_days < min_rainy_days:
        return np.nan

    rng = np.random.default_rng(rng)
    diff = [abs(rain_var - cascade_variance(total_rain, a, rainy_days,
                                            n_sim, rng))
            for a in alphas]
    return float(alphas[int(np.argmin(diff))])


def fit_alpha_table(table, alphas=ALPHAS_VALDEZ, n_sim=10000,
                    min_rainy_days=2, rng=None, verbose=False):
    """Fits alpha for every row (year, period) of the aggregate_rain table.

    Periods without rain or with fewer than min_rainy_days rainy days are\
        not fitted: 'Alpha' is left as nan for them (see fit_alpha).

    Returns:
        pd.DataFrame: Copy of the table with an extra column 'Alpha'.
    """
    rng = np.random.default_rng(rng)
    table = table.copy()
    fitted = []
    for row in table.itertuples():
        if verbose:
            print(f"Year {row.Year}, period {row.Period}")
        fitted.append(fit_alpha(row.TotalRain, row.RainyDays, row.RainVar,
                                alphas, n_sim, min_rainy_days, rng))
    table['Alpha'] = fitted
    return table


def climatology(table):
    """Typical rain per period: mean over the years.

    TotalRain and RainyDays are averaged over all the years (including the\
        years without rain). Alpha is averaged only over the years with a\
        valid fit: fit_alpha_table already leaves 'Alpha' as nan for the\
        periods without rain or with too few rainy days, and pandas skips\
        nan in mean/std/median/count.

    Args:
        table (pd.DataFrame): Output of fit_alpha_table.

    Returns:
        pd.DataFrame: Indexed by Period, columns TotalRain | RainyDays |\
            Std_TR | Std_RD | Median_TR | Alpha | Std_Alpha | Median_Alpha |\
            N_Alpha.
    """
    rain_stats = table.groupby('Period').agg(
        TotalRain=('TotalRain', 'mean'),
        RainyDays=('RainyDays', 'mean'),
        Std_TR=('TotalRain', 'std'),
        Std_RD=('RainyDays', 'std'),
        Median_TR=('TotalRain', 'median'),
    )

    alpha_stats = table.groupby('Period').agg(
        Alpha=('Alpha', 'mean'),
        Std_Alpha=('Alpha', 'std'),
        Median_Alpha=('Alpha', 'median'),
        N_Alpha=('Alpha', 'count'),
    )

    return rain_stats.join(alpha_stats)


def synthetic_rain_year(total_rain, rainy_days, alpha, periods='Q',
                        leap_year=False, rng=None):
    """Daily synthetic rain for one year.

    In each period the total rain is split by the cascade in the rainy days\
        (rounded to the nearest integer), and the pieces are placed in days\
        of the period randomly chosen.

    Args:
        total_rain (array): Total rain per period.
        rainy_days (array): Rainy days per period (can be non integer).
        alpha (array): Alpha per period.
        periods: 'M', 'Q' or list of lists of months.
        leap_year (bool): True for a 366 days year.
        rng (np.random.Generator, optional): Random generator.

    Returns:
        np_array: Daily rain, 365 or 366 days.
    """
    rng = np.random.default_rng(rng)
    total_rain = np.asarray(total_rain, dtype=float)
    rainy_days = np.asarray(rainy_days, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    day_indices = period_day_indices(periods, leap_year)
    if not (len(total_rain) == len(rainy_days) == len(alpha) == len(day_indices)):
        raise ValueError("total_rain, rainy_days y alpha deben tener un valor por periodo.")

    serie = np.zeros(366 if leap_year else 365)

    for p, days in enumerate(day_indices):
        n = min(int(round(rainy_days[p])), len(days))
        if n == 0 or total_rain[p] <= 0:
            continue
        pieces = cascade(total_rain[p], alpha[p], n, 1, rng)[0]
        serie[rng.choice(days, n, replace=False)] = pieces

    return serie


def markov_climatology(rain, periods='Q', threshold=0.):
    """Climatology for the Markov generator of rainy days.

    For each period, pools all the years of `rain` and computes the\
        probabilities of a rainy day after a rainy day and after a dry day\
        (transitions within the period of the same year only), and the\
        position of the first rainy day of the period in each year.

    Args:
        rain (pd.Series): Daily rain with a DatetimeIndex (e.g.\
            load_daily_rain()).
        periods: 'M', 'Q' or list of lists of months.
        threshold (float): Minimum rain [mm] for a rainy day (same\
            convention as aggregate_rain: rain > 0 and rain >= threshold).

    Returns:
        dict: 'periods', 'transitions' (list of (P(wet|wet), P(wet|dry)) per\
            period) and 'first_days' (list per period of the 0-based index,\
            within the period, of the first rainy day of each year with\
            rain, in chronological order).
    """
    periods = _get_periods(periods)
    month_to_period = _month_to_period(periods)
    df = pd.DataFrame({'Year': rain.index.year, 'Month': rain.index.month,
                       'rain': rain.values})
    df['Period'] = df['Month'].map(month_to_period)

    transitions, first_days = [], []
    for p in range(1, len(periods) + 1):
        wet_wet = wet_total = dry_wet = dry_total = 0
        firsts = []
        for _, g in df[df['Period'] == p].groupby('Year'):
            x = g['rain'].to_numpy()
            wet = ((x > 0) & (x >= threshold)).astype(int)
            for t in range(len(wet) - 1):
                if wet[t] == 1:
                    wet_total += 1
                    wet_wet += wet[t + 1]
                else:
                    dry_total += 1
                    dry_wet += wet[t + 1]
            rainy = np.where(wet == 1)[0]
            if len(rainy) > 0:
                firsts.append(int(rainy[0]))
        transitions.append((wet_wet / wet_total if wet_total else 0.,
                            dry_wet / dry_total if dry_total else 0.))
        first_days.append(firsts)

    return {'periods': periods, 'transitions': transitions,
            'first_days': first_days}


def synthetic_rain_year_markov(total_rain, rainy_days, alpha, markov,
                               leap_year=False, rng=None):
    """Daily synthetic rain for one year, with the rainy days placed by a\
        two-state Markov chain.

    Same as synthetic_rain_year (same total rain, rainy days and alpha per\
        period, amounts from the cascade), except for the choice of the\
        rainy days. In each period: the first rainy day is drawn from the\
        historical first rainy days (`markov['first_days']`), then the\
        chain walks day by day with the period transition probabilities\
        until the rainy days are complete; if the period ends before, the\
        missing days are drawn at random among the remaining days after the\
        first one (or among all the days of the period if there are not\
        enough). The cascade pieces are placed at random on the chosen days.\
        Unlike the uniform placement, it reproduces the clustering of the\
        observed rainy days (see notebooks/lluvia/lluvia_modelo_corregido).

    Args:
        total_rain (array): Total rain per period.
        rainy_days (array): Rainy days per period (can be non integer).
        alpha (array): Alpha per period.
        markov (dict): Output of markov_climatology (defines the periods).
        leap_year (bool): True for a 366 days year.
        rng (np.random.Generator, optional): Random generator.

    Returns:
        np_array: Daily rain, 365 or 366 days.
    """
    rng = np.random.default_rng(rng)
    total_rain = np.asarray(total_rain, dtype=float)
    rainy_days = np.asarray(rainy_days, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    day_indices = period_day_indices(markov['periods'], leap_year)
    if not (len(total_rain) == len(rainy_days) == len(alpha) == len(day_indices)):
        raise ValueError("total_rain, rainy_days y alpha deben tener un valor por periodo.")

    serie = np.zeros(366 if leap_year else 365)

    for p, days in enumerate(day_indices):
        n = min(int(round(rainy_days[p])), len(days))
        if n == 0 or total_rain[p] <= 0:
            continue
        firsts = markov['first_days'][p]
        if not firsts:
            raise ValueError(f"El periodo {p + 1} no tiene dias de lluvia en la climatologia.")
        first = min(firsts[rng.integers(len(firsts))], len(days) - 1)
        p_ww, p_wd = markov['transitions'][p]

        chosen = [first]
        state = 1                                   # the first day is rainy
        d = first + 1
        while len(chosen) < n and d < len(days):
            state = 1 if rng.random() < (p_ww if state == 1 else p_wd) else 0
            if state == 1:
                chosen.append(d)
            d += 1

        if len(chosen) < n:
            remaining = [x for x in range(first + 1, len(days)) if x not in chosen]
            missing = n - len(chosen)
            if len(remaining) < missing:
                remaining = [x for x in range(len(days)) if x not in chosen]
            chosen.extend(rng.choice(remaining, missing, replace=False).tolist())

        pieces = cascade(total_rain[p], alpha[p], n, 1, rng)[0]
        rng.shuffle(pieces)
        serie[days[np.array(chosen[:n])]] = pieces

    return serie


def _day_months(leap_year=False):
    """Calendar month (1-12) of each day of the year."""
    return pd.date_range('2000-01-01' if leap_year else '2001-01-01',
                         periods=366 if leap_year else 365).month.to_numpy()


def synthetic_rain_year_markov_weighted(total_rain, rainy_days, alpha, markov,
                                        weights, leap_year=False, rng=None):
    """Daily synthetic rain for one year: Markov placement with monthly\
        weights inside each period.

    Same period parameters as synthetic_rain_year_markov (total rain, rainy\
        days and alpha per period, one cascade per period, persistence of the\
        period chain), but the rain is distributed among the months of each\
        period with two sets of monthly weights:

    - frequency: in month m the chain has stationary probability of a rainy\
        day pi_P*w_m (pi_P of the period chain) and the same persistence\
        r = P(wet|wet) - P(wet|dry) as the period: P(wet|wet) = pi_m + r(1 - pi_m),\
        P(wet|dry) = pi_m(1 - r). The missing days (if the chain ends before)\
        are drawn with probability proportional to w_m.
    - intensity: the cascade pieces, from largest to smallest, go to the\
        chosen days with probability proportional to v_m (without\
        replacement), so heavier months receive the larger pieces.

    With all weights equal to 1 the placement is that of\
        synthetic_rain_year_markov (not bit to bit: the random numbers are\
        used in a different order). Weights are fitted with\
        calibrate_monthly_weights.

    Args:
        total_rain, rainy_days, alpha (array): One value per period.
        markov (dict): Output of markov_climatology (defines the periods).
        weights (dict): 'frequency' and 'intensity', arrays of 12 values\
            (calendar months).
        leap_year (bool): True for a 366 days year.
        rng (np.random.Generator, optional): Random generator.

    Returns:
        np_array: Daily rain, 365 or 366 days.
    """
    rng = np.random.default_rng(rng)
    total_rain = np.asarray(total_rain, dtype=float)
    rainy_days = np.asarray(rainy_days, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    w_freq = np.asarray(weights['frequency'], dtype=float)
    w_int = np.asarray(weights['intensity'], dtype=float)

    day_indices = period_day_indices(markov['periods'], leap_year)
    if not (len(total_rain) == len(rainy_days) == len(alpha) == len(day_indices)):
        raise ValueError("total_rain, rainy_days y alpha deben tener un valor por periodo.")
    month = _day_months(leap_year)

    serie = np.zeros(366 if leap_year else 365)

    for p, days in enumerate(day_indices):
        n = min(int(round(rainy_days[p])), len(days))
        if n == 0 or total_rain[p] <= 0:
            continue
        m = month[days] - 1                         # month (0-11) of each day of the period
        p_ww, p_wd = markov['transitions'][p]
        pi, r = p_wd / (1. - p_ww + p_wd), p_ww - p_wd
        pi_m = np.clip(pi * w_freq, 1e-4, 0.999)
        ww_m, wd_m = pi_m + r * (1. - pi_m), pi_m * (1. - r)

        firsts = markov['first_days'][p]
        if not firsts:
            raise ValueError(f"El periodo {p + 1} no tiene dias de lluvia en la climatologia.")
        first = min(firsts[rng.integers(len(firsts))], len(days) - 1)
        chosen = [first]
        state = 1
        d = first + 1
        while len(chosen) < n and d < len(days):
            mm = m[d]
            state = 1 if rng.random() < (ww_m[mm] if state == 1 else wd_m[mm]) else 0
            if state == 1:
                chosen.append(d)
            d += 1

        if len(chosen) < n:
            remaining = np.array([x for x in range(first + 1, len(days)) if x not in chosen], dtype=int)
            missing = n - len(chosen)
            if len(remaining) < missing:
                remaining = np.array([x for x in range(len(days)) if x not in chosen], dtype=int)
            prob = w_freq[m[remaining]]
            chosen.extend(rng.choice(remaining, missing, replace=False, p=prob / prob.sum()).tolist())
        chosen = np.array(chosen[:n])

        pieces = np.sort(cascade(total_rain[p], alpha[p], n, 1, rng)[0])[::-1]
        free = list(range(n))
        w_day = w_int[m[chosen]]
        target = np.empty(n, dtype=int)
        for k in range(n):                          # largest piece first
            prob = w_day[free]
            j = rng.choice(len(free), p=prob / prob.sum())
            target[k] = free.pop(j)
        serie[days[chosen[target]]] = pieces

    return serie


def monthly_fractions(rain, periods='Q'):
    """Pooled fraction of the rainy days and of the rain of each period that\
        falls in each of its months.

    Args:
        rain (pd.Series): Daily rain with a DatetimeIndex.
        periods: 'M', 'Q' or list of lists of months.

    Returns:
        tuple: (days, rain), arrays of 12 values (calendar months); each\
            period sums 1.
    """
    periods = _get_periods(periods)
    df = pd.DataFrame({'Month': rain.index.month, 'rain': rain.values})
    by_month = df.groupby('Month')['rain'].agg(total='sum', days=lambda x: (x > 0).sum())
    f_days, f_rain = np.zeros(12), np.zeros(12)
    for months in periods:
        idx = np.array(months) - 1
        t = by_month.loc[months, 'total'].to_numpy(float)
        dd = by_month.loc[months, 'days'].to_numpy(float)
        f_rain[idx] = t / t.sum() if t.sum() > 0 else 1. / len(months)
        f_days[idx] = dd / dd.sum() if dd.sum() > 0 else 1. / len(months)
    return f_days, f_rain


def calibrate_monthly_weights(rain, table, markov, n_iter=15, n_rep=30,
                              step=0.8, rng=None):
    """Monthly weights of synthetic_rain_year_markov_weighted that reproduce\
        the observed monthly fractions inside each period.

    Simulates every year of `table` with its own period parameters and\
        updates the weights multiplicatively,\
        w <- w*(observed/simulated)**step, until the pooled fractions of\
        rainy days (frequency weights) and of rain (intensity weights) of\
        each month match monthly_fractions(rain). Weights are normalized to\
        mean 1 inside each period.

    Args:
        rain (pd.Series): Daily rain (observed), for the target fractions.
        table (pd.DataFrame): Year, Period, TotalRain, RainyDays, Alpha (e.g.\
            fit_alpha_table output, same periods as `markov`). Missing alpha\
            -> median of the period.
        markov (dict): Output of markov_climatology.
        n_iter (int): Iterations.
        n_rep (int): Realizations per year and iteration.
        step (float): Exponent of the multiplicative update (damping).
        rng (np.random.Generator, optional): Random generator.

    Returns:
        dict: 'frequency' and 'intensity' (arrays of 12), and 'fit': pooled\
            simulated fractions (days, rain) with the final weights.
    """
    rng = np.random.default_rng(rng)
    periods = markov['periods']
    obs_days, obs_rain = monthly_fractions(rain, periods)
    years = sorted(table['Year'].unique())
    alpha_med = table.groupby('Period')['Alpha'].median()
    params = {}
    for y in years:
        f = table[table['Year'] == y].set_index('Period').reindex(range(1, len(periods) + 1))
        al = f['Alpha'].to_numpy(float)
        al = np.where(np.isnan(al), alpha_med.reindex(range(1, len(periods) + 1)).to_numpy(), al)
        params[y] = (f['TotalRain'].to_numpy(float), f['RainyDays'].to_numpy(float), al)

    def simulate(weights, n):
        acc_days, acc_rain = np.zeros(12), np.zeros(12)
        for y in years:
            leap = calendar.isleap(y)
            month = _day_months(leap) - 1
            for _ in range(n):
                s = synthetic_rain_year_markov_weighted(*params[y], markov, weights, leap, rng)
                np.add.at(acc_days, month, s > 0)
                np.add.at(acc_rain, month, s)
        f_days, f_rain = np.zeros(12), np.zeros(12)
        for months in periods:
            idx = np.array(months) - 1
            f_days[idx] = acc_days[idx] / acc_days[idx].sum()
            f_rain[idx] = acc_rain[idx] / acc_rain[idx].sum()
        return f_days, f_rain

    weights = {'frequency': np.ones(12), 'intensity': np.ones(12)}
    for _ in range(n_iter):
        f_days, f_rain = simulate(weights, n_rep)
        weights['frequency'] = weights['frequency'] * (obs_days / f_days) ** step
        weights['intensity'] = np.maximum(weights['intensity'] * (obs_rain / f_rain) ** step, 1e-4)
        for months in periods:
            idx = np.array(months) - 1
            for k in ('frequency', 'intensity'):
                weights[k][idx] = weights[k][idx] / weights[k][idx].mean()
    weights['fit'] = simulate(weights, n_rep)
    return weights

def year_parameters(year, table=None, typical_alpha_q3=True):
    """Period parameters of an observed year, as used in the papers.

    Alpha missing (periods without rain or with a single rainy day) is\
        replaced by the median alpha of the period over all years and, if\
        typical_alpha_q3, the alpha of Jul-Sep (few rainy days, unreliable\
        fit) is always the median of that period.

    Args:
        year (int): Calendar year.
        table (pd.DataFrame, optional): Quarterly table (load_quarterly_table\
            or fit_alpha_table output). If None, load_quarterly_table().
        typical_alpha_q3 (bool): Use the median alpha for Jul-Sep.

    Returns:
        tuple: (total_rain, rainy_days, alpha), arrays of one value per period.
    """
    if table is None:
        table = load_quarterly_table()
    med = table.groupby('Period')['Alpha'].median()
    f = table[table['Year'] == year].set_index('Period').sort_index()
    total = f['TotalRain'].to_numpy(float)
    days = f['RainyDays'].to_numpy(float)
    alpha = f['Alpha'].to_numpy(float)
    alpha = np.where(np.isnan(alpha), med.reindex(f.index).to_numpy(), alpha)
    if typical_alpha_q3:
        alpha[2] = med.loc[3]
    return total, days, alpha


def synthetic_rain_year_paired(total_rain, rainy_days, alpha, markov, weights,
                               leap_year=False, seed=None):
    """One synthetic year with one random generator per period (convention of\
        the papers, for paired realizations).

    Period i is generated with np.random.default_rng(list(seed) + [i + 1]) and\
        the parameters of the other periods set to zero, so changing the\
        parameters of one period leaves the rain of the others unchanged.

    Args:
        total_rain, rainy_days, alpha (array): One value per period.
        markov (dict): Output of markov_climatology.
        weights (dict): 'frequency' and 'intensity' (e.g. WEIGHTS_ORAN).
        leap_year (bool): True for a 366 days year.
        seed (list of int): Base seed, e.g. [16000, season, realization, year].

    Returns:
        np_array: Daily rain, 365 or 366 days.
    """
    total_rain = np.asarray(total_rain, dtype=float)
    rainy_days = np.asarray(rainy_days, dtype=float)
    seed = [] if seed is None else list(seed)
    serie = np.zeros(366 if leap_year else 365)
    for i in range(len(total_rain)):
        m = np.zeros(len(total_rain)); m[i] = 1
        serie += synthetic_rain_year_markov_weighted(total_rain * m, rainy_days * m, alpha, markov,
                                                     weights, leap_year,
                                                     np.random.default_rng(seed + [i + 1]))
    return serie


def synthetic_rain_season(season, realization=0, table=None, markov=None, weights=None,
                          base_seed=16000, typical_alpha_q3=True):
    """Synthetic rain of a July-June season with the parameters of its four\
        quarters (Jul-Sep and Oct-Dec of season - 1, Jan-Mar and Apr-Jun of\
        season), as the baseline of the papers.

    The two calendar years are generated with synthetic_rain_year_paired and\
        seed [base_seed, season, realization, year], and the July-June part is\
        kept; the discarded quarters do not affect the kept ones (one random\
        generator per quarter).

    Args:
        season (int): Season, e.g. 2009 = July 2008 to June 2009.
        realization (int): Realization number (0, 1, 2, ...).
        table (pd.DataFrame, optional): Quarterly table. If None,\
            load_quarterly_table().
        markov (dict, optional): Output of markov_climatology. If None, from\
            load_daily_rain() (2000-2022, quarters, threshold 0).
        weights (dict, optional): 'frequency' and 'intensity'. If None,\
            WEIGHTS_ORAN.
        base_seed (int): First element of the seed (16000 in the papers).
        typical_alpha_q3 (bool): Median alpha for Jul-Sep (see year_parameters).

    Returns:
        pd.Series: Daily rain from July 1 of season - 1 to June 30 of season.
    """
    if table is None:
        table = load_quarterly_table()
    if markov is None:
        markov = markov_climatology(load_daily_rain(), 'Q', 0.)
    if weights is None:
        weights = WEIGHTS_ORAN
    parts = []
    for year in (season - 1, season):
        total, days, alpha = year_parameters(year, table, typical_alpha_q3)
        parts.append(synthetic_rain_year_paired(total, days, alpha, markov, weights,
                                                calendar.isleap(year),
                                                [base_seed, season, realization, year]))
    dates = pd.date_range(f'{season - 1}-01-01', f'{season}-12-31')
    serie = pd.Series(np.concatenate(parts), index=dates)
    return serie.loc[f'{season - 1}-07-01':f'{season}-06-30']


if __name__ == '__main__':
    # Ejemplo: temporada 2016 (jul-2015 a jun-2016), realizacion 0, con los archivos
    # de epidemi/data/; compara los parametros de cada trimestre con la serie sintetica.
    tabla = load_quarterly_table()
    s = synthetic_rain_season(2016, realization=0, table=tabla)
    print('Temporada 2016, total y dias de lluvia por trimestre (tabla / sintetico):')
    for (anio, q) in ((2015, 3), (2015, 4), (2016, 1), (2016, 2)):
        f = tabla[(tabla.Year == anio) & (tabla.Period == q)].iloc[0]
        x = s[(s.index.year == anio) & (s.index.quarter == q)]
        print(f'  {anio} T{q}: {f.TotalRain:7.1f} / {x.sum():7.1f} mm,'
              f' dias {f.RainyDays:.0f} / {(x > 0).sum()}')
