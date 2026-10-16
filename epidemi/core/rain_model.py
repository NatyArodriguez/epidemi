""" Function for rain model """

import calendar
import math
import random

import numpy as np
import pandas as pd

from epidemi.core.utils import load_data_file_pandas

def contador_lluvia(tupla):
    long = 0
    aux = []
    sumar = 1
    no_sumar = 0
    days = 0
    for i in tupla:
        if i > 0:
            aux.append(i)
            long = long + sumar*i
            days = days + 1
        else:
            long = long + no_sumar*i
    aux = np.array(aux)
    mean = aux.mean()
    var = aux.var()
    return [long, days, mean, var]

def FT(lengths, alpha):
    rhos = np.random.rand(len(lengths))
    result = np.zeros_like(lengths)

    mask1 = rhos < alpha / 2
    mask2 = (rhos >= alpha / 2) & (rhos <= 1 - alpha / 2)
    mask3 = rhos > 1 - alpha / 2

    result[mask1] = lengths[mask1] * (rhos[mask1] * (1 - alpha) / alpha)
    result[mask2] = lengths[mask2] * (0.5 + ((rhos[mask2] - 0.5) * (alpha / (1 - alpha))))
    result[mask3] = lengths[mask3] * (1 - ((1 - rhos[mask3]) * ((1 - alpha) / alpha)))

    return result

def syntetic_rain(total_rain, alpha, number_pieces):
    L = np.array([total_rain])
    steps = math.log2(number_pieces)
    n = int(steps)

    for _ in range(n):
        A = FT(L, alpha)
        B = L - A
        L = np.concatenate((A, B))

    if steps % 1 != 0:
        while len(L) < number_pieces:
            idx = np.random.randint(0, len(L))
            elem = L[idx]
            a = FT(np.array([elem]), alpha)[0]
            b = elem - a
            L[idx] = a
            L = np.insert(L, idx + 1, b)

    return L

'''
def anual_rain(rain_data, alpha_data, columns):
    months_days = {
        0:31, 1:28, 2:31, 3:30, 4:31, 5:30,
        6:31, 7:31, 8:30, 9:31, 10:30, 11:31
    }

    rain = []
    rain_column, days_column, alpha_column = columns

    for j in range(12):
        total_rain = rain_data[j, rain_column]
        rainy_days = int(rain_data[j, days_column])
        alpha = alpha_data[j, alpha_column]
        month_days = months_days[j]

        if rainy_days == 0:
            rain.extend([0]*month_days)
            continue

        daily_rain = syntetic_rain(total_rain, alpha, rainy_days)
        np.random.shuffle(daily_rain)

        serie_rain = np.zeros(month_days)
        days_with_rain = random.sample(range(month_days), rainy_days)

        for i, val in zip(days_with_rain, daily_rain):
            serie_rain[i] = val

        rain.extend(serie_rain)

    return np.array(rain)
'''

def anual_rain(rain_data, alpha_data, days_data):
    months_days = {
        0:31, 1:28, 2:31, 3:30, 4:31, 5:30,
        6:31, 7:31, 8:30, 9:31, 10:30, 11:31
    }

    rain = []

    for j in range(12):
        total_rain = rain_data[j]
        rainy_days = int(days_data[j])
        alpha = alpha_data[j]
        month_days = months_days[j]

        if rainy_days == 0:
            rain.extend([0]*month_days)
            continue

        daily_rain = syntetic_rain(total_rain, alpha, rainy_days)
        np.random.shuffle(daily_rain)

        serie_rain = np.zeros(month_days)
        days_with_rain = random.sample(range(month_days), rainy_days)

        for i, val in zip(days_with_rain, daily_rain):
            serie_rain[i] = val

        rain.extend(serie_rain)

    return np.array(rain)


# ############################################################################
# Modelo de lluvia sintetica por periodos (Valdez et al. 2017, J. Theor. Biol.)
# ############################################################################
#
# Flujo de trabajo:
#   1. rain = load_daily_rain(2000, 2022)
#   2. tabla = aggregate_rain(rain, periods='Q')
#   3. tabla = fit_alpha_table(tabla)
#   4. clima = climatology(tabla)
#   5. serie = synthetic_rain_series(2007, 2016, clima['TotalRain'],
#                                    clima['RainyDays'], clima['Alpha'],
#                                    periods='Q')
#
# Un periodo es un grupo de meses completos. 'periods' puede ser:
#   'M' -> 12 periodos mensuales
#   'Q' -> 4 trimestres (Ene-Mar, Abr-Jun, Jul-Sep, Oct-Dic)
#   lista de listas de meses, p. ej. [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
# Cada mes debe pertenecer a un unico periodo.

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


def load_daily_rain(start_year=2000, end_year=2022):
    """Daily observed rain in Oran (oran_diaria_98_22.txt).

    Args:
        start_year (int): First year (included).
        end_year (int): Last year (included).

    Returns:
        pd.Series: Daily rain [mm] indexed by date.
    """
    data = load_data_file_pandas('oran_diaria_98_22.txt')
    data['day'] = pd.to_datetime(data['day'])
    rain = data.set_index('day')['rain']
    return rain.loc[str(start_year):str(end_year)]


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


def synthetic_rain_series(start_year, end_year, total_rain, rainy_days, alpha,
                          periods='Q', rng=None, markov=None):
    """Daily synthetic rain from 1 January of start_year to 31 December of\
        end_year, taking into account the leap years.

    Args:
        markov (dict, optional): Output of markov_climatology. If given, the\
            rainy days are placed by the Markov chain\
            (synthetic_rain_year_markov) and `periods` is taken from it;\
            otherwise they are placed uniformly (synthetic_rain_year).

    Returns:
        np_array: Daily rain, one value per day of the interval.
    """
    rng = np.random.default_rng(rng)
    if markov is not None:
        return np.concatenate([
            synthetic_rain_year_markov(total_rain, rainy_days, alpha, markov,
                                       calendar.isleap(year), rng)
            for year in range(start_year, end_year + 1)
        ])
    return np.concatenate([
        synthetic_rain_year(total_rain, rainy_days, alpha, periods,
                            calendar.isleap(year), rng)
        for year in range(start_year, end_year + 1)
    ])
