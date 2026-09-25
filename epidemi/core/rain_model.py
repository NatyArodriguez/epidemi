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


def synthetic_rain_series(start_year, end_year, total_rain, rainy_days, alpha,
                          periods='Q', rng=None):
    """Daily synthetic rain from 1 January of start_year to 31 December of\
        end_year, taking into account the leap years.

    Returns:
        np_array: Daily rain, one value per day of the interval.
    """
    rng = np.random.default_rng(rng)
    return np.concatenate([
        synthetic_rain_year(total_rain, rainy_days, alpha, periods,
                            calendar.isleap(year), rng)
        for year in range(start_year, end_year + 1)
    ])
