""" Function for rain model """

import numpy as np
import math
import random

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
