# Final epidemic size matrices working with synthetic temperature only.

import numpy as np
import epidemi.core.utils as u
from epidemi.core.temp_model import anual_temp, TemperatureGenerator
from epidemi.core.utils import load_data_file


# decil | days | tmin | tmax | tmean | std tmin | std tmax | std tmean 
temp = load_data_file('hystoric_temp.txt')

# tmin	| tmax	| tmean | tmedian | rain | hrm | hr
oran_medio = load_data_file('Oran_2001_2017_medio.txt')

fechas_oran = np.arange('2001-01-01', '2018-01-01', dtype='datetime64[D]')
fechas_str = fechas_oran.astype(str) 
mask_no_29feb = np.array([not f.endswith('02-29') for f in fechas_str])
fechas_oran = fechas_oran[mask_no_29feb]
oran_medio = oran_medio[mask_no_29feb]

rain = oran_medio[:,4]
hr = oran_medio[:,6]

def temp_matrix_epic_size(delta, n_iterations, season, suma, data_T=temp):
    """This function generates a matrix where each column represents the daily\
        total epidemic size for a year, simulated under different temperature\
        scenaries specific by delta values.

    Parameters:
        sigmas: tuple of float
            tuple with two entrance, the firts it most be sigma rain and\
            the second must be sigma alpha.
        n_iterations: int
            Number of iteration that you wish.
    """
    date_i = np.datetime64(suma[0])
    date_f = np.datetime64(suma[1])
    ci_sim = np.arange(date_i, date_f + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    ci_sim_str = ci_sim.astype(str)
    mask_no_29feb = np.array([not f.endswith('02-29') for f in ci_sim_str])

    ci_sim = ci_sim[mask_no_29feb]
    
    n_sim = len(ci_sim)
    
    matriz = np.zeros([n_sim, n_iterations])
    
    std_tmin = data_T[:,5]
    std_tmean = data_T[:,7]
    
    tmin_syntetic = data_T[:,2] + delta
    tmean_syntetic = data_T[:,4] + delta
    
    days_decil = data_T[:,1].astype(int)

    for i in range(n_sim):
        ci = [ci_sim[i], 1]
        for j in range(n_iterations):
            tmin_serie = anual_temp(tmin_syntetic, std_tmin, days_decil)
            tmean_serie = anual_temp(tmean_syntetic, std_tmean, days_decil)
            
            tmin = np.tile(tmin_serie, 17)
            tmean = np.tile(tmean_serie, 17)
            aux = u.fun(350, 1.69, season, suma, ci, rain, tmin,
                        tmean, hr)
            matriz[i,j] = aux[1]
    return matriz


def temp_warm_cold(deltas, n_iterations, season, suma, data_T=temp):
    """Function to increas the center of the deciles. Deltas it must\
        be a list with the values for the increments for warm months\
        (Oct - Mar) and for cold months (Apr - Sep), respectively.

    Args:
        deltas (list): [delta warm months, delta cold months]
        n_iterations (int): number of simulations.
        season (tuple): start and end dates of the season that you\
            want simulated.
        suma (tuple): start and end dates to add up the total cases.
        data_T (_type_, optional): _description_. Defaults to temp.

    Returns:
        _type_: _description_
    """
    
    deciles = np.ones(36)
    deciles[:9] *= deltas[0]
    deciles[9:27] *= deltas[1]
    deciles[27:] *= deltas[0]
    
    date_i = np.datetime64(suma[0])
    date_f = np.datetime64(suma[1])
    ci_sim = np.arange(date_i, date_f + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    ci_sim_str = ci_sim.astype(str)
    mask_no_29feb = np.array([not f.endswith('02-29') for f in ci_sim_str])

    ci_sim = ci_sim[mask_no_29feb]
    
    n_sim = len(ci_sim)
    
    matriz = np.zeros([n_sim, n_iterations])
    
    std_tmin = data_T[:,5]
    std_tmean = data_T[:,7]
    
    tmin_syntetic = data_T[:,2] + deciles
    tmean_syntetic = data_T[:,4] + deciles
    
    days_decil = data_T[:,1].astype(int)

    for i in range(n_sim):
        ci = [ci_sim[i], 1]
        for j in range(n_iterations):
            tmin_serie = anual_temp(tmin_syntetic, std_tmin, days_decil)
            tmean_serie = anual_temp(tmean_syntetic, std_tmean, days_decil)
            
            tmin = np.tile(tmin_serie, 17)
            tmean = np.tile(tmean_serie, 17)
            aux = u.fun(350, 1.69, season, suma, ci, rain, tmin,
                        tmean, hr)
            matriz[i,j] = aux[1]
    return matriz

def temp_matrix_interpolation(delta, n_iterations, season, suma, days, data_T=temp):
    date_i = np.datetime64(suma[0])
    date_f = np.datetime64(suma[1])
    ci_sim = np.arange(date_i, date_f + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    ci_sim_str = ci_sim.astype(str)
    mask_no_29feb = np.array([not f.endswith('02-29') for f in ci_sim_str])

    ci_sim = ci_sim[mask_no_29feb]
    
    n_sim = len(ci_sim)
    
    matriz = np.zeros([n_sim, n_iterations])
    
    std_tmin = data_T[:,5]*0.5
    std_tmean = data_T[:,7]*0.5
    
    tmin_syntetic = data_T[:,2] + delta
    tmean_syntetic = data_T[:,4] + delta
    
    temp_gen = TemperatureGenerator()

    for i in range(n_sim):
        ci = [ci_sim[i], 1]
        for j in range(n_iterations):
            p_tmin = temp_gen.generate_temperature_points(days, tmin_syntetic, std_tmin)
            p_tmean = temp_gen.generate_temperature_points(days, tmean_syntetic, std_tmean)
            
            tmin_serie = temp_gen.generate_daily_temperature(p_tmin)
            tmean_serie = temp_gen.generate_daily_temperature(p_tmean)
            
            tmin = np.tile(tmin_serie, 17)
            tmean = np.tile(tmean_serie, 17)
            aux = u.fun(350, 1.69, season, suma, ci, rain, tmin,
                        tmean, hr)
            matriz[i,j] = aux[1]
    return matriz

def temp_matrix_epic_size2(sigmas, n_iterations, season, suma, data_T=temp):
    # Esta la hice para trabajar con la media de entre una serie de temperatura maxima y minima
    """This function generates a matrix where each column represents the daily\
        total epidemic size for a year, simulated under different rainfall\
        scenaries specific by the tuple sigmas.

    Parameters:
        sigmas: tuple of float
            tuple with two entrance, the firts it most be sigma rain and\
            the second must be sigma alpha.
        n_iterations: int
            Number of iteration that you wish.
    """
    date_i = np.datetime64(suma[0])
    date_f = np.datetime64(suma[1])
    ci_sim = np.arange(date_i, date_f + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    ci_sim_str = ci_sim.astype(str)
    mask_no_29feb = np.array([not f.endswith('02-29') for f in ci_sim_str])

    ci_sim = ci_sim[mask_no_29feb]
    
    n_sim = len(ci_sim)
    
    matriz = np.zeros([n_sim, n_iterations])
    sigma_tmin = sigmas[0]
    sigma_tmean = sigmas[1]
    
    std_tmin = data_T[:,5]
    std_tmax = data_T[:,6]
    
    tmin_syntetic = std_tmin*sigma_tmin + data_T[:,2]
    tmax_syntetic = std_tmax*sigma_tmean + data_T[:,3]
    
    days_decil = data_T[:,1].astype(int)

    for i in range(n_sim):
        ci = [ci_sim[i], 1]
        for j in range(n_iterations):
            tmin_serie = anual_temp(tmin_syntetic, std_tmin, days_decil)
            tmax_serie = anual_temp(tmax_syntetic, std_tmax, days_decil)
            
            tmean_serie = (tmin_serie + tmax_serie)/2
            
            tmin = np.tile(tmin_serie, 17)
            tmean = np.tile(tmean_serie, 17)
            aux = u.fun(350, 1.69, season, suma, ci, rain, tmin,
                        tmean, hr)
            matriz[i,j] = aux[1]
    return matriz
