import numpy as np
import epidemi.core.utils as u
from .utils import load_data_file

rain = load_data_file('hystoric_rain.txt')
alpha = load_data_file('hystoric_alphas.txt')
oran_medio = load_data_file('Oran_2001_2017_medio.txt')

fechas_oran = np.arange('2001-01-01', '2018-01-01', dtype='datetime64[D]')
fechas_str = fechas_oran.astype(str) 
mask_no_29feb = np.array([not f.endswith('02-29') for f in fechas_str])
fechas_oran = fechas_oran[mask_no_29feb]
oran_medio = oran_medio[mask_no_29feb]

tmin = oran_medio[:,0]
tmean = oran_medio[:,2]
hr = oran_medio[:,6]

def matrix_epic_size_point(sigmas, n_iterations, season, suma, data_rain=rain,
                           data_alpha=alpha):
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
    sigma_rain = sigmas[0]
    sigma_alpha = sigmas[1]
    
    rain_syntetic = data_rain[:,2]*sigma_rain + data_rain[:,0]
    alpha_syntetic = data_alpha[:,1]*sigma_alpha + data_alpha[:,0]
    days_syntetic = data_rain[:,1]

    for i in range(n_sim):
        ci = [ci_sim[i], 1]
        for j in range(n_iterations):
            rain_serie = u.anual_rain(rain_syntetic, alpha_syntetic,
                                      days_syntetic)
            rr = np.tile(rain_serie, 17)
            aux = u.fun(350, 1.69, season, suma, ci, rr,
                        tmin, tmean, hr)
            matriz[i,j] = aux[1]
    
    return matriz
