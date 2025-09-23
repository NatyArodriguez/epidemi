"""Diferents models to generate daily serie of temperature"""
# Tal vez deba agregar las funciones para calcular la media y desviacion por decil
import numpy as np

from scipy.stats import gamma

def anual_temp(mean_params, std_params,
                                shape_params, scale_params, days_decil):
    """Random values of temperature are obtained\
        following the normal distribution of each\
        decile chatacterized by its mean and\
        standard deviation.

    Args:
        data_t (np_array): average temperature per decil.
        data_std (np_array): std per decil.
        data_days (np_array): numbers of day per decil

    Returns:
        np_array: A daily serie of temperature
    """
    
    total_days = np.sum(days_decil)
    
    max_temp_limit = 46
    min_temp_limit = -2
    
    intento = 0
    max_intentos = 1000
    
    while intento < max_intentos:
        tmean_array = np.empty(total_days)
        tmax_array = np.empty(total_days)
        current_index = 0
        limites_excedidos = False
        
        for i, days in enumerate(days_decil):
            daily_means = np.random.normal(mean_params[i], std_params[i], days)
            variations = gamma.rvs(a=shape_params[i], scale=scale_params[i], 
                                  loc=0, size=days)
            daily_maxs = daily_means + variations
            daily_mins = 2 * daily_means - daily_maxs
            
            if np.any(daily_maxs > max_temp_limit)or np.any(daily_mins < min_temp_limit):
                limites_excedidos = True
                break
            
            end_index = current_index + days
            tmean_array[current_index:end_index] = daily_means
            tmax_array[current_index:end_index] = daily_maxs
            current_index = end_index
        
        if not limites_excedidos and current_index == total_days:
            tmin_array = 2 * tmean_array - tmax_array
            return tmin_array, tmax_array, tmean_array
        
        intento += 1

class TemperatureGenerator:
    """Generator of daily temperature serie as a result\
        of point interpolation.
    """
    
    # NUMBER OF DAYS PER MONTH IN A LEAP YEAR 
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    def __init__(self, max_temp_limit=46, min_temp_limit=-2, max_attempts=1000):
        "Initializes decil ranges and temperature limits"
        self.decil_ranges = self._generate_decil_ranges()
        self.max_temp_limit = max_temp_limit
        self.min_temp_limit = min_temp_limit
        self.max_attempts = max_attempts
    
    def _generate_decil_ranges(self):
        """Generates decile intervals (36 in total)
        """
        decil_ranges = []
        day_counter = 0
        decil = 1
        
        for days in self.MONTH_DAYS:
            if days == 31:
                group_sizes = [10, 10, 11]
            elif days == 30:
                group_sizes = [10, 10, 10]
            elif days == 28:
                group_sizes = [10, 10, 8]
        
            start = 0
            for group_size in group_sizes:
                end = start + group_size - 1
                decil_ranges.append((day_counter + start,
                                    day_counter + end, decil))
                start = end + 1
                decil += 1
            
            day_counter += days
        
        return decil_ranges
    
    @staticmethod
    def _linear_interpolation(p1, p2):
        """Determines a straight line between two points.

        Args:
            pi = (day, temperature)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if x2 == x1:
            return 0, y1
        
        m = (y2-y1) / (x2-x1)
        b = y1 - m * x1
        return m, b
    
    def get_decil(self, day):
        """find the decile corresponding to a specific day

        Args:
            day (_type_): _description_
        """
        for start, end, decil in self.decil_ranges:
            if start <= day <= end:
                return decil
    
    def generate_daily_temperature(self, points):
        """Generates a daily temperature series by linear interpolation.

        Args:
            points (np_array): Array of points [[days, temperature], ...]
        
        Returns:
            Array with daily temperatures
        """
        if len(points) < 2:
            raise ValueError("Must be at least two points")
    
        daily_temp = []
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            x1, x2 = p1[0], p2[0]
            
            m, b = self._linear_interpolation(p1,p2)
            
            n_points = int((x2-x1))
            xs = np.linspace(x1, x2, n_points, endpoint=False)
            ys = m * xs + b
            daily_temp.append(ys)
        daily_temp.append([points[-1][1]])
        
        return np.concatenate(daily_temp)
    
    def _validate_temperature_limits(self, max_value, min_value):
        """Valida que las temperaturas estén dentro de los límites permitidos.
        
        Args:
            max_value (float): Temperatura máxima
            min_value (float): Temperatura mínima
            
        Returns:
            bool: True si están dentro de los límites
        """
        return (max_value <= self.max_temp_limit and 
                min_value >= self.min_temp_limit)
    
    def generate_temperature_points(self, days_interval, mean_params, std_params,
                                    shape_params, scale_params):
        """Generates temperature points for specific days according\
            to the decile they belong to.
        Args:
            days_interval (_type_): Array of days.
            mean_params (_type_): Data of mean temperature per decil.
            std_params (_type_): Data of standard deviation per decil.
            shape_params (_type_): Shape parameters for gamma distribution.
            scale_params (_type_): Scale parameters for gamma distribution.
        Returns:
            Array of points [[days, temperature], ...]
        """
        p_tmax = []
        p_tmean = []
        p_tmin = []
        
        for day in days_interval:
            decil = self.get_decil(day)
            # First decil correspond to 0 index
            idx = decil - 1
            
            # Generar temperaturas con validación de límites
            attempt = 0
            valid_temperature = False
            
            while not valid_temperature and attempt < self.max_attempts:
                # Generar valores de temperatura
                mean_value = np.random.normal(mean_params[idx], std_params[idx])
                variations = gamma.rvs(a=shape_params[idx], loc=0,
                                       scale=scale_params[idx])
                max_value = mean_value + variations
                min_value = mean_value - variations
                
                # Validar límites
                if self._validate_temperature_limits(max_value, min_value):
                    valid_temperature = True
                else:
                    attempt += 1
            
            p_tmean.append([day, mean_value])
            p_tmax.append([day, max_value])
            p_tmin.append([day, min_value])
        
        return np.array(p_tmin), np.array(p_tmax), np.array(p_tmean)
    