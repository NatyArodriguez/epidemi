"""Diferents models to generate daily serie of temperature"""
# Tal vez deba agregar las funciones para calcular la media y desviacion por decil
import numpy as np

def anual_temp(data_t, data_std, data_days):
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
    temp_array = np.empty([0])
    long = len(data_t)
    for i in range(0,long):
        centro = data_t[i]
        std = data_std[i]
        days = data_days[i]
        aux = np.random.normal(centro, std, days)
        temp_array = np.concatenate([temp_array, aux])
    return temp_array

class TemperatureGenerator:
    """Generator of daily temperature serie as a result\
        of point interpolation.
    """
    
    # NUMBER OF DAYS PER MONTH IN A LEAP YEAR 
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    def __init__(self):
        "Initializes decil ranges"
        self.decil_ranges = self._generate_decil_ranges()
    
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
    
    def generate_temperature_points(self, days_interval, temp_data, std_data):
        """Generates tempurature points for specific days according\
            to the decile they belong to.
        Args:
            days_interval (_type_): Array of days.
            temp_data (_type_): Data of temperature per decil.
            std_data (_type_): Data of standard deviation per decil.
        Returns:
            Array of points [[days, temperature], ...]
        """
        points = []
        
        for day in days_interval:
            decil = self.get_decil(day)
            # First decil correspond to 0 index
            idx = decil - 1
            temp = np.random.normal(temp_data[idx], std_data[idx])
            points.append([day, temp])
        
        return np.array(points)
