import pandas as pd
import math

class Description:

    def __init__(self, df):
        self.df = df
    
    def mean_and_count(self, column: str):
        count = 0
        sum = 0
        for value in self.df[column]:
            if not pd.isnull(value):
                count += 1
                sum += value
        obj = {
            'count': count,
            'mean': sum / count
        }
        return obj

    def get_sorted_list(self, column: str):
        sorted = self.df[column].sort_values().tolist()
        sorted = [x for x in sorted if not math.isnan(x)]
        return sorted

    def quantile(self, p: int, column: str, sorted) -> float:

        n = len(sorted)
        index = (p / 100) * (n - 1)

        if index.is_integer():
            return sorted[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            lower_value = sorted[lower_index]
            upper_value = sorted[upper_index]
            fraction = index - lower_index
            interpolate_value = lower_value + (upper_value - lower_value) * fraction
            return interpolate_value
    
    def get_values(self, column: str):
        mc = self.mean_and_count(column)
        sorted = self.get_sorted_list(column)
        obj = {
            'count': mc['count'],
            'mean': mc['mean'],
            'min': self.quantile(0, column, sorted),
            '25%': self.quantile(25, column, sorted),
            '50%': self.quantile(50, column, sorted),
            '75%': self.quantile(75, column, sorted),
            'max': self.quantile(100, column, sorted),
        }
        return obj
        
        
        

        