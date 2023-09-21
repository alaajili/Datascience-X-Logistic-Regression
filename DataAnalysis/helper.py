import pandas as pd

class Helper:

    def __init__(self, df):
        self.df = df

    def count(self, column: str) -> int:
        count = 0
        for value in self.df[column]:
            if not pd.isnull(value):
                count += 1
        return count
    
    def min(self, column: str) -> float:
        min = self.df[column].iloc[0]
        for value in self.df[column]:
            if (value < min):
                min = value
        return min

    def max(self, column: str) -> float:
        max = self.df[column].iloc[0]
        for value in self.df[column]:
            if (value > max):
                max = value
        return max
    
    def mean(self, column: str) -> float:
        count = 0
        sum = 0
        for value in self.df[column]:
            if not pd.isnull(value):
                count += 1
                sum += value
        return sum / count

    def quantile(self, p: int, column: str) -> float:
        sorted = self.df[column].sort_values()
        sorted.dropna(inplace=True)
        print(sorted)
        n = len(sorted)

        index = (p / 100) * (n - 1)

        if index.is_integer():
            return sorted[index]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            lower_value = sorted[lower_index]
            upper_value = sorted[upper_index]
            fraction = index - lower_index
            interpolate_value = lower_value + (upper_value - lower_value) * fraction
            return interpolate_value
        
        

        