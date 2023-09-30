import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.05, epochs=500) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
    

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        labels = np.unique(y)
        m, num_features = X.shape
        theta = np.zeros((y.shape[0], num_features))
        print(theta)
        # for label in labels:
            
