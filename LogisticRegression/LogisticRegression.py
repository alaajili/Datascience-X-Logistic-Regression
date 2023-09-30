import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.05, epochs=500) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    
    def __sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))

    def __gradient_descent(self, learning_rate, epochs):
        
        for _ in range(epochs):
            

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        labels = np.unique(y)
        num_labels = y.shape[0]
        m, n = X.shape
        theta = np.zeros((num_labels, n))

        for label in labels:
            yOneVsAll = np.where((y == label), 1, 0)
            print(label, yOneVsAll)

