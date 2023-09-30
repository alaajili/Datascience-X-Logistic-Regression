import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.05, epochs=500) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))

    def __cost(self, theta: float, X: np.ndarray, y: np.ndarray) -> float:
        m = len(y)
        h = self.__sigmoid(np.dot(X, theta))
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
        

    def __gradient_descent(self, theta, X, y):
        
        for _ in range(self.epochs):

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        labels = np.unique(y)
        num_labels = y.shape[0]
        m, n = X.shape
        theta_matrix = np.zeros((num_labels, n))

        for label in labels:
            yOneVsAll = np.where((y == label), 1, 0)
            theta = np.zeros(n)
            self.__gradient_descent(theta, X, yOneVsAll)

