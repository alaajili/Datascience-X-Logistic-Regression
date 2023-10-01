import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __cost(self, theta, X, y):
        m = len(y)
        h = self.__sigmoid(np.dot(X, theta))
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
        

    def __gradient_descent(self, theta, X, y):
        m = len(y)
        costs = []
        # print(self.__sigmoid(np.dot(X, theta)))
        for _ in range(self.epochs):
            h = self.__sigmoid(np.dot(X, theta))
            theta -= self.learning_rate * (np.dot(X.T, h - y) / m)
            costs.append(self.__cost(theta, X, y))
        return theta, costs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        labels = np.unique(y)
        num_labels = labels.shape[0]
        m, n = X.shape
        theta_matrix = np.zeros((num_labels, n))

        i = 0
        for label in labels:
            yOneVsAll = np.where((y == label), 1, 0)
            theta = np.zeros(n)
            theta , costs = self.__gradient_descent(theta, X, yOneVsAll)
            theta_matrix[i, :] = theta
            i += 1
            plt.plot(costs, color=f'C{i}', label=label)

        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()


