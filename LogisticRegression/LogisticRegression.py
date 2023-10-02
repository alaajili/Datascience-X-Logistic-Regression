import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self,
        learning_rate=0.05,
        epochs=1500,
        weights_file='weights.npy'
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_file = weights_file

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __cost(self, theta, X, y):
        m = y.shape[0]
        h = self.__sigmoid(np.dot(X, theta))
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def __gradient_descent(self, theta, X, y):
        costs = []
        for _ in range(self.epochs):
            h = self.__sigmoid(np.dot(X, theta))
            theta -= self.learning_rate * (np.dot(X.T, h - y) / y.shape[0])
            costs.append(self.__cost(theta, X, y))
        return theta, costs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.labels = np.unique(y)
        num_labels = self.labels.shape[0]
        n = X.shape[1]
        self.thetas = np.zeros((num_labels, n))

        i = 0
        for label in self.labels:
            yOneVsAll = np.where((y == label), 1, 0)
            theta = np.zeros(n)
            theta , costs = self.__gradient_descent(theta, X, yOneVsAll)
            self.thetas[i, :] = theta
            i += 1
            plt.plot(costs, color=f'C{i}', label=label)

        np.save(self.weights_file, self.thetas)

    # you should fit your model first before call this predict function !!!
    # if you want to run it without fitting a model you should give it a weights matrix (thetas)
    def predict(self, X, thetas=None):
        try:
            if thetas is None:
                thetas = self.thetas
        except AttributeError:
            print('Error: weights are not defined. Set it OR fit your model to be generated (thetas)')
            return None

        probabilities = self.__sigmoid(np.dot(X, thetas.T))
        predictions = np.argmax(probabilities, axis=1)

        labels = np.array(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'])
        predictions = labels[predictions]

        return predictions
