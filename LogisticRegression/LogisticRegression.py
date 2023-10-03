import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class LogisticRegression:

    def __init__(self,
        random_state=0,
        learning_rate=0.1,
        epochs=500,
        weights_file='weights.npy',
        batch_size=128
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_file = weights_file
        self.random_state = random_state
        self.batch_size = batch_size

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

    # Stochastic gradient descent
    def __sgd(self, theta, X, y):
        costs = []
        for _ in range(self.epochs):
            rand_index = np.random.randint(0, X.shape[0])
            xi, yi = X[rand_index], y[rand_index]
            h = self.__sigmoid(np.dot(xi, theta))
            theta -= self.learning_rate * xi * (h - yi)
            costs.append(self.__cost(theta, X, y))
        return theta, costs

    # mini batch gradient descent
    def __mini_batch_gd(self, theta, X, y):
        costs = []
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        for _ in range(self.epochs):
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, X.shape[0])
                X_batch = X[indices[start:end]]
                y_batch = y[indices[start:end]]
                h = self.__sigmoid(np.dot(X_batch, theta))
                
                theta -= self.learning_rate * (np.dot(X_batch.T, h - y_batch) / y_batch.shape[0])
            costs.append(self.__cost(theta, X, y))
        return theta, costs


    def __get_optimizer(self, opt: str):
        optimizers = {
            'gd': self.__gradient_descent,
            'sgd': self.__sgd,
            'mbgd': self.__mini_batch_gd
        }
        return optimizers[opt]

    def fit(self, X: np.ndarray, y: np.ndarray, optimizer='gd') -> None:
        np.random.seed(self.random_state)
        self.labels = np.unique(y)
        num_labels = self.labels.shape[0]
        n = X.shape[1]
        self.thetas = np.zeros((num_labels, n))
        optimizer = self.__get_optimizer(optimizer)

        i = 0
        for label in self.labels:
            yOneVsAll = np.where((y == label), 1, 0)
            theta = np.zeros(n)
            theta , costs = optimizer(theta, X, yOneVsAll)
            self.thetas[i, :] = theta
            i += 1
            plt.plot(costs, color=f'C{i}', label=label)
        # plt.show()

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
