import numpy as np



def preprocessor(X):
    for i in range(X.shape[1]):
        feature_mean = np.nanmean(X[:, i])
        X[np.isnan(X[:, i]), i] = feature_mean

    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    X = (X - min_val) / (max_val - min_val)

    X = np.c_[np.ones((X.shape[0], 1)), X]

    return X