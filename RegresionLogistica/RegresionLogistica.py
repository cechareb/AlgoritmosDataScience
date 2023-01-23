import numpy as np

def sigmoide(x):
    return 1/(1+np.exp(-x))

class RegresionLogistica():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def ajusta(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            prediccion_linear = np.dot(X, self.weights) + self.bias
            predicciones = sigmoide(prediccion_linear)

            dw = (1/n_samples) * np.dot(X.T, (predicciones - y))
            db = (1/n_samples) * np.sum(predicciones - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predice(self, X):
        prediccion_linear = np.dot(X, self.weights) + self.bias
        y_pred = sigmoide(prediccion_linear)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred