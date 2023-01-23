import numpy as np
from collections import Counter

def distancia_euclideana(x1, x2):
    distancia = np.sqrt(np.sum((x1-x2)**2))
    return distancia

class KNN:
    def __init__(self, k=3):
        self.k = k

    def ajusta(self, X, y):
        self.X_train = X
        self.y_train = y

    def predice(self, X):
        predicciones = [self._predice(x) for x in X]
        return predicciones

    def _predice(self, x):
        # medir las distancias
        distancias = [distancia_euclideana(x, x_train) for x_train in self.X_train]

        # obtener el k mas cercano
        k_indices = np.argsort(distancias)[:self.k]
        k_cercanos_etiquetas = [self.y_train[i] for i in k_indices]

        # voto mayoritario
        mas_comun = Counter(k_cercanos_etiquetas).most_common()
        return mas_comun[0][0]