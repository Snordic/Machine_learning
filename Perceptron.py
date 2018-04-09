"""
Нейронная сеть на основе персептрона
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """
    Параметры нейрона:
    eta : float 
            Темп обучения (между 0.0 и 1.0)
    n_iter : int
            Проходы по тренировочному набору данных
            
    Атрибуты
    w_ : 1-мерный массив
            Весовые коэфициенты после подгонки
    errors_ : список
            Число случаев ошибочной классификации в каждой эпохе
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        X : {массивоподобный}, форма = [n_samples, n_features]
            тренировачный векторы, где
            n_samples - число образцов
            n_features - числов признаков
        y : массивоподобный, форма = [n_samples]
            Целевые значения.
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def new_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        
        return np.where(self.new_input(X) >= 0.0, 1, -1)




Per = Perceptron
print(Per.__doc__)