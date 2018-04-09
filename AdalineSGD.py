from numpy.random import seed
import numpy as np


class AdalineSGD:
    """"
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
       shuffle : bool 
                Перемешивает тренировачные данные в каждой эпохе, если Тру,
                для предотвращения зацикливания.
       random_state : int 
                Инициализирует генератор случайных чисел
                для перемешивания и инициализаций весов
       """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """"Выполнить подгонку под тренированче данные
        без повторной инициализаций весов"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """"Применить обучающее правило ADALINE, чтобы обновить веса"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """"Расчитать чистый вход"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activate(self, X):
        """"Рассчитать линейную активацию"""
        return self.net_input(X)

    def predict(self, X):
        """"Вернуть метку класса после единичного скачка"""
        return np.where(self.activate(X) >= 0.0, 1, -1)


