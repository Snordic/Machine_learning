from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from io import StringIO


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # настроить генератор маркеров и палитру
    markers = ('s', 'x', '^', 'o', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='yellow',
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='тестовый набор')



# # Создаем набор данных
# X_xor = np.random.randn(200, 2)
# y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# y_xor = np.where(y_xor, 1, -1)
#
# plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
#             c='b', marker='x', label='1')
# plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
#             c='r', marker='s', label='-1')
# plt.ylim(-3.0)
# plt.legend()
# plt.show()
#
# # Создание НС на основе ядра из функций радиального базиса
# svm = SVC(kernel='rbf', gamma=0.2, C=1.0, random_state=0)
# svm.fit(X_xor, y_xor)
#
# plot_decision_regions(X=X_xor, y=y_xor, classifier=svm)
# plt.title('НС на основе ядра из функций радиального базиса)')
# plt.legend(loc='upper left')
# plt.show()
csv_data = '''A, B, C, D
            1.0, 2.0, 3.0, 4.0
            5.0, 6.0,, 8.0
            10.0, 11.0, 12.0,'''
df = pd.read_csv(StringIO(csv_data))
print(df)
print('')
# print(df.isnull().sum())
# print(df.dropna())
imr = Imputer(missing_values='NaN', strategy='mean', axis=1)
imr = imr.fit(df)
impured_date = imr.transform(df.values)
print(impured_date)
print('')

dv = pd.DataFrame([
    ['зеленый', 'M', 10.1, 'класс1'],
    ['красный', 'L', 13.5, 'класс2'],
    ['синий', 'XL', 15.3, 'класс1']
])

dv.columns = ['Цвет', 'Размер', 'Цена', 'Метка']
size_mapping = {
    'XL': 3,
    'L' : 2,
    'M' : 1
}
print(dv)
print('')

dv['Размер'] = dv['Размер'].map(size_mapping)
print(dv)
print('')

inv_size_mapping = {v: k for k, v in size_mapping.items()}
dv['Размер'] = dv['Размер'].map(inv_size_mapping)
print(dv)
print('')