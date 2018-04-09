from Perceptron import Perceptron
from AdalineGD import AdalineGD
from AdalineSGD import AdalineSGD
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
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


direct = 'data\iris\iris.data'
df = pd.read_csv(direct, header=None)
df.tail()

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

# Масштабирования признаков, принимает свойство нормального распрделения
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,y)
# ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
# ax[0].set_xlabel('Эпохи')
# ax[0].set_ylabel('log(Сумма квадртичных ошибок)')
# ax[0].set_title('Adaline (темп обучения 0.01)')
# ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
# ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
# ax[1].set_xlabel('Эпохи')
# ax[1].set_ylabel('log(Сумма квадртичных ошибок)')
# ax[1].set_title('Adaline (темп обучения 0.0001)')
# plt.show()


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X_std,y)
plot_decision_regions(X_std, y, classifier=ada)
plt.xlabel('длина чашелистика [стандартизованная]')
plt.ylabel('длина лепистка [стандартизованная]')
plt.title('Adaline (стохастический градиентный спуск)')
plt.legend(loc='upper left')
plt.show()
plt.close()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Средняя стоимость)')
plt.show()


# # Рисуем графики
# plt.scatter(X[:50, 0], X[:50, 1],
#             color='red', marker='o', label='щетинистый')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='blue', marker='x', label='разноцветный')
# plt.xlabel('длина чашелистика')
# plt.ylabel('длина лепестка')
# plt.legend(loc='upper left')
# # plt.show()
# plt.close()
#
# # Включаем нейрон
# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)
#
# # Отображаем число ошибок
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Эпохи')
# plt.ylabel('Число случаев ошибчной классификаций')
# plt.show()
# plt.close()
#
# plot_decision_regions(X, y, classifier=ppn)
# plt.xlabel('длина чашелистика [см]')
# plt.ylabel('длина лепестка [см]')
# plt.legend(loc='upper left')
# plt.show()
