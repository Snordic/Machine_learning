from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


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



# Загрузка данных для НС
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Распределение данных для тестирования и обучения НС
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Масштабирование признаков
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# Создание НС на основе Персептрона
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)


# Создание НС на основе логической регрессий
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)


# Создание НС на основе метода опороных векторов
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)


# Создание НС на основе ядра из функций радиального базиса
svm_2 = SVC(kernel='rbf', gamma=10.0, C=1.0, random_state=0)
svm_2.fit(X_train_std, y_train)


# Создание НС на основе деревьев решений
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)


# Создание НС на основе случайного леса
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=2, random_state=1)
forest.fit(X_train, y_train)


# Создание НС на основе обучение на примерах
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)


# Число ошибок в ходе обучения
y_pred_ppn = ppn.predict(X_test_std)
print('Число ошибочно классифицированных образцов для персептрона: %d' %(y_test != y_pred_ppn).sum())
y_pred_lr = lr.predict(X_test_std)
print('Число ошибочно классифицированных образцов для логической регрессий: %d' %(y_test != y_pred_lr).sum())
y_pred_svm = svm.predict(X_test_std)
print('Число ошибочно классифицированных образцов для метода опороных векторов: %d' %(y_test != y_pred_svm).sum())
y_pred_svm_2 = svm_2.predict(X_test_std)
print('Число ошибочно классифицированных образцов для метода ядра из функций радиального базиса: %d' %(y_test != y_pred_svm_2).sum())
y_pred_tree = tree.predict(X_test)
print('Число ошибочно классифицированных образцов для метода деревьев решений: %d' %(y_test != y_pred_tree).sum())
y_pred_forest = forest.predict(X_test)
print('Число ошибочно классифицированных образцов для метода случайного леса: %d' %(y_test != y_pred_forest).sum())
y_pred_knn = knn.predict(X_test_std)
print('Число ошибочно классифицированных образцов для метода обучение на примерах: %d' %(y_test != y_pred_knn).sum())


# Верность классификаций
print('Точность для персептрона: %.2f' % accuracy_score(y_test, y_pred_ppn))
print('Точность для логической регрессий: %.2f' % accuracy_score(y_test, y_pred_lr))
print('Точность для метода опороных векторов: %.2f' % accuracy_score(y_test, y_pred_svm))
print('Точность для метода ядра из функций радиального базиса: %.2f' % accuracy_score(y_test, y_pred_svm_2))
print('Точность для метода деревьев решений: %.2f' % accuracy_score(y_test, y_pred_tree))
print('Точность для метода случайного леса: %.2f' % accuracy_score(y_test, y_pred_forest))
print('Точность для метода обучение на примерах: %.2f' % accuracy_score(y_test, y_pred_knn))


X_combined_std = np.vstack((X_train_std, X_test_std))
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn)
plt.xlabel('длина чашелистика [стандартизованная]')
plt.ylabel('длина лепистка [стандартизованная]')
plt.title('НС')
plt.legend(loc='upper left')
plt.show()
