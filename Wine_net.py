from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from SBS import SBS

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


direct = 'data\wine\wine.data'
df_wine = pd.read_csv(direct, header=None)
df_wine.columns = ['Метка класса', 'Алкоголь',
                   'Яблочная кислота', 'Зола',
                   'Целочность золы', 'Магний',
                   'Всего фенола', 'Флаваноиды',
                   'Фенолы нефлаваноидные', 'Проантоцианины',
                   'Интенсивность цвета', 'Оттенок',
                   'OD280/OD315 разбавленных вин', 'Пролин',
                   ]
print('Метка классов: ' , np.unique(df_wine['Метка класса']))
print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Масштабируем данные при помощи нормализаций
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

# Мастшабируем данные при помощи стандартизаций
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

# Установка штрафного параметра
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Верность на тренировочном наборе: ', lr.score(X_train_std, y_train))
print('Верность на тренировочном наборе: ', lr.score(X_test_std, y_test))

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.1])
# plt.ylabel('Верность')
# plt.xlabel('Число признаков')
# plt.grid()
# plt.show()

# Тренировка леса состоящий из 10 000 деревьев
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30 ,feat_labels[indices[f]], importances[indices[f]]))

# plt.title('Важность признаков')
# plt.bar(range(X_train.shape[1]), importances[indices], color='blue', align='center')
# plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.show()

# Создаем ковариационную матрицу
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Вычисляем кумулятивную сумму
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Сортировка собственных пар в порядке убывания
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
               for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# Создадим проекционную матрицу 13х2
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

# Преобразовываем весь 124х13 тренировачный набор
X_train_pca = X_train_std.dot(w)