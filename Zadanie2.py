# Zadanie2: Wybierz dowolny zbiór danych lub wygeneruj syntetyczne dane. Zastosuj model regresji liniowej z regularyzacją L1.
#Na podstawie zbioru walidacyjnego wybierz optymalne parametry ni oraz alfa (rozważ chociaż 15-20 różnych kombinacji).

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from Zadanie1 import LinearModel2v

# imports ML data from boston dataset
boston_data = load_boston()
d = boston_data['data']
d2 = d[:, [2,5]] #wyciągamy tylko 2 cechy: INDUS, RM
target = boston_data['target']

# split dataset into training and validation
X_train, X_test, y_train, y_test = train_test_split(d2, target, test_size=0.4, random_state=42)
X_walid, X_test, y_walid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# define testing values for ni and alfa
tested_ni = list(np.logspace(-5, 0, num=30))
tested_alfa = list(np.logspace(-10, 1, num=10))

# create a combination array, which will allow to create a 3d optimization map
combo = list(itertools.product(tested_ni, tested_alfa))

# test combinations of parameters, and save loss_function into 'result' array
# code will result in overflow in cases where linear model does not converge
result = []
for i in combo:
    model = LinearModel2v(eta=i[0], alfa=i[1])
    model.train(X_walid, y_walid)
    result.append(model.loss_function(X_walid, y_walid))

# print the combination of parameters which achieved the best score
print(f'Najmniejsza wartosc funkcji kosztu otrzymano dla eta = {combo[result.index(min(result))][0]}, alfa = {combo[result.index(min(result))][1]} ')

# plot the 3d surface map
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

for idx, x in enumerate(result):
    ax.scatter(np.log10(combo[idx][0]), np.log10(combo[idx][1]), result[idx])

plt.xlabel('log10 eta')
plt.ylabel('log10 alfa')
ax.set_zlabel('Wartosc funkcji kosztu')
plt.show()