# Zadanie3: Uzasadnij ponizsze wartości prawdopodobieństw w oparciu o parametry modelu.

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

model = linear_model.LogisticRegression()

X = np.array([[1,2],[2,3],[4,5],[1,-4],[5,-7],[-3,-1]])
y = np.array([0, 0, 0, 1, 1, 1])

model.fit(X, y)

# model.coef_ Wspolczynniki kierunkowe
# output = array[-0.39682866, -0.82497163]

# model.intercept_ Wyraz wolny
# output = array([0.11932619]

# model.predict([[3,3]]) Predykcja klasy
# output = array[0]

# model.predict_proba([3,3]) Obliczone prawdopodobienstwa
# output = array([[0.97197068, 0.02802932]]))

###########################

# I'm not sure what's the best way to justify prediction output. I guess I'll do it by calculating the output manually?
X1 = 3
X2 = 3
print(f'obliczone prawdopodobienstwo otrzymania \'1\' dla podanych X1 X2 to: {1/(1 + math.exp(-(X1*-0.39682866 + X2*-0.82497163 + 0.11932619)))}')
# plt.scatter(X[:,0],X[:,1] )
# plt.show()