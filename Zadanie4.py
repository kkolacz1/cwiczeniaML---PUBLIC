# Zadanie4: Wyznacz ponizsze prawdopodobieństwa ręcznie na podstawie parametrów modelu.

from collections import Counter
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1) #generating data

print(X.shape, y.shape)
print(Counter(y))

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X, y)

print(model.coef_, model.intercept_) #parametry modelu

#nowa obserwacja
row = [1.89149379, -0.39847585, 1.63856893, 0.01647165, 1.51892395, -3.52651223, 1.80998823, 0.58810926, -0.02542177, -0.52835426]
#przewiduje jej klasę
yhat = model.predict([row])
print(f'Przewidziana klasa: {yhat[0]}')

#przewiduje prawdopodobienstwa
yhat = model.predict_proba([row])
print(f'Przewidziane prawdopodobieństwa: {yhat[0]}')
