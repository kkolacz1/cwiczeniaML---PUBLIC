# Zadanie1: Rozważ poniższe dane. Zastosuj algorytm SVM dla różnych parametrów C: 0.01, 10.
# Zwizualizuj i skomentuj w kilku zdaniach otzymane wyniki.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from mlxtend.plotting import plot_decision_regions



X, y = make_blobs(n_samples=200, random_state=1, n_features=2, centers = 2, cluster_std = 2.4)

#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

####################

# normalization of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# creating first model
svm = SVC(kernel='linear', C=0.01, random_state=0)
svm.fit(X_train_std, y_train)

# creating second model
svm2 = SVC(kernel='linear', C=10, random_state=0)
svm2.fit(X_train_std, y_train)

plt.figure(figsize=(8, 6))

X_all = np.vstack((X_train_std, X_test_std))
y_all = np.hstack((y_train, y_test))

plt.subplot(2,1,1)
plot_decision_regions(X=X_all, y=y_all, clf=svm)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Wykres SVM dla hiperparametru C = 0.01')
L = plt.legend(loc='upper left')
L.get_texts()[0].set_text('Blob A')
L.get_texts()[1].set_text('Blob B')

plt.subplot(2,1,2)
plot_decision_regions(X=X_all, y=y_all, clf=svm2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Wykres SVM dla hiperparametru C = 10')
L = plt.legend(loc='upper left')
L.get_texts()[0].set_text('Blob A')
L.get_texts()[1].set_text('Blob B')
plt.show()

# Answer: High values of C result in a tighter fit to the data in order to avoid missclassification,
# but also tend to overfit the resulting model.

# Zadanie2: Przetestować inne wartości (np. 10, 50) dla danych ponizej.
# Skomentować wyniki w konkteście definicji parametru gamma.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets

# generate data
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

############################################

# normalization of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# stitching data together
X_all = np.vstack((X_train_std, X_test_std))
y_all = np.hstack((y_train, y_test))


def plot_SVC(gamm, nplot):

    plt.subplot(3, 3, nplot)
    svm = SVC(kernel='rbf', C=1., gamma=gamm, random_state=0)
    svm.fit(X_train_std, y_train)

    plot_decision_regions(X=X_all, y=y_all, clf=svm)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title(f'SVC plot for gamma = {gamm}')


tested_gamma = [0.1, 0.5, 1, 2, 4, 8, 20, 40, 80]

for idx, gamm in enumerate(tested_gamma):
    plot_SVC(gamm, idx+1)

plt.tight_layout(pad=1.0)
plt.show()

# Comment: gamma parameter seems to influence how tightly the classification area is wrapped around its objects.

# Zadanie3: Uzupełnij definicję poniżej klasy (Momentum).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer

from typing import List


class Optimizer:
    """Base class for each optimizer"""

    def __init__(self, initial_params):
        # store model weights
        self.params = initial_params

    def step(self):
        """Updates the weights stored in self.params"""
        raise NotImplementedError()

    def zero_grad(self):
        """Torch accumulates gradients, so we need to clear them after every update"""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


class GradientDescent(Optimizer):

    def __init__(self, initial_params: List[torch.tensor], learning_rate):
        super().__init__(initial_params)
        self.learning_rate = learning_rate

    @torch.no_grad()
    def step(self):
        for param in self.params:
            param -= self.learning_rate * param.grad


class Momentum(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma):
        super().__init__(initial_params)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.oldUpdtVec = 0
        self.deltas = []

        for param in self.params:
            self.deltas.append(torch.zeros_like(param))

    @torch.no_grad()
    def step(self):
        for param in self.params:
            self.oldUpdtVec = self.gamma*self.oldUpdtVec + self.learning_rate * param.grad
            param -= self.gamma*self.oldUpdtVec + self.learning_rate * param.grad





# visualize_optimizer(Momentum, n_steps=20, learning_rate=0.05, gamma=0.8)
# plt.show()

# Zadanie5: Uzupełnij definicję poniżej klasy (RMSPROP).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List


class RMSProp(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.ht = torch.zeros_like(self.params[0])

    @torch.no_grad()
    def step(self):
        for param in self.params:
            self.ht = self.gamma*self.ht + (1-self.gamma)*torch.pow(param.grad, 2)
            param -= (self.learning_rate / (self.ht + self.epsilon) ** 0.5) * param.grad



visualize_optimizer(RMSProp, n_steps=10, learning_rate=0.5, gamma=0.9, epsilon=1e-8)
plt.show()

# Zadanie6: Uzupełnij definicję poniżej klasy (Adadelta).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List


class Adadelta(Optimizer):

    def __init__(self, initial_params, gamma, epsilon):
        super().__init__(initial_params)
        self.gamma = gamma
        self.epsilon = epsilon
        self.ht = torch.zeros_like(self.params[0])
        self.dt = torch.zeros_like(self.params[0])

    @torch.no_grad()
    def step(self):
        for param in self.params:
            self.ht = self.gamma*self.ht + (1-self.gamma)*torch.pow(param.grad, 2)
            term = ((self.dt + self.epsilon)**0.5 / (self.ht + self.epsilon) ** 0.5) * param.grad
            param -= term
            self.dt = self.gamma*self.dt + (1-self.gamma)*term**2



visualize_optimizer(Adadelta, n_steps=20, epsilon=5e-2, gamma=0.9)
plt.show()

# Zadanie7: Uzupełnij definicję poniżej klasy (Adadelta).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List


class Adam(Optimizer):

    def __init__(self, initial_params, learning_rate, beta1, beta2, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mi = torch.zeros_like(self.params[0])
        self.vi = torch.zeros_like(self.params[0])
        self.count = 0


    @torch.no_grad()
    def step(self):
        for idx, param in enumerate(self.params):
            self.mi = self.beta1*self.mi + (1-self.beta1)*param.grad
            self.vi = self.beta2*self.vi + (1-self.beta2)*torch.pow(param.grad, 2)
            licznik = self.mi / (1-self.beta1**(self.count+1))
            mianownik_part = self.vi / (1-self.beta2**(self.count+1))
            param -= self.learning_rate * licznik / (mianownik_part**0.5 + self.epsilon)
            self.count += 1




visualize_optimizer(Adam, n_steps=20, learning_rate=0.35, beta1=0.9, beta2=0.999, epsilon=1e-8)
plt.show()