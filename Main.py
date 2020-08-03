import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    pl = np.exp(x)
    mn = np.exp(-x)
    return (pl - mn) / (pl + mn)


class Model:
    inPut = 1
    Neuron = 3
    output = 1

    def __init__(self):
        self.Wx = np.random.random((self.Neuron, self.inPut + 1))
        self.Wby = np.random.random((self.Neuron, self.Neuron))
        self.back = np.zeros((self.Neuron, 1))
        self.Wy = np.random.random((self.output, self.Neuron + 1))
        self.epo = 50
        self.alpha = 0.001

    def predict(self, x):
        Iy = np.dot(self.Wx, np.array([[x], [1]]))
        Iy = Iy + np.dot(self.Wby, self.back)
        y = tanh(Iy)
        self.back = y
        z = np.dot(self.Wy, np.vstack((y, np.array([1]))))
        return z, y, Iy

    def fit(self, X, T):
        for epo in range(self.epo):
            for i in range(len(X)):
                for j in range(len(X[0])):
                    pre = self.predict(X[i][j])
                    z, y, Iy = pre[0], pre[1], pre[2]
                    deltaWy = ((T[i][j] - z) * np.vstack((y, np.array([1])))).T
                    deltaIy = np.delete(self.Wy, len(self.Wy), 1).T * (T[i][j] - z) * (
                            4 / (np.exp(Iy) + np.exp(-Iy)) ** 2)
                    deltaWx = deltaIy * np.array(([[X[i][j], 1]]))
                    deltaWyb = deltaIy * self.back.T
                    self.Wy -= self.alpha * deltaWy
                    self.Wx -= self.alpha * deltaWx
                    self.Wby -= self.alpha * deltaWyb


X = []
T = []
for k in range(7000):
    sita = np.random.randint(2, 5) * np.pi / 2
    X.append([np.sin(sita + i) for i in range(7)])
    T.append([np.sin(sita + i + 1) for i in range(7)])

plt.plot(X, linestyle='None', marker="o")
plt.plot(T, linestyle='None', marker="o")
s = np.linspace(0, 10, 100)

model = Model()
model.fit(X, T)
plt.plot(s, np.sin(np.pi + s))
plt.show()
