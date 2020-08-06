import numpy as np
import matplotlib.pyplot as plt
import copy


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    pl = np.exp(x)
    mn = np.exp(-x)
    return (pl - mn) / (pl + mn)


class Model:
    inPut = 1
    Neuron = 5
    output = 1

    def __init__(self):
        self.epo = 30
        self.alpha = 0.05
        self.loss = []
        self.historyLoss = []
        self.n_input = self.inPut
        self.n_hidden = self.Neuron
        self.n_output = self.output
        self.hidden_weight = np.random.randn(self.Neuron, self.inPut + 1)
        self.output_weight = np.random.randn(self.output, self.Neuron + 1)
        self.recurr_weight = np.random.randn(self.Neuron, self.Neuron + 1)

    def predict(self, x, y):
        Iy = np.dot(self.hidden_weight, np.array([[x], [1]]))
        Iy = Iy + np.dot(self.recurr_weight, np.vstack((y, np.array([1]))))
        y = sigmoid(Iy)
        z = np.dot(self.output_weight, np.vstack((y, np.array([1]))))
        return y.reshape(-1, 1), tanh(z).reshape(-1, 1)

    def __forward(self, x, z):
        r = self.recurr_weight.dot(np.hstack((1.0, z)))
        z = sigmoid(self.hidden_weight.dot(np.hstack((1.0, x))) + r)
        y = tanh(self.output_weight.dot(np.hstack((1.0, z))))
        return z, y

    def __forward_seq(self, X):
        z = np.zeros(self.n_hidden)
        zs, ys = ([], [])
        for x in X:
            z, y = self.__forward(x, z)
            zs.append(z)
            ys.append(y)
        return zs, ys

    def fit(self, Xl, T):
        for epo in range(self.epo):
            deltaLoss = 0
            print(epo)
            for X in np.random.permutation(Xl):
                tau = X.shape[0]
                temp = []
                rY, rZ = self.__forward_seq(X)
                deltaIy = np.zeros(self.n_hidden)
                SdWy = np.zeros(self.output_weight.shape)
                SdWx = np.zeros(self.hidden_weight.shape)
                SdWby = np.zeros(self.recurr_weight.shape)
                for t in range(tau - 1)[::-1]:
                    deltaO = (rZ[t] - X[t + 1][0]) * (1.0 - rZ[t] ** 2)
                    deltaWy = deltaO.reshape(-1, 1) * np.hstack((1.0, rY[t]))
                    deltaIy = (self.output_weight[:, 1:].T.dot(deltaO) + self.recurr_weight[:, 1:].T.dot(
                        deltaIy)) * rY[t] * (1.0 - rY[t])
                    deltaWx = deltaIy.reshape(-1, 1) * np.hstack((1.0, X[t][0]))
                    deltaWyb = deltaIy.reshape(-1, 1) * np.vstack((rY[t - 1] if t > 0 else np.zeros(self.n_hidden)))
                    SdWy += deltaWy
                    SdWx += deltaWx
                    SdWby += deltaWyb
                    temp.append(0.5 * (rZ[t] - X[t + 1][0]) ** 2)
                self.output_weight -= self.alpha * SdWy
                self.hidden_weight -= self.alpha * SdWx
                self.recurr_weight -= self.alpha * SdWby
                deltaLoss += np.mean(temp)
            self.loss.append(deltaLoss)
        plt.figure()
        plt.plot(self.loss)

    def predict_loop(self, X, times):
        zs, ys = self.__forward_seq(X)
        y, z = ys[-1], zs[-1]
        for i in range(times):
            z, y = self.__forward(y, z)
            zs.append(z)
            ys.append(y)

        return ys


X = []
T = []
N = 7000
pi = np.pi
div = 6
s = (np.random.rand(N) * pi).reshape(-1, 1)
e = s + np.random.randint(2, 5, N).reshape(-1, 1) * pi / 2

Xl = [np.linspace(_s, _e, int((_e - _s) / pi * div + 1)).reshape(-1, 1) for _s, _e in np.hstack((s, e))]
Xl = map(lambda X: np.sin(X), Xl)
st = np.random.rand() * pi
en = st + 20 * pi
n = int((en - st) / pi * div + 1)
x = np.linspace(st, en, n)
for k in range(7000):
    X.append([np.sin(0 + i * 0.1) for i in range(k, k + 5)])
    T.append([np.sin(0 + (i + 1) * 0.1) for i in range(k, k + 5)])

model = Model()
model.fit(list(Xl), T)
model.back = np.zeros((model.Neuron, 1))
plt.figure()
# plt.plot(s, np.sin(s))
m = 7
ys = model.predict_loop(np.sin(x[:m]), n - m)
ys = np.array(ys)[:, 0]
plt.plot(x[1:m], ys[1:m], '--o')
plt.plot(x[m - 1:], ys[m - 1:], '--o')
plt.show()
