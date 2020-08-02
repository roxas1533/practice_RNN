import numpy as np


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

    def predict(self, x):
        Iy = np.dot(self.Wx, np.array([[x], [1]]))
        Iy = Iy + np.dot(self.Wby, self.back)
        y = tanh(Iy)
        self.back = y
        z = np.dot(self.Wy, np.vstack((y, np.array([1]))))
        return z, y


model = Model()
model.predict(1)
