from model_identify import Identifier
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


class Data_Generator:

    def __init__(self):
        self.sys = signal.TransferFunction([1], [0.5, 1])
        self.t = np.linspace(0,5,500)
        _, self.y = signal.step(self.sys, T=self.t)
        self.measurement = np.ndarray
        self.mean, self.variance = 0, 0.1

    def add_gaussian(self):
        noise = np.random.normal(self.mean, self.variance, self.y.shape)
        self.measurement = self.y + noise


    def plot_original(self):

        plt.figure(figsize=(8,6))
        plt.plot(self.t, self.y)
        plt.title("ideal step response")
        plt.xlabel("time")
        plt.ylabel("magnitude")
        plt.grid(True)
        plt.show()

    def plot_measurement(self):

        plt.figure(figsize=(8,6))
        plt.plot(self.t, self.measurement)
        plt.title("measurement")
        plt.xlabel("time")
        plt.ylabel("magnitude")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    DATA = Data_Generator()
    DATA.plot_original()
    DATA.add_gaussian()
    DATA.plot_measurement()

    testOrder = {}
    for order in range(2, 9):
        Iden = Identifier(cutoff_frequency=5, order_filter=order)
        Iden.load_data(DATA.t, DATA.measurement)
        Iden.fit()
        testOrder[order] = abs(Iden.theta - 0.5)
    data = list(zip(testOrder.keys(), testOrder.values()))
    data = sorted(data, key=lambda x: x[1])
    print("test best order is ", data[0][0])

    test_cutoff = {}
    for f in range(3, 9):
        Iden = Identifier(cutoff_frequency=f, order_filter=data[0][0])
        Iden.load_data(DATA.t, DATA.measurement)
        Iden.fit()
        testOrder[f] = abs(Iden.theta - 0.5)
    data = list(zip(testOrder.keys(), testOrder.values()))
    data = sorted(data, key=lambda x: x[1])
    print("test best f is ", data[0][0])
    #test best order is  4  test best f is  5