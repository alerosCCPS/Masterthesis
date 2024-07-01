import numpy as np
from scipy import signal, optimize
import matplotlib.pyplot as plt
from utils import check_path
import sys
import os
import glob
import pandas as pd
import json
Script_Root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(Script_Root, ".."))

class Identifier:

    def __init__(self, cutoff_frequency=5, order_filter=4, K=1):
        self.x = np.ndarray
        self.y = np.ndarray
        self.filtered = np.ndarray
        self.sampling_freq = 1000  # Hz
        self.fft_freq, self.fft_results = [], []
        self.threshold = 0.6
        self.cutoff_frequency = cutoff_frequency
        self.order_filter = order_filter
        self.K = K
        self.theta = 0

    def load_data(self, x, y):
        self.x, self.y = x, y
        # self.sampling_freq = 1/(self.x[1] - self.x[0])

    def fft(self):
        # print(self.sampling_freq)
        fft_results = np.abs(np.fft.fft(self.y))
        fft_freq = np.fft.fftfreq(len(self.y), self.sampling_freq**-1)
        data = list(zip(fft_freq, fft_results))
        data = sorted(data, key=lambda x: x[0])
        data = list(zip(*data))
        self.fft_freq, self.fft_results = data[0], data[1]

    def butterworth_filtering(self):
        b, a = signal.butter(self.order_filter, self.cutoff_frequency/(0.5*self.sampling_freq), 'low')
        self.filtered = signal.filtfilt(b, a, self.y)

    def fit(self):
        # self.butterworth_filtering()
        self.filtered = self.y
        def f(x, theta):
            return self.K*(1 - np.exp(-theta*x))

        popt, pcov = optimize.curve_fit(f, self.x, self.filtered)
        self.theta = 1/popt[0]
        print(f"Estimated tau: {self.theta}")

    def find_cutoff_frequency(self):
        length = len(self.fft_freq)
        start = int(length/2) if self.fft_freq[int(length/2)] > 0 else int(length/2) + 1
        sum = 0
        step = self.fft_freq[1] - self.fft_freq[0]
        for index in range(start, len(self.fft_freq)):
            sum += step * self.fft_results[index]
        threshold = 0
        for index in range(start, len(self.fft_freq)):
            threshold += step * self.fft_results[index]/sum
            if threshold >= self.threshold:
                self.cutoff_frequency = self.fft_freq[index]
                break
        # print(sum, threshold)
        # print(self.cutoff_frequency)


    def plot_fft(self):

        plt.figure(figsize=(8,6))
        plt.plot(self.fft_freq, self.fft_results)
        plt.title("Signal Spectrum")
        plt.xlabel("frequency")
        plt.ylabel("magnitude")
        plt.grid(True)
        plt.xlim(-15, 15)
        plt.ylim(0, 25)
        plt.show()

    def plot_filtered(self):

        plt.figure(figsize=(8,6))
        plt.plot(self.x, self.filtered)
        plt.title("filtered signal")
        plt.xlabel("time")
        plt.ylabel("magnitude")
        plt.grid(True)
        plt.show()

def process():
    data_path = os.path.join(Script_Root, 'DATA')
    data_list = glob.glob(os.path.join(data_path, '*.csv'))
    print(data_list)
    save_path = os.path.join(Script_Root, 'results')
    check_path(save_path)
    results = {}
    results_list = []
    for d in data_list:
        df = pd.read_csv(d)
        K = df["K"][0]
        x = np.array(list(df["time_steps"]))
        y = np.array(list(df["velocity"]))
        y = y - y[0]
        # print(y)
        identifier = Identifier(K=K)
        identifier.load_data(x, y)
        identifier.fit()
        result = identifier.theta
        results_list.append(result)
        results[d] = result
    results["average"] = sum(results_list) / len(results_list)

    with open(os.path.join(save_path, "results.json"), 'w')as f:
        json.dump(results, f, indent=4)



if __name__ == '__main__':
    # Iden = Identifier()
    process()