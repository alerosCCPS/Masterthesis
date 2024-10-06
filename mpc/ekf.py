import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
import os

Script_Root = os.path.abspath(os.path.dirname(__file__))

class EKF:

    def __init__(self, casename='circle'):
        self.path_data = os.path.join(Script_Root, "DATA", casename, 'path.csv')
        self.kappa_of_s = self.load_path_data()
        self.length_front = 0.125  # m
        self.length_rear = 0.125  # m
        self.T = 0.00984
        self.P = np.diag([1,1,1,1])
        self.F = np.zeros((4,4))
        self.H = np.diag([1]*4)
        self.x = np.zeros(4)
        self.u = np.zeros(2)
        self.z = np.zeros(4)
        self.Q = np.diag([0,0,0,0])
        self.R = np.diag([0,0.05,0,0])

    def update_F(self, x, u):
        s, n, alpha, v = x[0], x[1], x[2], x[3]
        v_command, delta = u[0], u[1]
        kappa = self.kappa_of_s(s)
        beta = 0.5 * delta

        self.F[0,1] = kappa*v*np.cos(alpha + beta) / (1 - n*kappa)**2
        self.F[0,2] = -v*np.sin(alpha + beta) / (1 - n*kappa)
        self.F[0,3] = np.cos(alpha + beta) / (1 - n*kappa)

        self.F[1,2] = v * np.cos(alpha + beta)
        self.F[1,3] = np.sin(alpha + beta)

        self.F[2,1] = -kappa**2*v*np.cos(alpha + beta)/(1 - n*kappa)**2
        self.F[2,2] = kappa*v*np.sin(alpha + beta) / (1 - n*kappa)
        self.F[2,3] = np.sin(beta)/self.length_rear - kappa*np.cos(alpha+beta)/(1-n*kappa)

        self.F[3,3] = -v_command/self.T

    def predict(self, x, u):
        self.x = np.array(x)
        self.u = np.array(u)
        self.update_F(self.x, self.u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        self.z = np.array(z)
        K_k = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R + np.diag([1e-5]*4)))
        self.x = self.x + np.dot(K_k, (self.z - self.x))
        self.P = np.dot((np.eye(4) - np.dot(K_k, self.H)), self.P)
    def process(self, x, u, z):
        self.predict(x,u)
        self.update(z)
        return self.x

    def load_path_data(self):
        df = pd.read_csv(self.path_data)
        s = df['s'].values
        curvature = df['curvature'].values
        tck= splrep(s, curvature)

        def interp(s):
            return splev(s, tck=tck)

        return interp
if __name__ == "__main__":
    noise = np.random.normal(0, 0.01)
    print(noise)