import os
from scipy.interpolate import splrep, splev
import pandas as pd
import casadi as ca
import numpy as np
Script_Root = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(Script_Root, "DATA", 'test_traj')

def get_f():
    df = pd.read_csv(os.path.join(data_path, 'path.csv'))
    tck = splrep(df['s'].values, df['curvature'].values)
    def inter(value):
        return splev(value, tck)
    return inter
def get_ca_f():
    df = pd.read_csv(os.path.join(data_path, 'path.csv'))
    s = df["s"].values
    curvature = df['curvature'].values
    interpolator = ca.interpolant("kappa_at_s", "bspline", [s], curvature)
    return interpolator
deg2R = lambda x: ca.pi*(x/180)
R = 0.25/np.tan(deg2R(27) )
print(R, 1/R)

# s = ca.MX.sym("s")
# kappa = ca_f(s)

