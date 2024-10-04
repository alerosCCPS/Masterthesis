# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:13:24 2024

Check_circle
"""

import os, sys
#sys.path.append(os.path.join(os.path.dirname(__file__), "mpc"))
from new_ocp import MPC_Fine
#from visualization import SimPlotter, ResultePlotter
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt
from hamster_dynamic_nobeta import get_hamster_model


import numpy as np
plt.close("all")

#import torch
def reset_nlp(MPC):
    MPC.solver = None
    MPC.J = 0
    MPC.g = [] # currently forced to zero later (system dynamic constraints; multiple shooting)
    MPC.lbg = []
    MPC.ubg = []
    MPC.lbx = []
    MPC.ubx = []
    MPC.U_opt = []
    MPC.X_opt = []
    MPC.X = ca.MX.sym("X", MPC.nx, MPC.N + 1)
    MPC.U = ca.MX.sym("U", MPC.nu, MPC.N)

    MPC.setup_nlp()

    return MPC



path_name = 'test_traj_mpc'
mpc = MPC_Fine(path_name)
mpc.No_track= True
mpc.simple_mode = True
mpc.kappa_ref = 1
mpc.N = 10
mpc.x_init = [0,0,0,0.6]

mpc = reset_nlp(mpc)
mpc.sim()
mpc.plot_closed_loop()
# a = mpc.X_opt
# b = mpc.U_opt
# add open loop simulation
hamster, constraints = get_hamster_model(path_name)

x_now = mpc.x_init
X_ol = [x_now]
for k in range(100):
    
    x_now = x_now + hamster.dynamic(x_now, [ ca.atan(0.25*mpc.kappa_ref), 0.6 ], mpc.kappa_ref) / mpc.controller_freq
    
    X_ol.append(x_now.full())
    
    
    
