# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:19:45 2024

@author: rose
See what steering each model puts out, if n and alpha are zero
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "mpc"))
from mpc.mpc_controller import MPC
#from visualization import SimPlotter, ResultePlotter
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt

import torch
import numpy as np
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.means import  ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

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

    MPC.setup_nlp()

    return MPC


kappa_test = np.linspace(-2,2,10)
# ' MPC '
path_name = 'test_traj_mpc'
mpc = MPC(path_name)
mpc.No_track= True
mpc.simple_mode = True
mpc.kappa_ref = 1
mpc.N = 10
mpc.x_init = [0,0,0,0.6] # start on the track with desired ref vel (currently fixed to 0.6)
# mpc.Q = np.diag([0., 10, 10, 1e-1])
# mpc.QN = np.diag([0., 10, 10, 1e-1])
# mpc.Q = np.diag([0., 0, 10, 1e-1])
# mpc.QN = np.diag([0., 0, 10, 1e-1])
# setup with "new" conditions

# mpc = reset_nlp(mpc)
# mpc.sim() # maybe it is slightly off, i am not sur eabout the solution at the start of the closed loop sim
# X_cl = mpc.X_opt
# U_cl = mpc.U_opt
# x0x = ca.repmat(mpc.x_init,(mpc.N + 1))
# x0u = ca.repmat([0.6 ,0.25*mpc.kappa_ref],mpc.N)
# x0 = ca.vertcat(x0x,x0u)

u_steer_mpc = []
for n in range(len(kappa_test)):
    mpc.kappa_ref = kappa_test[n]
    x0x = ca.repmat(mpc.x_init,(mpc.N + 1))
    x0u = ca.repmat([0.6 ,0.25*mpc.kappa_ref],mpc.N)
    x0 = ca.vertcat(x0x,x0u)
    mpc = reset_nlp(mpc)

    x0_res, con, cal_time, status, U_OCP, X_OCP = mpc.predict(x0, mpc.x_init)
    u_steer_mpc.append(con[1])
    
    
plt.figure(1)
plt.plot(kappa_test,np.array(u_steer_mpc),label = 'OCP')
plt.xlabel('kappa')
plt.ylabel('steering')
plt.title('n,alpha=0')
    
'2d GP'
class GP_2d(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP_2d, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        # self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
state_dict = torch.load('gp/DATA/test_traj_2D/gp_2D.pth')
likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 0.1)).to(torch.float32)
#self.train_x, self.train_y = self.load_data(os.path.join(self.data_root, 'train_data.csv'))
data_path = 'gp/DATA/test_traj_2D/train_data.csv'
df = pd.read_csv(data_path)
n, alpha, rec_diff = df['n'].values.reshape((-1, 1)), \
df['alpha'].values.reshape((-1, 1)), \
df['rec_diff'].values.reshape((-1, 1))
train_x = torch.tensor(np.hstack((n, alpha))).to(torch.float32)
train_y = torch.tensor(rec_diff).to(torch.float32).squeeze_()


model = GP_2d(train_x, train_y, likelihood)
model.eval()
pred_u = model.likelihood(model(torch.tensor([[0,0]])))
u = pred_u.mean.detach().numpy() # not exactly zero

u_steer_2D = 0.25*kappa_test + u
plt.figure(1)
plt.plot(kappa_test,u_steer_2D,label = '2D_GP')
plt.xlabel('kappa')
plt.ylabel('steering')
plt.title('n,alpha=0')



'3D GP'
class GP_3d(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP_3d, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        # self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

#DATA/test_traj_3D/gp_3D.pth
state_dict = torch.load('gp/DATA/test_traj_3D/gp_3D.pth')
likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 0.1)).to(torch.float32)
#self.train_x, self.train_y = self.load_data(os.path.join(self.data_root, 'train_data.csv'))
data_path = 'gp/DATA/test_traj_3D/train_data.csv'
df = pd.read_csv(data_path)
kappa, n, alpha, delta = df['curvature'].values.reshape((-1, 1)), \
df['n'].values.reshape((-1, 1)), \
df['alpha'].values.reshape((-1, 1)), \
df['delta'].values.reshape((-1, 1))
train_x = torch.tensor(np.hstack((kappa,n, alpha))).to(torch.float32)
train_y = torch.tensor(delta).to(torch.float32).squeeze_()


model = GP_3d(train_x, train_y, likelihood)
model.eval()

u_steer_3d = []

for n in range(len(kappa_test)):
    pred_u = model.likelihood(model(torch.tensor([[float(kappa_test[n]),0,0]])))
    u_steer_3d.append(pred_u.mean.detach().numpy())


u_steer_3d=np.array(u_steer_3d)   

plt.plot(kappa_test,u_steer_3d,label='3D_GP')
plt.legend()
plt.show()


