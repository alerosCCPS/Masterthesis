from gpytorch.distributions import MultivariateNormal
from gpytorch.means import  ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
import os
import torch
import gpytorch
import pandas as pd
import numpy as np

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        # self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class GP3D:

    def __init__(self, data_root):
        self.data_root = data_root
        self.data_type = torch.float32
        self.data_x, self.data_y = None, None
        self._load_data()

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 0.2)).to(self.data_type)
        self.model = GP(train_x=self.data_x, train_y=self.data_y, likelihood=self.likelihood).to(self.data_type)
        self._load_path()

    def _load_data(self):
        path = os.path.join(self.data_root, "train_data.csv")
        df = pd.read_csv(path)
        kappa, n, alpha, delta = df['curvature'].values.reshape((-1, 1)), \
            df['n'].values.reshape((-1, 1)), \
            df['alpha'].values.reshape((-1, 1)), \
            df['delta'].values.reshape((-1, 1))
        self.data_x = torch.tensor(np.hstack((kappa, n, alpha))).to(self.data_type)
        self.data_y = torch.tensor(delta).to(self.data_type).squeeze_()

    def _load_path(self):
        path = os.path.join(self.data_root, "gp_3D.pth")
        check_point = torch.load(path)
        self.likelihood.load_state_dict(check_point['likelihood_state_dict'])
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()
        self.likelihood.eval()
        print(f"loading from check point: {path}")

    def predict(self,x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            u = observed_pred.mean.numpy().item()
        return u

class GP3D_LF:

    def __init__(self, data_root):
        self.data_root = data_root
        self.data_type = torch.float32
        self.data_x, self.data_y = None, None
        self._load_data()

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 0.2)).to(self.data_type)
        self.model = GP(train_x=self.data_x, train_y=self.data_y, likelihood=self.likelihood).to(self.data_type)
        self._load_path()

    def _load_data(self):
        path = os.path.join(self.data_root, "train_data.csv")
        df = pd.read_csv(path)
        kappa, n, alpha, delta = df['curvature'].values.reshape((-1, 1)), \
            df['n'].values.reshape((-1, 1)), \
            df['alpha'].values.reshape((-1, 1)), \
            df['delta'].values.reshape((-1, 1))
        kappa_LF = df['look_fw_curvature'].values.reshape((-1, 1))
        self.data_x = torch.tensor(np.hstack((kappa,kappa_LF, n, alpha))).to(self.data_type)
        self.data_y = torch.tensor(delta).to(self.data_type).squeeze_()

    def _load_path(self):
        path = os.path.join(self.data_root, "gp_3D_LF.pth")
        check_point = torch.load(path)
        self.likelihood.load_state_dict(check_point['likelihood_state_dict'])
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()
        self.likelihood.eval()
        print(f"loading from check point: {path}")

    def predict(self,x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            u = observed_pred.mean.numpy().item()
        return u

class GP2D:

    def __init__(self, data_root):
        self.data_root = data_root
        self.data_type = torch.float32
        self.data_x, self.data_y = None, None
        self._load_data()

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 0.2)).to(self.data_type)
        self.model = GP(train_x=self.data_x, train_y=self.data_y, likelihood=self.likelihood).to(self.data_type)
        self._load_path()

    def _load_data(self):
        path = os.path.join(self.data_root, "train_data.csv")
        df = pd.read_csv(path)
        n, alpha, delta = df['n'].values.reshape((-1, 1)), \
            df['alpha'].values.reshape((-1, 1)), \
            df['delta'].values.reshape((-1, 1))
        self.data_x = torch.tensor(np.hstack((n, alpha))).to(self.data_type)
        self.data_y = torch.tensor(delta).to(self.data_type).squeeze_()

    def _load_path(self):
        path = os.path.join(self.data_root, "gp_2D.pth")
        check_point = torch.load(path)
        self.likelihood.load_state_dict(check_point['likelihood_state_dict'])
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()
        self.likelihood.eval()
        print(f"loading from check point: {path}")

    def predict(self,x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            u = observed_pred.mean.numpy().item()
        return u

class GP2D_LF:

    def __init__(self, data_root):
        self.data_root = data_root
        self.data_type = torch.float32
        self.data_x, self.data_y = None, None
        self._load_data()

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 0.2)).to(self.data_type)
        self.model = GP(train_x=self.data_x, train_y=self.data_y, likelihood=self.likelihood).to(self.data_type)
        self._load_path()

    def _load_data(self):
        path = os.path.join(self.data_root, "train_data.csv")
        df = pd.read_csv(path)
        n, alpha, delta = df['n'].values.reshape((-1, 1)), \
            df['alpha'].values.reshape((-1, 1)), \
            df['delta'].values.reshape((-1, 1))
        kappa_LF = df['look_fw_curvature'].values.reshape((-1, 1))
        self.data_x = torch.tensor(np.hstack((kappa_LF, n, alpha))).to(self.data_type)
        self.data_y = torch.tensor(delta).to(self.data_type).squeeze_()

    def _load_path(self):
        path = os.path.join(self.data_root, "gp_2D_LF.pth")
        check_point = torch.load(path)
        self.likelihood.load_state_dict(check_point['likelihood_state_dict'])
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()
        self.likelihood.eval()
        print(f"loading from check point: {path}")

    def predict(self,x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            u = observed_pred.mean.numpy().item()
        return u