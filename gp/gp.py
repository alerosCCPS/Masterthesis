import torch
import numpy as np
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.means import  ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import os
import pandas as pd
import matplotlib.pyplot as plt

Script_Root = os.path.abspath(os.path.dirname(__file__))


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class Trainer:

    def __init__(self, case_name='test_traj'):
        self.data_type = torch.float32
        self.learning_rate = 0.01
        self.data_root = os.path.join(Script_Root, "DATA", case_name)
        self.train_x, self.train_y = self.load_data(os.path.join(self.data_root, 'train_data.csv'))
        self.test_x, self.test_y = self.load_data(os.path.join(self.data_root, 'test_data.csv'))

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 0.2)).to(self.data_type)
        self.model = GP(self.train_x, self.train_y, self.likelihood).to(self.data_type)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.training_iterations = 120
        print("completed initialization")

    def process(self):
        self.train(self.model, self.optimizer, self.likelihood, self.mll, self.train_y)
        print("training completed")
        self.test()
        print("testing completed")
        self.save_model()

    def test(self):
        print("start testing model")
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            pred_u = self.likelihood(self.model(self.test_x))
            u = pred_u.mean.numpy()
            lower, upper = pred_u.confidence_region()
            e = u - self.test_y.numpy()

            heads = [ 'delta','e_delta', 'lower', 'upper']
            df = pd.DataFrame(data=np.hstack((u.reshape(-1,1), e.reshape(-1,1), lower.numpy().reshape(-1,1), upper.numpy().reshape(-1,1))), columns=heads)
            df.to_csv(os.path.join(self.data_root, 'gp_test_results.csv'), index=False)
            self.simple_plot(e, 'error ')
            self.compare_plot(u, self.test_y.numpy(), "compare delta and label")
            self.plot_pred(u,lower.numpy(), upper.numpy())

    def plot_pred(self, pred, lower, upper):
        x = [i for i in range(len(pred))]
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.plot(x, pred, color='red', label='predic')
        ax.fill_between(x, lower, upper, alpha=0.5, label='Confidence')
        # ax.set_ylim(-1,1)
        plt.legend()
        plt.savefig(os.path.join(self.data_root,'pred_plot.png'))
        plt.show()
    def compare_plot(self,u,u_true,name):
        the_x = [i for i in range(len(u))]
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        ax.plot(the_x, u, color='blue', label="pred")
        ax.plot(the_x, u_true, color='red', label='label')
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(self.data_root,f'{name}.png'))
        plt.show()
    def simple_plot(self, data, name):
        x = [i for i in range(len(data))]
        value = sum(data)/len(data)
        aver = [value for _ in range(len(data))]
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        ax.plot(x, data)
        ax.plot(x, aver,label=f'Average: {value}', color='r')
        plt.title(name)
        plt.savefig(os.path.join(self.data_root, f'{name}.png'))
        plt.show()

    def train(self, model, optimizer, likelihood, mll, label):
        print(f"start training for {self.training_iterations} iterations")
        model.train()
        likelihood.train()
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = model(self.train_x)
            loss = -mll(output, label)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iterations, loss.item()))
            optimizer.step()

    def save_model(self):
        # 保存模型和似然的状态字典
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict()
        }, os.path.join(self.data_root, 'gp_model.pth'))
        print(f"saved model at {self.data_root}")

    def load_data(self, path):
        df = pd.read_csv(path)
        kappa, n, alpha, v, delta = df['curvature'].values.reshape((-1, 1)), \
            df['n'].values.reshape((-1, 1)), \
            df['alpha'].values.reshape((-1, 1)), \
            df['v'].values.reshape((-1, 1)), \
            df['delta'].values.reshape((-1, 1))
        x = torch.tensor(np.hstack((kappa, n, alpha, v))).to(self.data_type)
        y = torch.tensor(delta).to(self.data_type).squeeze_()
        return x, y


if __name__ == '__main__':
    trainer = Trainer()
    trainer.process()