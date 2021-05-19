import torch
import pandas as pd
import numpy as np
import os
from models import WHNet

import matplotlib

import finite_ntk as finite_ntk
import finite_ntk.utils as utils
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    # exact Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model, use_linearstrategy=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(
            model=model, use_linearstrategy=use_linearstrategy
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    model_name = "model_WH"

    # Settings
    n_b = 8
    n_a = 8

    # Column names in the dataset
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y_meas = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype=np.float32).item()
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts

    N_load = 1000
    y_meas = y_meas[:N_load, [0]]
    u = u[:N_load, [0]]

    # In[Instantiate models]

    # Create models
    model = WHNet()

    model_folder = os.path.join("models", model_name)
    # Create model parameters
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pkl")))


    # In[Predict]

    transfer_u = torch.tensor(u[None, :, :])
    transfer_y = torch.tensor(y_meas[None, :, :])
    sim_y = model(transfer_u)

    # construct likelihood and gp model
    gplh = gpytorch.likelihoods.GaussianLikelihood()
    gpmodel = ExactGPModel(
        transfer_u, transfer_y, gplh, model, use_linearstrategy=False #args.fisher
    )

    # set noise to be smaller
    print("residual error: ", torch.mean((model(transfer_u) - sim_y) ** 2))

    with torch.no_grad():
        gplh.noise = torch.max(1e-3 * torch.ones(1), torch.mean((model(transfer_u) - sim_y) ** 2))
        print("noise is: ", gplh.noise)

    gpmodel.train()
    loss = gplh(gpmodel(transfer_u)).log_prob(transfer_y)