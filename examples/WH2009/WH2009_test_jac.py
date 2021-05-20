import os
import matplotlib
import torch
import pandas as pd
import numpy as np
import functools

from models import WHNet
from util.extract_util import extract_weights, load_weights, f_par_mod_in


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

    # extract the parameters from the model in order to be able to take jacobians using the convenient functional API
    # see the discussion in https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240
    params, names = extract_weights(model)
    n_param = sum(map(torch.numel, params))

    # from module to function of the parameters only
    f_par = functools.partial(f_par_mod_in, param_names=names, module=model, inputs=transfer_u)
    f_par(*params)

    jacs = torch.autograd.functional.jacobian(f_par, params)
    jac_dict = dict(zip(names, jacs))

    with torch.no_grad():
        y_out_1d = torch.ravel(transfer_y).detach().numpy()
        params_1d = list(map(torch.ravel, params))
        theta = torch.cat(params_1d, axis=0).detach().numpy()  # parameters concatenated
        jacs_2d = list(map(lambda x: x.reshape(N_load, -1), jacs))
        J = torch.cat(jacs_2d, dim=-1).detach().numpy()

    # Adaptation in parameter space
    Ip = np.eye(n_param)
    sigma = 0.1
    A = J.transpose() @ J + sigma * Ip
    P_lin = np.linalg.solve(A, J.transpose() @ y_out_1d)
