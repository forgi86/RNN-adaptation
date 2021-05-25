import os
import matplotlib
import torch
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt
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

    N_load = 10000
    y_meas = y_meas[:N_load, [0]]
    u = u[:N_load, [0]]
    t = t[:N_load]

    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)


    # In[Instantiate models]

    # Create models
    model = WHNet()
    model_folder = os.path.join("models", model_name)
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))
    theta_lin = np.load(os.path.join("models", model_name, "theta_lin.npy"))

    # In[Parameter Jacobians]
    with torch.no_grad():
        y_sim_torch = model(u_torch)

    y_sim_torch = y_sim_torch.numpy()[0, ...]

    # extract the parameters from the model in order to be able to take jacobians using the convenient functional API
    # see the discussion in https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240
    params, names = extract_weights(model)
    params_dict = dict(zip(names, params))
    n_param = sum(map(torch.numel, params))
    scalar_names = [f"{name}_{pos}" for name in params_dict for pos in range(params_dict[name].numel())]
    # [f"{names[i]}_{j}" for i in range(len(names)) for j in range(params[i].numel())]

    # from Pytorch module to function of the module parameters only
    f_par = functools.partial(f_par_mod_in, param_names=names, module=model, inputs=u_torch)
    f_par(*params)

    jacs = torch.autograd.functional.jacobian(f_par, params)
    jac_dict = dict(zip(names, jacs))

    with torch.no_grad():
        jacs_2d = list(map(lambda x: x.reshape(N_load, -1), jacs))
        J = torch.cat(jacs_2d, dim=-1).detach().numpy()
    y_transfer = J @ theta_lin

    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    #plt.plot(t, y_sim_torch, 'b', label="$\hat y$")
    plt.plot(t, y_transfer, 'r', label="$y_{tr}$")
    plt.legend()
