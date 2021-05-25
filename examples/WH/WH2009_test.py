import os
import matplotlib
import torch
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt
from models import WHNet3
from util.extract_util import extract_weights, load_weights, f_par_mod_in


if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    model_name = "model_WH3"  # base model (only used for jacobian feature extraction)

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "data_all.csv"))

    # Extract data
    y_meas = np.array(df_X[["y0"]], dtype=np.float32)
    u = np.array(df_X[["u"]], dtype=np.float32)
    fs = 1.0
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts

    test_start = 0
    N_test = 10000
    y_meas = y_meas[test_start:test_start+N_test, [0]]
    u = u[test_start:test_start+N_test, [0]]
    t = t[test_start:test_start+N_test]

    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)


    # In[Instantiate models]

    # Create models
    model = WHNet3()
    model_folder = os.path.join("models", model_name)
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))
    theta_lin = np.load(os.path.join("models", model_name, "theta_lin.npy"))

    # In[Parameter Jacobians]
    with torch.no_grad():
        y_sim_torch = model(u_torch)

    y_sim_torch = y_sim_torch.numpy()[0, ...]


    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    plt.plot(t, y_sim_torch, 'b', label="$\hat y$")
    plt.legend()

