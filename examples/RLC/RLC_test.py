import torch
import pandas as pd
import numpy as np
import os
from dynonet.module.lti import SisoLinearDynamicalOperator
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    model_name = 'IIR'
    n_b = 2
    n_a = 2

    # In[Column names in the dataset]
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # In[Load dataset]
    df_X = pd.read_csv(os.path.join("data", "RLC_data_test_nl.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # In[Output]
    y = np.copy(x[:, [0]])

    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)

    # In[Second-order dynamical system custom defined]
    G = SisoLinearDynamicalOperator(n_b, n_a)
    model_folder = os.path.join("models", model_name)
    G.load_state_dict(torch.load(os.path.join(model_folder, "G.pt")))

    # In[Parameter Jacobians]
    with torch.no_grad():
        y_sim_torch = G(u_torch)

    y_sim_torch = y_sim_torch.numpy()[0, ...]

    # In[Plot]
    plt.figure()
    plt.plot(t, y, 'k', label="$y$")
    plt.plot(t, y_sim_torch, 'b', label="$\hat y$")
    plt.legend()



