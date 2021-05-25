import numpy
import torch
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import util.metrics
import control
import copy
from models import WHNet3


# In[Main]
if __name__ == '__main__':

    np.random.seed(44)
    #torch.seed(321)

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    # In[Settings]
    model_name = "model_WH3"


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

    t_fit_start = 0
    t_fit_end = 100000
    t_test_start = 100000
    t_test_end = 188000
    t_skip = 1000  # skip for statistics

    # In[Instantiate models]


    # Create models
    model = WHNet3()

    model_folder = os.path.join("models", model_name)
    # Create model parameters
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))
#    with torch.no_grad():
#        model.G1.b_coeff *= -1
#        model.F_nl.net[0].weight *= -1
#        model.F_nl.net[0].bias *= -1

    u_torch = torch.tensor(u[None, :, :], dtype=torch.float, requires_grad=False)
    y_train = model(u_torch)

    n_sim = 11
    Y_nl = numpy.empty((N, n_sim))
    for idx in range(n_sim):
        with torch.no_grad():
            model_ = copy.deepcopy(model)
            if idx > 0:
                model_.eval()
                model_.F_nl.net[0].weight += 0.05 * torch.randn(model.F_nl.net[0].weight.shape)
                model_.F_nl.net[2].weight += 0.05 * torch.randn(model.F_nl.net[2].weight.shape)

            Y_nl[:, idx] = model_(u_torch).squeeze()

    plt.plot(Y_nl[:, :10])

    data_mat = np.c_[t.reshape(-1,1), u, Y_nl]
    cols = ["t", "u"] + [f"y{i}" for i in range(n_sim)]
    df_data = pd.DataFrame(data_mat, columns=cols)

    df_data.to_csv(os.path.join("data", "transfer", "data_all.csv"), index=False)
