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

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    # In[Settings]
    model_name = "model_WH3"

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
    with torch.no_grad():
        model.G1.b_coeff *= -1
        model.F_nl.net[0].weight *= -1
        model.F_nl.net[0].bias *= -1

    G1 = control.TransferFunction(*model.G1.get_tfdata(), dt=ts)
    G1 = G1*-1
    G2 = control.TransferFunction(*model.G2.get_tfdata(), dt=ts)
    F_nl = model.F_nl


    x_lin = torch.linspace(-4, 4, 1000)
    with torch.no_grad():
        y_nl = F_nl(x_lin[..., None]).numpy()


    Y_nl = numpy.empty((100, 1000))
    for idx in range(100):
        with torch.no_grad():
            F_nl_ = copy.deepcopy(F_nl)
            F_nl_.eval()
            F_nl_.net[0].weight += 0.05*torch.randn(F_nl_.net[0].weight.shape)
            F_nl_.net[2].weight += 0.05 * torch.randn(F_nl_.net[2].weight.shape)
            Y_nl[idx, :] = F_nl_(x_lin[..., None]).squeeze().detach().numpy()

    x_lin = x_lin.numpy()

    plt.figure()
    control.bode(G1)

    plt.figure()
    control.bode(G2)

    plt.figure()
    plt.plot(x_lin, Y_nl.transpose(), 'r')
    plt.plot(x_lin, y_nl, 'k')
