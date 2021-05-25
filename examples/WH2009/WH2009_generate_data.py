import torch
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import util.metrics
import control

from models import WHNet

# In[Main]
if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    # In[Settings]
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

    t_fit_start = 0
    t_fit_end = 100000
    t_test_start = 100000
    t_test_end = 188000
    t_skip = 1000  # skip for statistics

    # In[Instantiate models]

    # Create models
    model = WHNet()

    model_folder = os.path.join("models", model_name)
    # Create model parameters
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))

    G1 = control.TransferFunction(*model.G1.get_tfdata(), dt=ts)
    G2 = control.TransferFunction(*model.G2.get_tfdata(), dt=ts)

    plt.figure()
    control.bode(G1)

    plt.figure()
    control.bode(G2)