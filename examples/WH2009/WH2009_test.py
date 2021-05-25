import torch
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import control
import util.metrics

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

    # In[Predict]

    u_torch = torch.tensor(u[None, :, :])
    y_hat = model(u_torch)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]

    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    plt.plot(t, y_hat, 'b', label="$\hat y$")
    plt.plot(t, y_meas - y_hat, 'r', label="$e$")
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(loc='upper right')
    plt.savefig('WH_fit.pdf')


    # In[Metrics]
    idx_test = range(t_test_start + t_skip, t_test_end)
    e_rms = 1000*util.metrics.error_rmse(y_meas[idx_test], y_hat[idx_test])[0]
    fit_idx = util.metrics.fit_index(y_meas[idx_test], y_hat[idx_test])[0]
    r_sq = util.metrics.r_squared(y_meas[idx_test], y_hat[idx_test])[0]

    print(f"RMSE: {e_rms:.1f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")


    # In[Plot for paper]

    t_test_start = 140000
    len_plot = 1000

    plt.figure(figsize=(4, 3))
    plt.plot(t[t_test_start:t_test_start+len_plot], y_meas[t_test_start:t_test_start+len_plot], 'k', label="$\mathbf{y}^{\mathrm{meas}}$")
    plt.plot(t[t_test_start:t_test_start+len_plot], y_hat[t_test_start:t_test_start+len_plot], 'b--', label="$\mathbf{y}$")
    plt.plot(t[t_test_start:t_test_start+len_plot], y_meas[t_test_start:t_test_start+len_plot] - y_hat[t_test_start:t_test_start+len_plot], 'r', label="$\mathbf{e}$")
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('WH_timetrace.pdf')