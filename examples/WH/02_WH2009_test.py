import os
import matplotlib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import WHNet3
from torchid import metrics

if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    model_name = "model_WH3"

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "transfer", "data_all.csv"))
    signal_num = 0  # signal used for test (nominal model trained on signal 0)

    # Extract data
    y_meas = np.array(df_X[[f"y{signal_num}"]], dtype=np.float32)
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

    # In[Simulate]
    with torch.no_grad():
        y_sim_torch = model(u_torch)

    y_sim = y_sim_torch.numpy()[0, ...]

    # In[Metrics]
    R_sq = metrics.r_squared(y_meas, y_sim)[0]
    rmse = metrics.error_rmse(y_meas, y_sim)[0]

    print(f"R-squared metrics: {R_sq}")
    print(f"RMSE metrics: {rmse}")

    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    plt.plot(t, y_sim, 'b', label="$\hat y$")
    plt.legend()

