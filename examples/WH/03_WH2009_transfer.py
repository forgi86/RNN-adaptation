import os
import matplotlib
import torch
import pandas as pd
import numpy as np
from models import WHNet3
from diffutil.jacobian import parameter_jacobian

if __name__ == '__main__':

    if __name__ == '__main__':
        matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

        model_name = "model_WH3"
        sigma = 1e-6

        # Load dataset
        df_X = pd.read_csv(os.path.join("data", "transfer", "data_all.csv"))
        signal_num = 1  # signal used for transfer (nominal model trained on signal 0)

        # Extract data
        y = np.array(df_X[[f"y{signal_num}"]], dtype=np.float32)
        u = np.array(df_X[["u"]], dtype=np.float32)
        N = y.size
        fs = 1.0
        ts = 1 / fs
        t = np.arange(N) * ts

        transfer_start = 0
        n_data = 10000
        y = y[transfer_start:transfer_start + n_data, [0]]
        u = u[transfer_start:transfer_start + n_data, [0]]

    # In[Instantiate models]

    # Create models
    model = WHNet3()
    model_folder = os.path.join("models", model_name)
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))

    # In[Simulate model]
    transfer_u = torch.tensor(u[None, :, :])
    transfer_y = torch.tensor(y[None, :, :])
    sim_y = model(transfer_u)

    # In[Parameter Jacobians]
    J = parameter_jacobian(model, transfer_u, vectorize=False)  # custom-made full parameter jacobian

    # Adaptation in parameter space
    n_param = J.shape[1]
    Ip = np.eye(n_param)
    F = J.transpose() @ J
    A = F + sigma**2 * Ip
    theta_lin = np.linalg.solve(A, J.transpose() @ y)  # adaptation!
    np.save(os.path.join("models", model_name, "theta_lin.npy"), theta_lin)
