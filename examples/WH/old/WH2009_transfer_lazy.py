import os
import matplotlib
import torch
import pandas as pd
import numpy as np
from models import WHNet3, DynoWrapper
from finite_ntk.lazy.ntk_lazytensor import NeuralTangent, Jacobian


if __name__ == '__main__':
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    model_name = "model_WH3"
    sigma = 10.0

    # Load dataset
    df_X = pd.read_csv(os.path.join("../data", "transfer", "data_all.csv"))
    signal_num = 1  # signal used for transfer (nominal model trained on signal 0)

    # Extract data
    y = np.array(df_X[[f"y{signal_num}"]], dtype=np.float32)
    u = np.array(df_X[["u"]], dtype=np.float32)
    N = y.size
    fs = 1.0
    ts = 1/fs
    t = np.arange(N)*ts

    transfer_start = 0
    n_data = 100000
    y = y[transfer_start:transfer_start+n_data, [0]]
    u = u[transfer_start:transfer_start+n_data, [0]]

    # In[Instantiate models]

    model = WHNet3()
    model_folder = os.path.join("../models", model_name)
    model.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))

    # In[Simulate model]
    transfer_u = torch.tensor(u[None, :, :])
    transfer_y = torch.tensor(y[None, :, :])
    sim_y = model(transfer_u)

    # In[Model wrapping]
    n_in = 1
    n_out = 1
    model_wrapped = DynoWrapper(model, n_in, n_out)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * n_data, n_in)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * n_data, n_out))  # [bsize*seq_len, ]

    # In[Adaptation in parameter space (the lazy/smart way)]
    # NOTE: the jacobian in the formulas and comments has the classical definition (not transposed as in the paper)
    K = NeuralTangent(model=model_wrapped, data=u_torch_f)
    JtJ = K.get_expansion(epsilon=1e-3)  # lazy J^T J using the Fisher matrix trick.
    # Note: 1e-4 is perhaps more accurate on this example, but I left it to 1e-3 to make it identical to the GP code...
    JtJ_hat = JtJ.add_jitter(sigma**2)  # lazy (J^T J + \sigma^2 I)
    Jt = Jacobian(model_wrapped, u_torch_f, y_torch_f, num_outputs=1)
    theta_lin = JtJ_hat.inv_matmul(Jt.matmul(y_torch_f))  # (J^T J + \sigma^2 I)^-1 J^T y

    torch.save(theta_lin, os.path.join("../models", model_name, "theta_lin_lazy.pt"))
