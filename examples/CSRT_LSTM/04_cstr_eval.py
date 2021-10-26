import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from diffutil.products import jvp, unflatten_like
from models import LSTMWrapper
from torchid import metrics


learn_state = False

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    n_skip = 64  # skip initial n_skip samples for metrics (ignore transient)
    model_name = "lstm"
    # dataset_name = "transf"
    dataset_name = "eval"

    # In[Load dataset]
    # u_new = np.load(os.path.join("data", "cstr", f"u_{dataset_name}.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    # y_new = np.load(os.path.join("data", "cstr", f"y_{dataset_name}.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size

    # In[Load dataset]
    u_new = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    y_new = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size

    u_new = u_new[:, :]
    y_new = y_new[:, :]


    # In[Check dimensions]
    batch_size = 1
    seq_len, input_size = u_new.shape
    seq_len_, output_size = y_new.shape
    assert(seq_len == seq_len_)

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters
    model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    if learn_state is False:
        with torch.no_grad():
            _, hid = model(torch.tensor(np.zeros((1, 1, input_size)), dtype=torch.float32))
        hid_ = [0*hid[0], 0*hid[1]]

        model_wrapped = LSTMWrapper(model, seq_len, input_size, hid_)

    else:
        model_wrapped = LSTMWrapper(model, seq_len, input_size)

    print(model_wrapped.h, model_wrapped.c)

    u_torch_new = torch.tensor(u_new, dtype=torch.float, requires_grad=False)
    y_torch_new = torch.tensor(y_new, dtype=torch.float)
    u_torch_new_f = torch.clone(u_torch_new.view((input_size * seq_len, 1)))  # [bsize*seq_len, n_in]
    y_torch_new_f = torch.clone(y_torch_new.view(output_size * seq_len, 1))  # [bsize*seq_len, ]

    # In[Load theta_lin]
    # theta_lin = np.zeros_like(theta_lin)
    theta_lin = np.load(os.path.join("models", "theta_lin_cf.npy"))  # closed-form
    # theta_lin = np.load(os.path.join("models", "theta_lin_gd.npy"))  # gradient descent
    # theta_lin = np.load(os.path.join("models", "theta_lin_lbfgs.npy"))  # L-BFGS
    theta_lin = torch.tensor(theta_lin)
    # In[Nominal model output]
    y_sim_new_f = model_wrapped(u_torch_new_f)
    y_sim_new = y_sim_new_f.reshape(seq_len, output_size).detach().numpy()

    # In[Linearized model output]
    theta_lin_f = unflatten_like(theta_lin, tensor_lst=list(model_wrapped.parameters()))
    time_jvp_start = time.time()
    y_lin_new_f = jvp(y_sim_new_f, model_wrapped.parameters(), theta_lin_f)[0]
    time_jvp = time.time() - time_jvp_start
    y_lin_new = y_lin_new_f.reshape(seq_len, output_size).detach().numpy()

    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(y_new[:, 0], 'y', label="True")
    ax[0].plot(y_sim_new[:, 0], 'r--', label="Nominal NN")
    ax[0].plot(y_lin_new[:, 0], 'b--', label="Adapted")
    ax[0].legend()

    ax[1].plot(y_new[:, 1], 'y')
    ax[1].plot(y_sim_new[:, 1], 'r')
    ax[1].plot(y_lin_new[:, 1], 'b')

    # R-squared metrics
    R_sq_lin = metrics.r_squared(y_new[n_skip:, :], y_lin_new[n_skip:, :])
    print(f"R-squared linear model: {R_sq_lin}")

    R_sq_sim = metrics.r_squared(y_new[n_skip:, :], y_sim_new[n_skip:, :])
    print(f"R-squared nominal model: {R_sq_sim}")


    plt.show()