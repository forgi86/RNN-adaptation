import os
import numpy as np
import time
import torch
import torch.nn as nn
from models import LSTMWrapper
from diffutil.products import jvp_diff, unflatten_like
import torch.optim as optim
from open_lstm import OpenLSTM
from torchid import metrics


if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    sigma = 0.03
    n_skip = 64  # skip initial n_skip samples for transfer (ignore transient)
    model_name = "lstm"
    n_iter = 250  # 750 # 100
    lr = 1e-2
    batch_size = 1
    context = 25

    # In[Load dataset]
    u = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[:batch_size, :, :]  # seq_len, input_size
    y = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[:batch_size, :, :]  # seq_len, output_size

    # In[Check dimensions]
    _, seq_len, input_size = u.shape
    _, seq_len_, output_size = y.shape
    N = y.size
    assert(seq_len == seq_len_)

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters
    model_op = OpenLSTM(context, input_size)
    model_filename = f"{model_name}.pt"
    model_op.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    model_wrapped = LSTMWrapper(model_op, seq_len, input_size, batch_s=batch_size)
    u_torch = torch.tensor(u[:, 1:, :].reshape(-1, input_size), dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y.reshape(-1, output_size), dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((input_size * (seq_len - 1), 1)))  # [bsize*seq_len*n_in, ]
    y_torch_f = torch.clone(y_torch[1:, :].view(output_size * (seq_len - 1), 1))  # [bsize*seq_len, ]
    y_f = y_torch_f.detach().numpy()

    u_torch_op = torch.cat((u_torch, y_torch[:-1, :]), dim=1).unsqueeze(0)

    n_param = sum(map(torch.numel, model_op.parameters()))
    #theta_lin = torch.tensor(np.load(os.path.join("models", "theta_lin.npy")).ravel())
    #theta_lin = 1/np.sqrt(n_param)*torch.randn(n_param)
    theta_lin = torch.zeros(n_param)
    theta_lin.requires_grad_(True)

    optimizer = optim.Adam([theta_lin], lr=lr)

    LOSS = []
    LOSS_REG = []
    LOSS_FIT = []
    y_sim_f = []
    y_lin_f = []
    for itr in range(n_iter):

        optimizer.zero_grad()

        theta_lin_f = unflatten_like(theta_lin, tensor_lst=list(model_wrapped.parameters()))

        # Compute nominal and linear model output
        y_sim_f = model_wrapped(u_torch_op)
        y_lin_f = jvp_diff(y_sim_f, model_wrapped.parameters(), theta_lin_f)[0]

        # Compute loss
        err_fit = y_torch_f[n_skip * output_size:] - y_lin_f[n_skip * output_size:]
        loss_fit = torch.sum(err_fit**2)
        loss_reg = sigma**2 * theta_lin.dot(theta_lin)
        loss = loss_fit + loss_reg
        loss = loss/1000

        # Statistics
        print(f'Iter {itr} | Tradeoff Loss {loss:.3f} | Fit Loss {loss_fit:.6f} | Reg Loss {loss_reg:.6f}')
        LOSS.append(loss.item())
        LOSS_FIT.append(loss_fit.item())
        LOSS_REG.append(loss_reg.item())

        # Optimization
        loss.backward()
        optimizer.step()

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")

    np.save(os.path.join("models", "theta_lin_gd.npy"), theta_lin.detach().numpy())

    # In[Plot]
    y_sim = y_sim_f.detach().numpy().reshape(seq_len-1, output_size)
    y_lin = y_lin_f.detach().numpy().reshape(seq_len-1, output_size)

    np.save(os.path.join("data", "cstr", "cstr_transf_gd_eval_sim.npy"), y_sim)
    np.save(os.path.join("data", "cstr", "cstr_transf_gd_eval_lin.npy"), y_lin)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(y[0, 1:, 0], 'k', label="Ground truth")
    ax[0].plot(y_sim[:, 0], 'b', label="LSTM")
    ax[0].plot(y_lin[:, 0], 'r--', label="JVP-LSTM")  # Jacobian-Vector Product
    ax[0].axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax[0].set_ylabel('Y')
    ax[0].set_xlabel('X')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(y[0, 1:, 1], 'k')
    ax[1].plot(y_sim[:, 1], 'b')
    ax[1].plot(y_lin[:, 1], 'r--')
    ax[1].axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax[1].set_ylabel('Y')
    ax[1].set_xlabel('X')
    # ax[1].legend()
    ax[1].grid(True)
    plt.show()