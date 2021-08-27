import os
import numpy as np
import time
import torch
import torch.nn as nn
from models import LSTMWrapper
from utils import jvp, jvp_diff, unflatten_like
import torch.optim as optim


if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    sigma = 0.03
    n_skip = 64  # skip initial n_skip samples for transfer (ignore transient)
    model_name = "lstm"
    n_iter = 10  # 100
    lr = 1e-1

    # In[Load dataset]
    u = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    y = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size

    # In[Check dimensions]
    batch_size = 1
    seq_len, input_size = u.shape
    seq_len_, output_size = y.shape
    N = y.size
    assert(seq_len == seq_len_)

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters

    model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    model_wrapped = LSTMWrapper(model, seq_len, input_size)
    u_torch = torch.tensor(u, dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y, dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view(input_size * seq_len, 1))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(output_size * seq_len, 1))  # [bsize*seq_len, ]
    y_f = y_torch_f.detach().numpy()

    n_param = sum(map(torch.numel, model.parameters()))
    #theta_lin = torch.tensor(np.load(os.path.join("models", "theta_lin.npy")).ravel())
    #theta_lin = 1/np.sqrt(n_param)*torch.randn(n_param)
    theta_lin = torch.zeros(n_param)
    theta_lin.requires_grad_(True)

    optimizer = optim.LBFGS([theta_lin], lr=lr)

    LOSS = []
    LOSS_REG = []
    LOSS_FIT = []

    def closure():
        optimizer.zero_grad()

        theta_lin_f = unflatten_like(theta_lin, tensor_lst=list(model_wrapped.parameters()))
        y_sim_f = model_wrapped(u_torch_f)
        y_lin_f = jvp_diff(y_sim_f, model_wrapped.parameters(), theta_lin_f)[0]

        # Compute loss
        err_fit = y_torch_f[n_skip * output_size:] - y_lin_f[n_skip * output_size:]
        loss_fit = torch.sum(err_fit**2)
        loss_reg = sigma**2 * theta_lin.dot(theta_lin)
        loss = loss_fit + loss_reg
        loss = loss/1000

        print(f'Iter {itr} | Tradeoff Loss {loss:.3f} | Fit Loss {loss_fit:.6f} | Reg Loss {loss_reg:.6f}')
        loss.backward()
        return loss

    for itr in range(n_iter):

        loss = optimizer.step(closure)

        # Statistics
        #LOSS.append(loss.item())
        #LOSS_FIT.append(loss_fit.item())
        #LOSS_REG.append(loss_reg.item())

        # Optimization
        #loss.backward()

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")

    np.save(os.path.join("models", "theta_lin_lbfgs.npy"), theta_lin.detach().numpy())

    # In[Plot]

    theta_lin_f = unflatten_like(theta_lin, tensor_lst=list(model_wrapped.parameters()))
    y_sim_f = model_wrapped(u_torch_f)
    y_lin_f = jvp_diff(y_sim_f, model_wrapped.parameters(), theta_lin_f)[0]

    y_sim = y_sim_f.detach().numpy().reshape(seq_len, output_size)
    y_lin = y_lin_f.detach().numpy().reshape(seq_len, output_size)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(y[:, 0], 'k')
    ax[0].plot(y_sim[:, 0], 'r')
    ax[0].plot(y_lin[:, 0], 'b')

    ax[1].plot(y[:, 1], 'k')
    ax[1].plot(y_sim[:, 1], 'r')
    ax[1].plot(y_lin[:, 1], 'b')
