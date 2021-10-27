import os
import numpy as np
import time
import torch
import torch.nn as nn
from open_lstm import OpenLSTM
from diffutil.jacobian import parameter_jacobian
from models import LSTMWrapper

if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    vectorize = True  # vectorize jacobian evaluation (experimental!)
    context = 25
    sigma = 0.03  # 1.0
    n_skip = 0  # skip initial n_skip samples for transfer (ignore transient)
    model_name = "lstm"

    # In[Load dataset]
    u = torch.from_numpy(np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[0, :, :])  # seq_len, input_size
    y = torch.from_numpy(np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[0, :, :])  # seq_len, output_size

    # In[Check dimensions]
    batch_size = 1
    seq_len, input_size = u.shape
    seq_len_, output_size = y.shape
    assert(seq_len == seq_len_)

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters
    n_inputs = u.shape[-1]

    model = OpenLSTM(context, n_inputs)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    # TODO: Remove all this reshaping stuff
    model_wrapped = LSTMWrapper(model, seq_len, input_size)
    """
    u_torch = torch.tensor(u, dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y, dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((input_size * seq_len, 1)))  # [bsize*seq_len*n_in, ]
    y_torch_f = torch.clone(y_torch.view(output_size * seq_len, 1))  # [bsize*seq_len, ]
    y_f = y_torch_f.detach().numpy()    
    """
    u = torch.unsqueeze(u, dim=0)
    y = torch.unsqueeze(y, dim=0)
    print("Input shape: ", y.shape, u.shape)

    u_torch = torch.cat((u[:, 1:, :], y[:, :-1, :]), -1)
    y_torch = y[:, 1:, :]

    y_torch_f = torch.clone(y_torch.view(output_size * (seq_len-1), 1))  # [bsize*seq_len, ]
    y_f = y_torch_f.detach().numpy()

    # In[Adaptation in parameter space (naive way)]
    time_jac_start = time.time()
    J = parameter_jacobian(model_wrapped, u_torch, vectorize=vectorize).detach().numpy()  # full parameter jacobian
    time_jac = time.time() - time_jac_start

    J_red = J[n_skip * output_size:, :]
    y_f_red = y_f[n_skip * output_size:]
    n_param = J.shape[1]

    Ip = np.eye(n_param)
    F = J_red.transpose() @ J_red
    A = F + sigma**2 * Ip
    theta_lin = np.linalg.solve(A, J_red.transpose() @ y_f_red)  # adaptation!
    print("theta lin: ", theta_lin.shape)
    np.save(os.path.join("models", "theta_lin_cf.npy"), theta_lin)  # cf: closed-form

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")
