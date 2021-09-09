import os
import numpy as np
import time
import torch
import torch.nn as nn
from diffutil.jacobian import parameter_jacobian
from models import LSTMWrapper

if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    vectorize = True  # vectorize jacobian evaluation (experimental!)
    sigma = 0.03  # 1.0
    n_skip = 64  # skip initial n_skip samples for transfer (ignore transient)
    model_name = "lstm"

    # In[Load dataset]
    u = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    y = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size

    # In[Check dimensions]
    batch_size = 1
    seq_len, input_size = u.shape
    seq_len_, output_size = y.shape
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
    u_torch_f = torch.clone(u_torch.view((input_size * seq_len, 1)))  # [bsize*seq_len*n_in, ]
    y_torch_f = torch.clone(y_torch.view(output_size * seq_len, 1))  # [bsize*seq_len, ]
    y_f = y_torch_f.detach().numpy()

    # In[Adaptation in parameter space (naive way)]
    time_jac_start = time.time()
    J = parameter_jacobian(model_wrapped, u_torch_f, vectorize=vectorize)  # custom-made full parameter jacobian
    time_jac = time.time() - time_jac_start

    J_red = J[n_skip * output_size:, :]
    y_f_red = y_f[n_skip * output_size:]
    n_param = J.shape[1]

    Ip = np.eye(n_param)
    F = J_red.transpose() @ J_red
    A = F + sigma**2 * Ip
    theta_lin = np.linalg.solve(A, J_red.transpose() @ y_f_red)  # adaptation!
    np.save(os.path.join("models", "theta_lin_cf.npy"), theta_lin)  # cf: closed-form

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")
