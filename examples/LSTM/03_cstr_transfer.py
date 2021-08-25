import os
import numpy as np
import time
import torch
import torch.nn as nn
from dynonet.utils.jacobian import parameter_jacobian

class LSTMWrapper(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size

    def forward(self, u_in_f):

        u_in = u_in_f.view(1, self.seq_len, self.input_size)
        y_out, _ = self.lstm(u_in)
        return y_out.view(-1, 1)


if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    vectorize = True  # vectorize jacobian evaluation (experimental!)
    sigma = 1.0
    model_name = "lstm"

    # In[Load dataset]
    u = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)
    y = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)

    # In[Check dimensions]
    batch_size, seq_len, input_size = u.shape
    batch_size_, seq_len_, output_size = y.shape
    assert(batch_size == 1)
    assert(batch_size == batch_size_)
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
    u_torch_f = torch.clone(u_torch.view((input_size * seq_len, 1)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(output_size * seq_len, 1))  # [bsize*seq_len, ]
    y_f = y_torch_f.detach().numpy()

    # In[Adaptation in parameter space (naive way)]
    J = parameter_jacobian(model_wrapped, u_torch_f, vectorize=vectorize)  # custom-made full parameter jacobian
    n_param = J.shape[1]
    Ip = np.eye(n_param)
    F = J.transpose() @ J
    A = F + sigma**2 * Ip
    theta_lin = np.linalg.solve(A, J.transpose() @ y_f)  # adaptation!
    np.save(os.path.join("models", "theta_lin.npy"), theta_lin)

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")
