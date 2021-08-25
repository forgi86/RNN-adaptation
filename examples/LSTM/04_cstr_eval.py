import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from finite_ntk.lazy.ntk_lazytensor import Jacobian
from torchid import metrics


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

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    vectorize = True  # vectorize jacobian evaluation (experimental!)
    sigma = 10.0
    model_name = "lstm"

    # In[Load dataset]
    u_new = np.load(os.path.join("data", "cstr", "u_eval.npy")).astype(np.float32)
    y_new = np.load(os.path.join("data", "cstr", "y_eval.npy")).astype(np.float32)

    # In[Check dimensions]
    batch_size, seq_len, input_size = u_new.shape
    batch_size_, seq_len_, output_size = y_new.shape
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
    u_torch_new = torch.tensor(u_new, dtype=torch.float, requires_grad=False)
    y_torch_new = torch.tensor(y_new, dtype=torch.float)
    u_torch_new_f = torch.clone(u_torch_new.view((input_size * seq_len, 1)))  # [bsize*seq_len, n_in]
    y_torch_new_f = torch.clone(y_torch_new.view(output_size * seq_len, 1))  # [bsize*seq_len, ]
    y_f = y_torch_new_f.detach().numpy()

    # In[Load theta_lin]
    theta_lin = np.load(os.path.join("models", "theta_lin.npy"))

    # In[Parameter jacobian-vector product]
    Jt_new = Jacobian(model_wrapped, u_torch_new_f, None, num_outputs=1)
    y_lin_new_f = Jt_new.t().matmul(torch.tensor(theta_lin)).numpy()
    y_lin_new = y_lin_new_f.reshape(seq_len, output_size)

    # In[Compute nominal model out on transfer dataset]
    with torch.no_grad():
        y_sim_new_torch_f = model_wrapped(u_torch_new_f)
    y_sim_new_f = y_sim_new_torch_f.detach().numpy()
    y_sim_new = y_sim_new_f.reshape(seq_len, output_size)

    # In[Plot]
    plt.plot(y_new[0, :, :], 'k')
    plt.plot(y_sim_new, 'r')
    plt.plot(y_lin_new, 'b')

    # R-squared metrics
    R_sq = metrics.r_squared(y_new, y_lin_new_f)
    print(f"R-squared linear model: {R_sq}")

    R_sq = metrics.r_squared(y_new, y_sim_new)
    print(f"R-squared nominal model: {R_sq}")
