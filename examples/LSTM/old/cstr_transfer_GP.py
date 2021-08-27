import os
import numpy as np
import time
import torch
import torch.nn as nn
import gpytorch
import finite_ntk
from models import LSTMWrapper


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, model, use_linearstrategy=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(model=model, use_linearstrategy=use_linearstrategy)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    use_linearstrategy = True
    sigma = 0.03
    n_skip = 64  # skip initial n_skip samples for transfer (ignore transient)
    model_name = "lstm"

    # In[Load dataset]
    u = np.load(os.path.join("../data", "cstr", "u_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    y = np.load(os.path.join("../data", "cstr", "y_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size

    # In[Check dimensions]
    batch_size = 1
    seq_len, input_size = u.shape
    seq_len_, output_size = y.shape
    assert(seq_len == seq_len_)

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters

    model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("../models", model_filename)))

    # In[Model wrapping]
    model_wrapped = LSTMWrapper(model, seq_len, input_size)
    u_torch = torch.tensor(u, dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y, dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((input_size * seq_len, 1)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(output_size * seq_len, 1))  # [bsize*seq_len, ]
    y_f = y_torch_f.detach().numpy()

    # In[Adaptation in function/parameter space]
    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_lh.noise = sigma**2
    gp_model = ExactGPModel(u_torch_f, y_torch_f.squeeze(), gp_lh, model_wrapped, use_linearstrategy=use_linearstrategy)

    # No GP training (we consider the kernel (hyper)parameters fixed.
    # We may think of training the measurement noise by mll optimization...
    gp_model.eval()
    gp_lh.eval()

    # In[Evaluate the GP-like model on new data]
    u_new = np.load(os.path.join("../data", "cstr", "u_eval.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    y_new = np.load(os.path.join("../data", "cstr", "y_eval.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size

    u_torch_new = torch.tensor(u_new, dtype=torch.float)
    y_torch_new = torch.tensor(y_new, dtype=torch.float)
    u_torch_new_f = torch.clone(u_torch_new.view((input_size * seq_len, 1)))  # [bsize*seq_len, n_in]
    y_torch_new_f = torch.clone(y_torch_new.view(output_size * seq_len, 1))  # [bsize*seq_len, ]

    with gpytorch.settings.fast_pred_var():  #, gpytorch.settings.max_cg_iterations(4000), gpytorch.settings.cg_tolerance(0.1):
        #pass
        predictive_dist = gp_model(u_torch_new_f)
        #y_lin_new = predictive_dist.mean.data
