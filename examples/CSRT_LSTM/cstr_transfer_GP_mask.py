import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import gpytorch
import finite_ntk
from open_lstm import OpenLSTM
from models import LSTMWrapperSingleOutput
from torchid import metrics


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, model_egp, use_linearstrategy=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(model=model_egp, use_linearstrategy=use_linearstrategy)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def update_covar(self, model_c):
        self.covar_module = finite_ntk.lazy.NTK(model=model_c, use_linearstrategy=use_linearstrategy)


if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    output_idx = 1  # must run the code twice for output 0/1
    use_linearstrategy = False
    sigma = 0.03
    context = 25
    n_skip = 0  # skip initial n_skip samples for transfer (ignore transient)
    model_name = "lstm"

    # In[Load dataset]
    u = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    y = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size
    y_ = y[..., [output_idx]]

    # In[Check dimensions]
    batch_size = 1
    seq_len, input_size = u.shape
    seq_len_, output_size = y.shape  # 2 outputs
    assert(seq_len == seq_len_)

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters
    model_op = OpenLSTM(context, input_size, is_estimator=False)
    model_filename = f"{model_name}.pt"
    model_op.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Wrap model to send LSTM 3D input
    model_wrapped = LSTMWrapperSingleOutput(model_op, seq_len, input_size, output_idx)
    u_torch = torch.tensor(u[1:, :], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y_[1:, :], dtype=torch.float) # single output
    y_torch_f = torch.tensor(y[:-1, :], dtype=torch.float) # 2 output

    # In[Adaptation in funloadction/parameter space]
    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_lh.noise = sigma**2

    # Before we initialize EGP and call eval() --> need to separately train LSMT estimator to initialize the state
    model_wrapped.estimate_state(u_torch, y_torch_f, nstep=25, output_size=output_size)
    gp_model = ExactGPModel(u_torch, y_torch.squeeze(), gp_lh, model_wrapped, use_linearstrategy=use_linearstrategy)

    # No GP training (we consider the kernel (hyper)parameters fixed.
    # We may think of training the measurement noise by mll optimization...
    gp_model.eval()
    gp_lh.eval()

    # In[Evaluate the GP-like model on new data]
    u_new = np.load(os.path.join("data", "cstr", "u_eval.npy")).astype(np.float32)[0, :, :]  # seq_len, input_size
    y_new = np.load(os.path.join("data", "cstr", "y_eval.npy")).astype(np.float32)[0, :, :]  # seq_len, output_size
    y_new_ = y_new[..., [output_idx]]

    u_torch_new = torch.tensor(u_new[1:, :], dtype=torch.float)
    y_torch_new = torch.tensor(y_new_, dtype=torch.float)
    y_torch_new_f = torch.tensor(y_new[:-1, :], dtype=torch.float)

    with torch.no_grad():
        # Initialize the estimator with evaluation data
        model_wrapped.estimate_state(u_torch_new, y_torch_new_f, nstep=25, output_size=output_size)
        gp_model.update_covar(model_wrapped)
        y_sim_new = model_wrapped(u_torch_new)
    y_sim_new = y_sim_new.detach().numpy()

    time_inference_start = time.time()
    with gpytorch.settings.fast_pred_var():  #, gpytorch.settings.max_cg_iterations(4000), gpytorch.settings.cg_tolerance(0.1):
        predictive_dist = gp_model(u_torch_new)
        y_lin_new = predictive_dist.mean.data
    y_lin_new = y_lin_new[..., None].detach().numpy()
    time_inference = time.time() - time_inference_start

    # In[Plot]
    fig = plt.figure()
    plt.plot(y_new_[:, 0], 'k', label="True")
    plt.plot(y_sim_new[:, 0], 'r', label="Sim")
    plt.plot(y_lin_new[:, 0], 'b', label="Lin")
    plt.legend()
    plt.grid()
    plt.show()

    # R-squared metrics
    R_sq_lin = metrics.r_squared(y_new_[n_skip:, :], y_lin_new[n_skip:, :])
    print(f"R-squared linear model: {R_sq_lin}")

    R_sq_sim = metrics.r_squared(y_new_[n_skip:, :], y_sim_new[n_skip:, :])
    print(f"R-squared nominal model: {R_sq_sim}")

    print(f"\nInference time: {time_inference:.2f}")