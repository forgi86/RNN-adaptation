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
    output_idx = 1 # must run the code twice for output 0/1
    use_linearstrategy = True
    sigma = 0.03
    context = 25
    n_skip = 0  # skip initial n_skip samples for transfer (ignore transient)
    model_name = "lstm"
    batch_size = 1

    # In[Load dataset]
    u = np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[:batch_size, :, :]  # seq_len, input_size
    y = np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[:batch_size, :, :]  # seq_len, output_size
    y_ = y[..., [output_idx]]

    # In[Check dimensions]
    _, seq_len, input_size = u.shape
    _, seq_len_, output_size = y.shape  # 2 outputs
    assert(seq_len == seq_len_)

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters
    model_op = OpenLSTM(context, input_size, is_estimator=False)
    model_filename = f"{model_name}.pt"
    model_op.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Wrap model to send LSTM 3D input
    model_wrapped = LSTMWrapperSingleOutput(model_op, seq_len, input_size, output_idx, batch_size)
    u_torch = torch.tensor(u[:, 1:, :].reshape(-1, input_size), dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y_[:, 1:, :].reshape(-1, 1), dtype=torch.float) # single output
    y_torch_f = torch.tensor(y[:, :-1, :].reshape(-1, output_size), dtype=torch.float) # 2 output

    u_gp = torch.tensor(u[:, context+1:, :].reshape(-1, input_size), dtype=torch.float, requires_grad=False)
    y_gp = torch.tensor(y_[:, context+1:, :].reshape(-1, 1), dtype=torch.float)  # single output

    # In[Adaptation in function/parameter space]
    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_lh.noise = sigma**2

    # Before we initialize EGP and call eval() --> need to separately train LSMT estimator to initialize the state
    model_wrapped.estimate_state(u_torch, y_torch_f, nstep=25, output_size=output_size)
    gp_model = ExactGPModel(u_gp, y_gp.squeeze(), gp_lh, model_wrapped, use_linearstrategy=use_linearstrategy)

    # No GP training (we consider the kernel (hyper)parameters fixed.
    # We may think of training the measurement noise by mll optimization...
    gp_model.eval()
    gp_lh.eval()

    print("----------------------------------------------------------------------------------------------------------")
    batch_size = 1
    model_wrapped.set_batch_size(batch_s=batch_size)

    # In[Evaluate the GP-like model on new data]
    u_new = np.load(os.path.join("data", "cstr", "u_eval.npy")).astype(np.float32)[:batch_size, :, :]  # seq_len, input_size
    y_new = np.load(os.path.join("data", "cstr", "y_eval.npy")).astype(np.float32)[:batch_size, :, :]  # seq_len, output_size
    y_new_ = y_new[..., [output_idx]] # Single output

    u_torch_new = torch.tensor(u_new[:, 1:, :].reshape(-1, input_size), dtype=torch.float, requires_grad=False)
    y_torch_new = torch.tensor(y_new_[:, 1:, :].reshape(-1, 1), dtype=torch.float) # Single output
    y_torch_new_f = torch.tensor(y_new[:, :-1, :].reshape(-1, output_size), dtype=torch.float)

    u_gp_new = torch.tensor(u_new[:, context+1:, :].reshape(-1, input_size), dtype=torch.float, requires_grad=False)
    y_gp_new = torch.tensor(y_new_[:, context+1:, :].reshape(-1, 1), dtype=torch.float)  # single output

    with torch.no_grad():
        # Initialize the estimator with evaluation data
        model_wrapped.estimate_state(u_torch_new, y_torch_new_f, nstep=25, output_size=output_size)
        y_sim_new = model_wrapped(u_torch_new)
    y_sim_new = y_sim_new.detach().numpy()

    time_inference_start = time.time()
    with gpytorch.settings.fast_pred_var(), gpytorch.settings.cg_tolerance(0.1), gpytorch.settings.max_cg_iterations(4000):
        model_wrapped.estimate_state(u_torch_new, y_torch_new_f, nstep=25, output_size=output_size)
        gp_model.update_covar(model_wrapped)
        predictive_dist = gp_model(u_gp_new)
        y_lin_new = predictive_dist.mean.data
        lower_conf, upper_conf = predictive_dist.confidence_region()

    upper_conf = upper_conf.detach().numpy()
    lower_conf = lower_conf.detach().numpy()
    
    if not os.path.exists(os.path.join("data", "cstr")):
        os.makedirs(os.path.join("data", "cstr"))

    y_lin_new = y_lin_new[..., None].detach().numpy()

    np.save(os.path.join("data", "cstr", "GP_upper_conf_1.npy"), upper_conf)
    np.save(os.path.join("data", "cstr", "GP_lower_conf_1.npy"), lower_conf)
    np.save(os.path.join("data", "cstr", "GP_predict_1.npy"), y_lin_new)

    time_inference = time.time() - time_inference_start

    y_context = y_torch_new[1:context, :].detach().numpy()

    # In[Plot]
    ax = plt.subplot()
    ax.plot(y_new_[0, 1:, 0], 'k', label="True")
    ax.plot(y_sim_new[:, 0], 'r', label="Sim")
    ax.plot(np.concatenate((y_context[:, 0], y_lin_new[:, 0]), axis=0), 'b', label="Lin")
    ax.axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax.legend()
    ax.grid(True)
    plt.show()

    # y_lin_new = np.load(os.path.join("data", "cstr", "GP_predict.npy")).astype(np.float32)
    # upper_conf = np.load(os.path.join("data", "cstr", "GP_upper_conf.npy")).astype(np.float32)
    # lower_conf = np.load(os.path.join("data", "cstr", "GP_lower_conf.npy")).astype(np.float32)

    print("U_lin_new : ", y_lin_new.shape, lower_conf.shape, upper_conf.shape)
    # Plot confidence bounds for GP
    x = np.arange(context + 1, seq_len)
    fig, ax1 = plt.subplots()
    ax1.plot(y_new_[0, 1:, 0], 'k', label="True")
    ax1.plot(np.concatenate((y_context[:, 0], y_lin_new[:, 0]), axis=0), 'b', label="Lin")
    ax1.fill_between(x=x, y1=lower_conf, y2=upper_conf, label="Bounds", color='b', alpha=.1)
    ax1.plot(y_sim_new[:, 0], 'r', label="Sim")
    ax1.axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax1.legend()
    ax1.grid(True)
    plt.show()

    print(f"\nInference time: {time_inference:.2f}")

    # R-squared metrics
    R_sq_lin = metrics.r_squared(y_new_[0, (context+1):, :], y_lin_new)
    print(f"R-squared linear model: {R_sq_lin}")

    R_sq_sim = metrics.r_squared(y_new_[0, 1:, :], y_sim_new)
    print(f"R-squared nominal model: {R_sq_sim}")

