import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
import gpytorch
import finite_ntk
import loader
from torchid import metrics


class StateSpaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(StateSpaceWrapper, self).__init__()
        self.model = model

    def forward(self, u_in):
        x_0 = torch.zeros(2)  # np.zeros(2).astype(np.float32)
        x_sim_torch = self.model(x_0, u_in)
        y_out = x_sim_torch[:, [0]]
        return y_out


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

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    use_linearstrategy = False
    sigma = 0.1
    model_type = "256step_noise_V"

    # In[Load dataset]
    t, u, y, x = loader.rlc_loader("transfer", noise_std=sigma, n_data=2000)
    seq_len = t.size

    # In[Second-order dynamical system custom defined]
    # Setup neural model structure and load fitted model parameters
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"model_SS_{model_type}.pt"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(nn_solution)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * seq_len, output_size))  # [bsize*seq_len, ]

    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_lh.noise = sigma**2
    gp_model = ExactGPModel(u_torch_f, y_torch_f.squeeze(), gp_lh, model_wrapped, use_linearstrategy=use_linearstrategy)

    # No GP training (we consider the kernel (hyper)parameters fixed.
    # We may think of training the measurement noise by mll optimization...
    gp_model.eval()
    gp_lh.eval()

    # In[Evaluate the GP-like model on new data]
    t_new, u_new, y_new, x_new = loader.rlc_loader("eval", noise_std=0.0, n_data=2000)
    u_torch_new = torch.tensor(u_new[None, :, :])
    u_torch_new_f = torch.clone(u_torch_new.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]

    with gpytorch.settings.fast_pred_var(): #, gpytorch.settings.max_cg_iterations(4000), gpytorch.settings.cg_tolerance(0.1):
        predictive_dist = gp_model(u_torch_new_f)
        y_lin_new_f = predictive_dist.mean.data
        y_lin_new = y_lin_new_f.reshape(seq_len, output_size).detach().numpy()

    # In[Nominal model output]
    with torch.no_grad():
        y_sim_new_f = model_wrapped(u_torch_new_f)
        y_sim_new = y_sim_new_f.reshape(seq_len, output_size).detach().numpy()

    # In[Plot]
    plt.plot(t_new, y_new, 'k', label="True")
    plt.plot(t_new, y_sim_new, 'r', label="Sim")
    plt.plot(t_new, y_lin_new, 'b', label="Lin")
    plt.legend()

    # R-squared metrics
    R_sq = metrics.r_squared(y_new, y_lin_new)
    print(f"R-squared linear model: {R_sq}")

    R_sq = metrics.r_squared(y_new, y_sim_new)
    print(f"R-squared nominal model: {R_sq}")

    #if use_linearstrategy:
    #    np.save("y_lin_gp_parspace.npy", y_lin_new.detach().numpy())
    #else:
    #    np.save("y_lin_gp_funspace.npy", y_lin_new.detach().numpy())
