import torch
import numpy as np
import os
from dynonet.module.lti import SisoLinearDynamicalOperator
import gpytorch
import finite_ntk
import loader


class DynoWrapper(torch.nn.Module):
    def __init__(self, dyno, n_in, n_out):
        super(DynoWrapper, self).__init__()
        self.dyno = dyno
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, u_in):
        u_in = u_in[None, :, :]  # [bsize, seq_len, n_in]
        y_out = self.dyno(u_in)  # [bsize, seq_len, n_out]
        n_out = y_out.shape[-1]
        y_out_ = y_out.reshape(-1, n_out)  # if n_out > 1 else y_out.reshape(-1, )
        # output size: [bsize*seq_len, n_out] or [bsize*seq_len, ]
        return y_out_


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
    model_name = 'IIR'  # model to be loaded
    n_b = 2  # numerator coefficients
    n_a = 2  # denominator coefficients
    sigma = 10.0
    use_linearstrategy = False

    # In[Load dataset]
    t, u, y, x = loader.rlc_loader("transfer", noise_std=sigma)
    n_data = t.size

    # In[Second-order dynamical system custom defined]
    G = SisoLinearDynamicalOperator(n_b, n_a)
    model_folder = os.path.join("models", model_name)
    G.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))

    # In[Model wrapping]
    n_in = 1
    n_out = 1
    G_wrapped = DynoWrapper(G, n_in, n_out)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * n_data, n_in)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * n_data, ))  # [bsize*seq_len, ]

    # In[Adaptation in function space]
    G_wrapped = DynoWrapper(G, n_in, n_out)
    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_lh.noise = sigma**2
    gp_model = ExactGPModel(u_torch_f, y_torch_f.squeeze(), gp_lh, G_wrapped, use_linearstrategy=use_linearstrategy)

    # No GP training (we consider the kernel (hyper)parameters fixed.
    # We may think of training the measurement noise by mll optimization...
    gp_model.eval()
    gp_lh.eval()

    # In[Evaluate the GP-like model on new data]
    t_new, u_new, y_new, x_new = loader.rlc_loader("eval", noise_std=0.0)
    u_torch_new = torch.tensor(u_new[None, :, :])
    u_torch_new_f = torch.clone(u_torch_new.view((1 * n_data, n_in)))  # [bsize*seq_len, n_in]

    with gpytorch.settings.fast_pred_var(): #, gpytorch.settings.max_cg_iterations(4000), gpytorch.settings.cg_tolerance(0.1):
        predictive_dist = gp_model(u_torch_new_f)
        y_lin_new = predictive_dist.mean.data

    # In[Plot]
    with torch.no_grad():
        y_torch_new = G(u_torch_new)
        y_sim_new = y_torch_new[0, :, [0]].numpy()

    import matplotlib.pyplot as plt
    plt.plot(t_new, y_new, 'k', label="True")
    plt.plot(t_new, y_sim_new, 'r', label="Sim")
    plt.plot(t_new, y_lin_new.detach().numpy(), 'b', label="Lin")
    plt.legend()

    if use_linearstrategy:
        np.save("y_lin_gp_parspace.npy", y_lin_new.detach().numpy())
    else:
        np.save("y_lin_gp_funspace.npy", y_lin_new.detach().numpy())
