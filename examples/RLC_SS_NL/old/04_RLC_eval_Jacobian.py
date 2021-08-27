import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from finite_ntk.lazy.ntk_lazytensor import Jacobian
from dynonet.utils.jacobian import parameter_jacobian
from torchid import metrics
import loader


class StateSpaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(StateSpaceWrapper, self).__init__()
        self.model = model

    def forward(self, u_in):
        x_0 = np.zeros(2).astype(np.float32)
        x_sim_torch = self.model(torch.tensor(x_0), u_in)
        y_out = x_sim_torch[:, [0]]
        return y_out


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    model_type = "256step_noise_V"
    sigma = 10.0
    n_data = 2000

    # In[Load dataset]
    t_new, u_new, y_new, x_new = loader.rlc_loader("eval", dataset_type="nl", noise_std=0.0, n_data=n_data)

    # In[Second-order dynamical system custom defined]
    # Setup neural model structure and load fitted model parameters
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"model_SS_{model_type}.pt"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("../models", model_filename)))

    # In[Model wrapping]
    n_in = 1
    n_out = 1
    u_torch_new = torch.tensor(u_new[None, :, :])
    u_torch_new_f = torch.clone(u_torch_new.view((1 * n_data, n_in)))  # [bsize*seq_len, n_in]
    model_wrapped = StateSpaceWrapper(nn_solution)

    # In[Load theta_lin]
    theta_lin = np.load(os.path.join("../models", "theta_lin.npy"))

    # In[Parameter jacobian-vector product]
    Jt_new = Jacobian(model_wrapped, u_torch_new_f, None, num_outputs=1)
    y_lin_new = Jt_new.t().matmul(torch.tensor(theta_lin)).numpy()
    #np.save("y_lin_parspace_lazy.npy", y_lin_new)

    # In[Compute nominal model out on transfer dataset]
    with torch.no_grad():
        y_sim_new_torch = model_wrapped(u_torch_new_f)
    y_sim_new = y_sim_new_torch.detach().numpy()

    # In[Plot]
    plt.plot(y_new, 'k')
    plt.plot(y_sim_new, 'r')
    plt.plot(y_lin_new, 'b')

    # R-squared metrics
    R_sq = metrics.r_squared(y_new, y_lin_new)
    print(f"R-squared linear model: {R_sq}")

    R_sq = metrics.r_squared(y_new, y_sim_new)
    print(f"R-squared nominal model: {R_sq}")
