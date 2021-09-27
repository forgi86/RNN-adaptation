import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from diffutil.products import jvp, unflatten_like
from torchid import metrics
import loader


class StateSpaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(StateSpaceWrapper, self).__init__()
        self.model = model

    def forward(self, u_in):
        x_0 = torch.zeros(2)
        x_sim_torch = self.model(x_0, u_in)
        y_out = x_sim_torch[:, [0]]
        return y_out


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    model_name = "ss_model"
    seq_len = 2000
    # dataset = "transfer"
    dataset = "eval"

    # In[Load dataset]
    t_new, u_new, y_new, x_new = loader.rlc_loader(dataset, dataset_type="nl", noise_std=0.0, n_data=seq_len)

    # In[Second-order dynamical system custom defined]
    # Setup neural model structure and load fitted model parameters
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    u_torch_new = torch.tensor(u_new[None, :, :])
    u_torch_new_f = torch.clone(u_torch_new.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]
    model_wrapped = StateSpaceWrapper(nn_solution)

    # In[Load theta_lin]
    theta_lin = torch.tensor(np.load(os.path.join("models", "theta_lin.npy")))

    # In[Nominal model output]
    y_sim_new_f = model_wrapped(u_torch_new_f)
    y_sim_new = y_sim_new_f.reshape(seq_len, output_size).detach().numpy()

    # In[Parameter jacobian-vector product]
    theta_lin = torch.tensor(theta_lin)
    theta_lin_f = unflatten_like(theta_lin, tensor_lst=list(model_wrapped.parameters()))
    y_lin_new_f = jvp(y_sim_new_f, model_wrapped.parameters(), theta_lin_f)[0]
    y_lin_new = y_lin_new_f.reshape(seq_len, output_size).detach().numpy()

    # In[Plot]
    plt.plot(y_new, 'k', label="data")
    plt.plot(y_sim_new, 'r', label="nominal")
    plt.plot(y_lin_new, '--b', label="adapted")

    # R-squared metrics
    R_sq = metrics.r_squared(y_new, y_lin_new)
    print(f"R-squared linear model: {R_sq}")

    R_sq = metrics.r_squared(y_new, y_sim_new)
    print(f"R-squared nominal model: {R_sq}")

    plt.legend()

    plt.show()