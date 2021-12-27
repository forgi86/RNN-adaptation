import os
import numpy as np
import time
import torch
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from diffutil.jacobian import parameter_jacobian
import loader
import time


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

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    vectorize = True  # vectorize jacobian evaluation (experimental!)
    sigma = 0.1
    model_name = "ss_model"

    # In[Load dataset]
    t, u, y, x = loader.rlc_loader("transfer", dataset_type="nl", noise_std=sigma, n_data=2000)
    seq_len = t.size

    # In[Setup neural model structure and load fitted model parameters]
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    model = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    model.ss_model.load_state_dict(torch.load(os.path.join("../models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(model)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * seq_len, output_size))  # [bsize*seq_len, ]

    # In[Recursive Least Squares estimate of the linear model parameters]
    n_param = sum(map(torch.numel, model.parameters()))
    P_step = 1 / sigma ** 2 * torch.eye(n_param)
    theta_step = torch.zeros(n_param)
    x_step = torch.zeros(2)

    P = []#torch.zeros(seq_len, n_param, n_param)
    theta = []#torch.zeros(seq_len, n_param)
    x = []#torch.zeros(seq_len, 2)

    for time_idx in range(seq_len):
        print(time_idx)

        # Current input
        u_step = u_torch_f[time_idx, :]

        # Current P, theta, x to store
        P.append(P_step)
        theta.append(theta_step)
        x.append(x_step)

        # System update
        x_step = x_step + 1.0 * ss_model(x_step, u_step)
        y_step = x_step[0]

        # Estimate update
        # New regressor
        phis = torch.autograd.grad(y_step, model.parameters(), retain_graph=True)
        phi = torch.cat([phi.ravel() for phi in phis]).view(-1, 1)  # column vector for simplicity here

        # RLS
        L = P_step @ phi / (1 + phi.t() @ P_step @ phi)
        theta_step = theta_step + L @ (y_torch_f[time_idx, 0] - phi.t() @ theta_step)
        P_step = P_step - (P_step @ phi @ phi.t() @ P_step) / (1 + phi.t() @ P_step @ phi)

    P = torch.stack(P)
    theta = torch.stack(theta)
    x = torch.stack(x)

    P = sigma ** 2 * P  #
    J = parameter_jacobian(model_wrapped, u_torch_f, vectorize=vectorize)

    import matplotlib.pyplot as plt
    plt.plot(y_torch_f, 'k')
    plt.plot(J @ theta_step.numpy(), 'r')
