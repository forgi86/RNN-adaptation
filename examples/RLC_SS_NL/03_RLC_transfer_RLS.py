import os
import numpy as np
import time
import torch
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from diffutil.jacobian import parameter_jacobian
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

    # In[Second-order dynamical system custom defined]
    # Setup neural model structure and load fitted model parameters
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    model = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    model.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(model)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * seq_len, output_size))  # [bsize*seq_len, ]

    y_sim_f = model_wrapped(u_torch_f)

    n_param = sum(map(torch.numel, model.parameters()))
    P_old = 1/sigma**2*torch.eye(n_param)
    theta_old = torch.zeros(n_param)

    P = torch.zeros(seq_len, n_param, n_param)
    theta = torch.zeros(seq_len, n_param)
    for time_idx in range(seq_len):
        print(time_idx)

        y_sim_f = model_wrapped(u_torch_f[:time_idx, :])
        phis = torch.autograd.grad(y_sim_f[time_idx, 0], model.parameters(), retain_graph=True)
        phi = torch.cat([phi.ravel() for phi in phis]).view(-1, 1)  # column vector for simplicity here

        L = P_old @ phi/(1 + phi.t()@P_old@phi)
        theta[time_idx, :] = theta_old + L @ (y_torch_f[time_idx, 0] - phi.t() @ theta_old)
        P[time_idx, :, :] = P_old - (P_old @ phi @ phi.t() @ P_old)/(1 + phi.t() @ P_old @ phi)

        theta_old = theta[time_idx, :]
        P_old = P[time_idx, :, :]

    P = sigma ** 2 * P  #
    J = parameter_jacobian(model_wrapped, u_torch_f, vectorize=vectorize)  # custom-made full parameter jacobian

    import matplotlib.pyplot as plt
    plt.plot(y_torch_f, 'k')
    plt.plot(J @ theta_old.numpy(), 'r')
