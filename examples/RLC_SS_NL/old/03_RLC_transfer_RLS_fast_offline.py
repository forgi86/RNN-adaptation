import os
import numpy as np
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
    n_x = 2

    # In[Setup neural model structure and load fitted model parameters]
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    model = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    model.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(model)
    u_torch = torch.tensor(u, dtype=torch.float, requires_grad=False)
    u_torch_f = torch.clone(u_torch.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]

    x_sim = model(torch.zeros(2), u_torch)
    x_sim_ = x_sim.detach().clone().requires_grad_(True)

    u_torch_ = u_torch.detach().clone().requires_grad_(True)
    f_xu = ss_model(x_sim_, u_torch_)  # B, T, S

    A_LST = []
    B_LST = []
    for state_idx in range(n_x):
        var = torch.zeros_like(f_xu)
        var[:, state_idx] = 1.0
        A_LST.append(torch.autograd.grad(f_xu, x_sim_, var, create_graph=True)[0])
        B_LST.append(torch.autograd.grad(f_xu, u_torch_, var, create_graph=True)[0])

    J_x = torch.stack(A_LST, axis=1)
    B_all = torch.stack(B_LST, axis=1)

    J_theta = parameter_jacobian(ss_model, (x_sim_, u_torch_), vectorize=vectorize, flatten=False)
    J_theta = torch.cat([jac.reshape(seq_len, n_x, -1) for jac in J_theta], dim=-1)

    n_param = sum(map(torch.numel, model.parameters()))
    s_step = torch.zeros(n_x, n_param)
    s = []

    for time_idx in range(seq_len):
        print(time_idx)
        s.append(s_step)
        s_step = s_step + J_x[time_idx, :] @ s_step + J_theta[time_idx, :]

    s = torch.stack(s)

    #%%
    J = parameter_jacobian(model_wrapped, u_torch_f, vectorize=vectorize).detach().numpy()
    sy = s[:, 0, :].detach().numpy()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(sy - J)

    plt.figure()
    plt.plot(J)
