import os
import numpy as np
import scipy.linalg
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
    n_x = x.shape[1]

    # In[Setup neural model structure and load fitted model parameters]
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(nn_solution)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * seq_len, output_size))  # [bsize*seq_len, ]

    # In[Adaptation in parameter space (fast jacobian computation via sensitivities)]
    n_param = sum(map(torch.numel, ss_model.parameters()))
    P_step = 1 / sigma ** 2 * torch.eye(n_param)
    theta_step = torch.zeros(n_param)
    x_step = torch.zeros(n_x, requires_grad=True)  # x_step initialized to initial state
    s_step = torch.zeros(n_x, n_param)  # initial sensitivity of state wrt theta set to 0
    time_start = time.time()

    x = []
    J = []
    for time_idx in range(seq_len):
         # print(time_idx)

        # Current input
        u_step = u_torch_f[time_idx, :]

        # Current state and current output sensitivity
        x.append(x_step)
        phi_step = s_step[[0], :].t()  # Special case of (14b), due to the simple output structure of this model
        J.append(phi_step)

        # Current P, theta, x to store
        # System update
        delta_x = 1.0 * ss_model(x_step, u_step)
        basis_x = torch.eye(n_x).unbind()

        # Jacobian of delta_x wrt x
        jacs_x = [torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0] for v in basis_x]
        J_x = torch.stack(jacs_x, dim=0)

        # Jacobian of delta_x wrt theta
        jacs_theta = [torch.autograd.grad(delta_x, ss_model.parameters(), v, retain_graph=True) for v in basis_x]
        jacs_theta_f = [torch.cat([jac.ravel() for jac in jacs_theta[j]]) for j in range(n_x)]  # ravel jacobian rows
        J_theta = torch.stack(jacs_theta_f)  # stack jacobian rows to obtain a jacobian matrix

        x_step = (x_step + delta_x).detach().requires_grad_(True)
        y_step = x_step[0]

        s_step = s_step + J_x @ s_step + J_theta  # Eq. 14a in the paper

    J = torch.stack(J).squeeze(-1).numpy()

    # Solution to linear Eq. 15 in the paper. Note: if you even use sp.linalg.solve and specify assume_a='pos',
    # it should use Cholesky factorization to solve the linear system, as mentioned in the paper.
    Ip = np.eye(n_param)
    F = J.transpose() @ J
    A = F + sigma**2 * Ip
    # np.linalg.solve(A, J.transpose() @ y)  # adaptation!
    theta_lin = scipy.linalg.solve(A, J.transpose() @ y, assume_a='pos')
    np.save(os.path.join("models", "theta_lin_sens.npy"), theta_lin)

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")
