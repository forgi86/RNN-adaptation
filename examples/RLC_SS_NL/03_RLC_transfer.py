import os
import numpy as np
import time
import torch
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from dynonet.utils.jacobian import parameter_jacobian
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

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    vectorize = True  # vectorize jacobian evaluation (experimental!)
    sigma = 10.0
    #model_type = '1step_nonoise'
    model_type = "256step_noise_V"

    # In[Load dataset]
    t, u, y, x = loader.rlc_loader("transfer", dataset_type="nl", noise_std=10.0, n_data=2000)
    n_data = t.size

    # In[Second-order dynamical system custom defined]
    # Setup neural model structure and load fitted model parameters
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"model_SS_{model_type}.pt"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    n_in = 1
    n_out = 1
    model_wrapped = StateSpaceWrapper(nn_solution)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * n_data, n_in)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * n_data, n_out))  # [bsize*seq_len, ]

    # In[Adaptation in parameter space (naive way)]
    J = parameter_jacobian(model_wrapped, u_torch_f, vectorize=vectorize)  # custom-made full parameter jacobian
    n_param = J.shape[1]
    Ip = np.eye(n_param)
    F = J.transpose() @ J
    A = F + sigma**2 * Ip
    theta_lin = np.linalg.solve(A, J.transpose() @ y)  # adaptation!
    np.save(os.path.join("models", "theta_lin.npy"), theta_lin)

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")
