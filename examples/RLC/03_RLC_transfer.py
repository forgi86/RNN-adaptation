import os
import numpy as np
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
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
    dataset = "transfer"  #  "eval"
    ds_filename = 'transfereval/R:4.0_L:5e-05_C:3.5e-07.npy'
    # ds_filename = 'val/R:1.0935361214295956_L:9.589787600234677e-05_C:2.9333830098482676e-07.npy'
    # In[Load dataset]
    # t, u, y, x = loader.rlc_loader(dataset, dataset_type="nl", noise_std=sigma, n_data=2000)
    t, u, y, x = loader.rlc_loader_multitask(ds_filename,
                                             trajectory=0,
                                             steps=50,
                                             noise_std=sigma,
                                             scale=False)

    seq_len = t.size

    # In[Setup neural model structure and load fitted model parameters]
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    nn_solution.ss_model.load_state_dict(
        torch.load(os.path.join("models", model_filename)))
    # In[Model wrapping]
    input_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(nn_solution)
    u_torch = torch.tensor(u[None, ...],
                           dtype=torch.float,
                           requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view(
        (1 * seq_len, input_size)))  # [bsize*seq_len, n_in] # [2000,1]
    y_torch_f = torch.clone(y_torch.view(1 * seq_len,
                                         output_size))  # [bsize*seq_len, ]
    print("Computing Jacobian: ")
    # In[Adaptation in parameter space (naive way)]
    J = parameter_jacobian(
        model_wrapped, u_torch_f,
        vectorize=vectorize).detach().numpy()  # full parameter jacobian
    print(J.shape)
    n_param = J.shape[1]
    Ip = np.eye(n_param)
    F = J.transpose() @ J
    A = F + sigma**2 * Ip
    print("Solving for linear parameter adaption: ")
    theta_lin = np.linalg.solve(A, J.transpose() @ y)  # adaptation!
    print(theta_lin.shape)
    np.save(os.path.join("models", "theta_lin.npy"), theta_lin)

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")
