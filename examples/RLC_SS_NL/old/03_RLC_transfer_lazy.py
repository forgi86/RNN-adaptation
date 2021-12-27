import os
import numpy as np
import time
import torch
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from finite_ntk.lazy.ntk_lazytensor import NeuralTangent
import loader


class StateSpaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(StateSpaceWrapper, self).__init__()
        self.model = model

    def forward(self, u_in):
        x_0 = np.zeros(2).astype(np.float32)
        print("a")
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
    sigma = 0.1
    model_type = "256step_noise_V"

    # In[Load dataset]
    t, u, y, x = loader.rlc_loader("transfer", dataset_type="nl", noise_std=sigma, n_data=2000)
    seq_len = t.size

    # In[Second-order dynamical system custom defined]
    # Setup neural model structure and load fitted model parameters
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"model_SS_{model_type}.pt"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("../models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(nn_solution)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * seq_len, output_size))  # [bsize*seq_len, ]

    # In[Adaptation in parameter space (the lazy/smart way)]
    # NOTE: the jacobian in the formulas and comments has the classical definition (not transposed as in the paper)
    K = NeuralTangent(model=model_wrapped, data=u_torch_f)
    JtJ = K.get_expansion(epsilon=1e-4)  # lazy J^T J using the Fisher matrix trick.
    # Note: 1e-4 is perhaps more accurate on this example, but I left it to 1e-3 to make it identical to the GP code...
    JtJ_hat = JtJ.add_jitter(sigma**2)  # lazy (J^T J + \sigma^2 I)
    Jt = K.get_root()  # or finite_ntl.lazy.jacobian.Jacobian(G_wrapped, u_torch_f, y_torch_f, num_outputs=1)
    theta_lin = JtJ_hat.inv_matmul(Jt.matmul(y_torch_f))  # (J^T J + \sigma^2 I)^-1 J^T y

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")
