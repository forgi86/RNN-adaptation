import torch
import numpy as np
import os
from torchid.dynonet.module.lti import SisoLinearDynamicalOperator
from diffutil.jacobian import parameter_jacobian
import loader

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    model_name = 'IIR'  # model to be loaded
    n_b = 2  # numerator coefficients
    n_a = 2  # denominator coefficients
    sigma = 10.0

    # In[Load dataset]
    t, u, y, x = loader.rlc_loader("transfer", noise_std=sigma)
    n_data = t.size

    # In[Second-order dynamical system custom defined]
    G = SisoLinearDynamicalOperator(n_b, n_a)
    model_folder = os.path.join("models", model_name)
    G.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))

    # In[Adaptation in parameter space (naive way)]
    # NOTE: the jacobian in the formulas and comments has the classical definition (not transposed as in the paper)
    u_torch = torch.tensor(u[None, :, :])
    J = parameter_jacobian(G, u_torch, vectorize=False)  # custom-made full parameter jacobian
    n_param = J.shape[1]
    Ip = np.eye(n_param)
    F = J.transpose() @ J
    A = F + sigma**2 * Ip
    theta_lin = np.linalg.solve(A, J.transpose() @ y)  # adaptation!
    np.save(os.path.join("models", model_name, "theta_lin.npy"), theta_lin)

    # In[Evaluate linearized model on new data]
    t_new, u_new, y_new, x_new = loader.rlc_loader("eval", noise_std=0.0)
    u_torch_new = torch.tensor(u_new[None, :, :])
    J_new = parameter_jacobian(G, u_torch_new, vectorize=False)
    y_lin_new = J_new @ theta_lin

    # In[Plot]
    with torch.no_grad():
        y_torch_new = G(u_torch_new)
        y_sim_new = y_torch_new[0, :, [0]].numpy()

    import matplotlib.pyplot as plt
    plt.plot(t_new, y_new, 'k', label="True")
    plt.plot(t_new, y_sim_new, 'r', label="Sim")
    plt.plot(t_new, y_lin_new, 'b', label="Lin")
    plt.legend()

    np.save("y_lin_parspace_naive.npy", y_lin_new)
    np.save("y_sim.npy", y_sim_new)
