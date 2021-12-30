import os
import numpy as np
import time
import torch
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from diffutil.jacobian import parameter_jacobian
import loader
import copy
import matplotlib.pyplot as plt


class StateSpaceWrapper(torch.nn.Module):
    def __init__(self, model_ss):
        super(StateSpaceWrapper, self).__init__()
        self.model = model_ss
        self.x_0 = torch.zeros(2)

    def forward(self, u_in):
        #  x_0 = torch.zeros(2)
        x_sim_torch = self.model(self.x_0, u_in)
        y_out = x_sim_torch[:, [0]]    # Prediction only for non-hidden state
        self.x_0 = x_sim_torch[-1, :]  # Keep track of state
        return y_out


if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(1)
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
    model.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    input_size = 1
    output_size = 1
    model_wrapped = StateSpaceWrapper(model)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * seq_len, input_size)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * seq_len, output_size))  # [bsize*seq_len, ]

    y_sim_f = []  # model_wrapped(u_torch_f)

    n_param = sum(map(torch.numel, model.parameters()))  # Total parameters: 302
    P_old = 1/sigma**2*torch.eye(n_param)  # Covariance
    theta_old = torch.zeros(n_param)       # Mean

    P = torch.zeros(seq_len, n_param, n_param)  # (256 * 302 * 302)
    theta = torch.zeros(seq_len, n_param)

    ekf_pred = []

    for time_idx in range(2, seq_len):  # TODO: One input at a time
        # print(time_idx)

        # Predict
        y_sim_f = model_wrapped(u_torch_f[:time_idx, :].reshape(-1, 1))
        ekf_pred.append(y_sim_f[-1, 0].detach().numpy())
        phis = torch.autograd.grad(y_sim_f[-1, 0], model.parameters(), create_graph=True, retain_graph=False)
        phi = torch.cat([phi.ravel() for phi in phis]).view(-1, 1)  # column vector for simplicity here

        sd = copy.deepcopy(model.ss_model.state_dict())
        with torch.no_grad():
            # Update
            y_curr = torch.tensor(y_sim_f[-1, 0].detach().numpy())
            L = P_old @ phi/(1 + phi.t()@P_old@phi)  # Kalman gain matrix
            theta[time_idx, :] = theta_old + L @ ((y_torch_f[time_idx, 0] - y_curr).view(1))  # State estimate update
            P[time_idx, :, :] = P_old - (P_old @ phi @ phi.t() @ P_old)/(1 + phi.t() @ P_old @ phi)  # Error variance

            theta_old = theta[time_idx, :]
            P_old = P[time_idx, :, :]

            idx = 0
            new_param = theta[time_idx, :].detach()
            # Update parameters based on EKF update
            for key, val in sd.items():
                sd[key] = new_param[idx: idx + torch.numel(val)].reshape(val.shape)
                idx = idx + torch.numel(val)

        # Reinitialize model
        ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
        model = ForwardEulerSimulator(ss_model)
        model.ss_model.load_state_dict(sd)  # Load updated parameters from EKF
        model_wrapped = StateSpaceWrapper(model)

    adapt_time = time.time() - time_start
    print(f"\nAdapt time: {adapt_time:.2f}")

    # ekf_pred = model_wrapped(u_torch_f).detach().numpy()
    # ekf_pred = y_sim_f.detach().numpy()
    P = sigma ** 2 * P  #

    # Saving state and input
    np.save(os.path.join("data", "RLC_SS_NL", "03_transfer_EKF_y_true.npy"), y_torch_f.detach().numpy())
    np.save(os.path.join("data", "RLC_SS_NL", "03_transfer_EKF_y_pred.npy"), np.array(ekf_pred))

    print("Sizes: ", y_torch_f.shape)
    plt.plot(y_torch_f[:-1, :], 'k', label='Ground truth')
    # Plot prediction list instead of linearized Jacobian stuff
    plt.plot(ekf_pred, 'r', label='EKF prediction')
    plt.ylim([-0.03, 0.03])
    plt.legend()
    plt.show()
