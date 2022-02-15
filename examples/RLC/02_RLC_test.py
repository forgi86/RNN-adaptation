import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from torchid import metrics
from loader import rlc_loader

if __name__ == '__main__':

    # matplotlib.rc('text', usetex=True)

    plot_input = False

    # dataset_type = 'train'
    dataset_type = 'test'
    # dataset_type = 'transfer'
    # dataset_type = 'eval'

    # model_name = 'ss_model_retrain'
    model_name = 'ss_model'

    # Column names in the dataset
    t, u, y, x = rlc_loader(dataset_type, "nl", noise_std=0.0)
    ts = t[1, 0] - t[0, 0]

    # Build validation data
    t_val_start = 0     #! one (?) time series for evaluation
    t_val_end = t[2000]
    idx_val_start = int(t_val_start // ts)
    idx_val_end = int(t_val_end // ts)
    u_val = u[idx_val_start:idx_val_end]
    x_true_val = x[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    time_val = t[idx_val_start:idx_val_end]

    # Setup neural model structure and load fitted model parameters
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Evaluate the model in open-loop simulation against validation data
    x_0 = np.zeros(2).astype(np.float32)
    with torch.no_grad():
        x_sim_torch = nn_solution(torch.tensor(x_0), torch.tensor(u_val))
        loss = torch.mean(torch.abs(x_sim_torch - torch.tensor(x_true_val))) #! mean absolute error !!!

    # Plot results
    x_sim = np.array(x_sim_torch)
    if not plot_input:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))
    else:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    time_val_us = time_val*1e6

    if dataset_type == 'id':
        t_plot_start = 0.0e-3  # 0.2e-3
    else:
        t_plot_start = 0.0e-3  # 1.9e-3
    t_plot_end = t_plot_start + 1.0  # 0.32e-3

    idx_plot_start = int(t_plot_start // ts)
    idx_plot_end = int(t_plot_end // ts)

    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end],
               x_true_val[idx_plot_start:idx_plot_end, 0],
               'k',  label='$v_C$')
    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end],
               x_sim[idx_plot_start:idx_plot_end, 0],
               'r--', label=r'$\hat{v}^{\mathrm{sim}}_C$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time mu_s")    # TODO: ("Time ($\mu$s)")
    ax[0].set_ylabel("Voltage (V)")
    # ax[0].set_ylim([-300, 300])

    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end],
               np.array(x_true_val[idx_plot_start:idx_plot_end:, 1]),
               'k', label='$i_L$')
    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end],
               x_sim[idx_plot_start:idx_plot_end:, 1],
               'r--', label=r'$\hat i_L^{\mathrm{sim}}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time mu_s")  # TODO: ($\mu$s)")
    ax[1].set_ylabel("Current (A)")
    # ax[1].set_ylim([-25, 25])

    if plot_input:
        ax[2].plot(time_val_us[idx_plot_start:idx_plot_end], u_val[idx_plot_start:idx_plot_end], 'k')
        # ax[2].legend(loc='upper right')
        ax[2].grid(True)
        ax[2].set_xlabel("Time mu_s")    # TODO: ("Time ($\mu$s)")
        ax[2].set_ylabel("Input voltage $v_C$ (V)")
        # ax[2].set_ylim([-400, 400])

    plt.show()
    fig_name = f"RLC_SS_{dataset_type}_{model_name}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # Saving state and input
    np.save(os.path.join("data", "RLC_SS_NL", "02_test_time_val.npy"), time_val_us)
    np.save(os.path.join("data", "RLC_SS_NL", "02_test_x_true.npy"), x_true_val)
    np.save(os.path.join("data", "RLC_SS_NL", "02_test_x_sim.npy"), x_sim)

    # R-squared metrics
    R_sq = metrics.r_squared(x_true_val, x_sim)
    print(f"R-squared metrics: {R_sq}")
