import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchid import metrics
from open_lstm import OpenLSTM

if __name__ == "__main__":

    model_name = "lstm"
    # model_name = "lstm_retrain"

    # dataset_name = "train"
    # dataset_name = "test"
    # dataset_name = "transf"
    dataset_name = "eval"
    context = 25
    n_skip = 0  # initial n_skip samples for metrics (ignore transient)

    # Test
    u_test = torch.tensor(np.load(os.path.join("data", "cstr", f"u_{dataset_name}.npy")).astype(np.float32))
    y_test = torch.tensor(np.load(os.path.join("data", "cstr", f"y_{dataset_name}.npy")).astype(np.float32))

    n_inputs = u_test.shape[-1]

    u_test = torch.cat((u_test[:, 1:, :], y_test[:, :-1, :]), -1)
    y_test = y_test[:, 1:, :]

    model = OpenLSTM(context, n_inputs)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    y_sim = model(u_test)

    batch_idx = 0
    np.save(os.path.join("data", "cstr", "02_cstr_eval.npy"), y_sim.detach().numpy())

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.suptitle("Test")
    ax[0].plot(y_test.detach().numpy()[batch_idx, :, 0], label='Ground truth')
    ax[0].plot(y_sim.detach().numpy()[batch_idx, :, 0], label='Estimated fit')
    ax[0].axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax[0].set_ylabel('Y')
    ax[0].set_xlabel('X')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(y_test.detach().numpy()[batch_idx, :, 1], label='Ground truth')
    ax[1].plot(y_sim.detach().numpy()[batch_idx, :, 1], label='Estimated fit')
    ax[1].axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax[1].set_ylabel('Y')
    ax[1].set_xlabel('X')
    ax[1].legend()
    ax[1].grid()
    plt.show()


    # R-squared metrics
    R_sq = metrics.r_squared(y_test.detach().numpy()[:, n_skip:, :],
                             y_sim.detach().numpy()[:, n_skip:, :], time_axis=1)
    print(f"R-squared metrics: {R_sq}")

    rmse = metrics.error_rmse(y_test.detach().numpy()[:, n_skip:, :],
                             y_sim.detach().numpy()[:, n_skip:, :], time_axis=1)

    # print(f"RMSE: {rmse}")
    R_sq_mean = R_sq.mean(axis=0)
    rmse_mean = rmse.mean(axis=0)
