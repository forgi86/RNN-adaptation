import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchid import metrics

if __name__ == "__main__":

    model_name = "lstm"
    n_skip = 64

    # Test
    u_test = torch.tensor(np.load(os.path.join("data", "cstr", "u_test.npy")).astype(np.float32))
    y_test = torch.tensor(np.load(os.path.join("data", "cstr", "y_test.npy")).astype(np.float32))

    model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    y_sim, _ = model(u_test)

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.suptitle("Test")
    batch_idx = 13
    ax[0].plot(y_test.detach().numpy()[batch_idx, :, 0], label='True')
    ax[0].plot(y_sim.detach().numpy()[batch_idx, :, 0], label='Fit')
    ax[0].legend()

    ax[1].plot(y_test.detach().numpy()[batch_idx, :, 1], label='True')
    ax[1].plot(y_sim.detach().numpy()[batch_idx, :, 1], label='Fit')
    ax[1].legend()

    # R-squared metrics
    R_sq = metrics.r_squared(y_test.detach().numpy()[:, n_skip:, :],
                             y_sim.detach().numpy()[:, n_skip:, :], time_axis=1)
    print(f"R-squared metrics: {R_sq}")

    R_sq_mean = R_sq.mean(axis=0)
