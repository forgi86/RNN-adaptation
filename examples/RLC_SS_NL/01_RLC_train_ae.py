import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from examples.RLC_SS_NL.lstm_rev import LSTMFlippedStateEstimator
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from loader import rlc_loader
import scipy.linalg


# Truncated simulation error minimization method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 20000  # gradient-based optimization steps
    seq_len = 64  # subsequence length m
    batch_size = 32  # batch size q
    t_fit = 2e-3  # fitting on t_fit ms of data
    lr = 1e-4  # learning rate
    test_freq = 100  # print message every test_freq iterations
    add_noise = True
    output = "V_C"

    # Column names in the dataset
    t, u, y, x = rlc_loader("train", "nl", output=output, noise_std=0.0)

    U = scipy.linalg.toeplitz(u, np.r_[u[0, 0], np.zeros(seq_len - 1)]).astype(np.float32)
    Y = scipy.linalg.toeplitz(y, np.r_[y[0, 0], np.zeros(seq_len - 1)]).astype(np.float32)

    # Get fit data #
    ts = t[1] - t[0]
    n_fit = int(t_fit // ts)  # x.shape[0]
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    # Setup neural model structure
    state_estimator = LSTMFlippedStateEstimator(n_u=1, n_y=1, n_x=2)
    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': state_estimator.parameters(),    'lr': lr},
        {'params': nn_solution.parameters(), 'lr': lr},
    ], lr=lr)

    # Batch extraction funtion
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = y_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64),
                                       batch_size, replace=False)  # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len)  # batch samples indices
        batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_y = torch.tensor(y_fit[batch_idx])

        return batch_t, batch_u, batch_y


    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []
    start_time = time.time()
    # Training loop

    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        batch_t, batch_u, batch_y = get_batch(batch_size, seq_len)

        # Compute fit loss
        batch_x0 = state_estimator(batch_u, batch_y)[0, :, :]
        batch_x_sim = nn_solution(batch_x0, batch_u)
        batch_y_sim = batch_x_sim[..., [0]]

        # Compute consistency loss
        err_ae = batch_y - batch_y_sim
        loss_ae = torch.mean(err_ae**2)

        # Compute trade-off loss
        loss = loss_ae

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | AE Loss {loss:.4f} '
                      )

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%%

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = "ss_model_ae.pt"
    torch.save(ss_model.state_dict(), os.path.join("models", model_filename))

    t_val = 5e-3
    n_val = int(t_val // ts)  # x.shape[0]

    #%%
    with torch.no_grad():
        u_v = torch.tensor(
            u[:, None, :]
        )
        y_v = torch.tensor(
            y[:, None, :]
        )
        x0 = state_estimator(u_v, y_v)[0, :, :]
        y_sim = nn_solution(x0, u_v)


    #%%
    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_v[:, 0, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0, 0], 'b', label='sim')
