import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchid.statespace.module.ssmodels_ct import NeuralStateSpaceModel
from torchid.statespace.module.ss_simulator_ct import ForwardEulerSimulator
from loader import rlc_loader


# Truncated simulation error minimization method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 10000  # 10000  # gradient-based optimization steps
    seq_len = 256  # subsequence length m
    batch_size = 16  # batch size q
    t_fit = 2e-3  # fitting on t_fit ms of data
    alpha = 1.0  # regularization weight
    lr = 1e-4  # learning rate
    test_freq = 500  # print message every test_freq iterations
    var_idx = 0  # voltage
    add_noise = True

    # Column names in the dataset
    t, u, y, x = rlc_loader("transfer", "nl", noise_std=0.1)

    # model_name = 'ss_model_retrain'
    model_name = 'ss_model'

    # Get fit data #
    ts = t[1] - t[0]
    n_fit = int(t_fit // ts)  # x.shape[0]
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    time_torch_fit = torch.from_numpy(time_fit)
    x_hidden_fit = torch.tensor(np.c_[y_fit, np.zeros_like(y_fit)], requires_grad=True)

    # Setup neural model structure and load fitted model parameters
    # Then start retraining on transfer data using the original parameters as initialization
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=50)
    nn_solution = ForwardEulerSimulator(ss_model)
    model_filename = f"{model_name}.pt"
    # Initialize model with nominal parameters
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Setup optimizer
    params_net = list(nn_solution.ss_model.parameters())     # Initialize model with nominal parameters
    params_hidden = [x_hidden_fit]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': 10*lr},
    ], lr=lr)

    # Batch extraction function
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = y_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64),
                                       batch_size, replace=False)  # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len)  # batch samples indices
        batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_x_hidden = x_hidden_fit[[batch_idx]]
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_y = torch.tensor(y_fit[batch_idx])

        return batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden


    # Scale loss with respect to the initial one
    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution(batch_x0_hidden, batch_u)
        batch_y_sim = batch_x_sim[..., [var_idx]]
        traced_nn_solution = torch.jit.trace(nn_solution, (batch_x0_hidden, batch_u))
        err_init = batch_y_sim - batch_y
        scale_error = torch.sqrt(torch.mean(err_init**2, dim=(0, 1)))

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []
    start_time = time.time()
    # Training loop

    scripted_nn_solution = torch.jit.script(nn_solution)
    indx = 0
    save15 = True
    # for itr in range(0, num_iter):
    while True:
        optimizer.zero_grad()

        # Simulate
        batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
        #  batch_x_sim = traced_nn_solution(batch_x0_hidden, batch_u) # 52 seconds RK | 13 FE
        #  batch_x_sim = nn_solution(batch_x0_hidden, batch_u) # 70 seconds RK | 13 FE
        batch_x_sim = scripted_nn_solution(batch_x0_hidden, batch_u)  # 71 seconds RK | 13 FE

        # Compute fit loss
        batch_y_sim = batch_x_sim[..., [var_idx]]
        err_fit = batch_y_sim - batch_y
        err_fit_scaled = err_fit/scale_error
        loss_fit = torch.mean(err_fit_scaled**2)

        # Compute consistency loss
        err_consistency = batch_x_sim - batch_x_hidden
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        # Compute trade-off loss
        loss = loss_fit + alpha*loss_consistency

        # Statistics
        LOSS.append(loss.item())
        LOSS_CONSISTENCY.append(loss_consistency.item())
        LOSS_FIT.append(loss_fit.item())
        if indx % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {indx} | Tradeoff Loss {loss:.4f} '
                      f'Consistency Loss {loss_consistency:.4f} Fit Loss {loss_fit:.4f}')
                # print("Time: ", time.time() - start_time)

        if save15 and (time.time() - start_time) > 15.0:
            # Save model trained for 15 sec
            save15 = False
            print("Model save: 15", save15)
            torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", "ss_model_nomtransfer_15_.pt"))

        indx = indx+1
        # Optimize
        loss.backward()
        optimizer.step()

        if LOSS[-1] < 0.014 or indx > 10000:  # Run until loss < 1.1%
            print(LOSS[-1])
            break

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    # if add_noise:
    #     model_filename = f"model_SS_{seq_len}step_noise_V.pt"
    #     hidden_filename = f"hidden_SS_{seq_len}step_noise_V.pt"
    # else:
    #     model_filename = f"model_SS_{seq_len}step_nonoise_V.pt"
    #     hidden_filename = f"hidden_SS_{seq_len}step_nonoise_V.pt"

    model_filename = "ss_model_nomtransfer_.pt"
    hidden_filename = "ss_hidden_nomtransfer_.pt"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))
    torch.save(x_hidden_fit, os.path.join("models", hidden_filename))

    t_val = 5e-3
    n_val = int(t_val // ts)  # x.shape[0]

    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]

    x0_val = np.zeros(2, dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(input_data_val)
    x_true_torch_val = torch.from_numpy(state_data_val)

    with torch.no_grad():
        x_sim_torch_val = nn_solution(x0_torch_val[None, :], u_torch_val[:, None, :])
        x_sim_torch_val = x_sim_torch_val.squeeze(1)

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.array(x_true_torch_val[:, 0]), label='True')
    ax[0].plot(np.array(x_sim_torch_val[:, 0]), label='Fit')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(x_true_torch_val[:, 1]), label='True')
    ax[1].plot(np.array(x_sim_torch_val[:, 1]), label='Fit')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch_val), label='Input')
    ax[2].grid(True)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.plot(LOSS_CONSISTENCY, 'r', label='CONSISTENCY')
    ax.plot(LOSS_FIT, 'b', label='FIT')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")
    # plt.show()

    if add_noise:
        fig_name = f"RLC_SS_loss_{seq_len}step_noise.pdf"
    else:
        fig_name = f"RLC_SS_loss_{seq_len}step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    x_hidden_fit_np = x_hidden_fit.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x[:, 0], 'k', label='True')
    #ax[0].plot(x_fit[:, 0], 'b', label='Measured')
    ax[0].plot(x_hidden_fit_np[:, 0], 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(x[:, 1], 'k', label='True')
    #ax[1].plot(x_fit[:, 1], 'b', label='Measured')
    ax[1].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[1].legend()
    ax[1].grid(True)
    # plt.show()

