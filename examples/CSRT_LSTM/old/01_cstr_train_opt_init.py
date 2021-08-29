import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    num_iter = 10000  # gradient-based optimization steps
    lr = 1e-3  # learning rate
    test_freq = 10
    hidden_size = 16

    u_train = torch.tensor(np.load(os.path.join("../data", "cstr", "u_train.npy")).astype(np.float32))
    y_train = torch.tensor(np.load(os.path.join("../data", "cstr", "y_train.npy")).astype(np.float32))
    batch_size = u_train.shape[0]
    n_out = y_train.shape[2]
    n_in = u_train.shape[2]

    model = nn.LSTM(input_size=2, hidden_size=hidden_size, proj_size=2, num_layers=1, batch_first=True)
    h0 = torch.randn(1, batch_size, n_out, requires_grad=True)
    c0 = torch.randn(1, batch_size, hidden_size, requires_grad=True)

    loss_fn = nn.MSELoss()

    params_net = list(model.parameters())
    params_init = [h0, c0]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_init, 'lr': lr},
    ], lr=lr)

    LOSS = []
    start_time = time.time()
    for itr in range(num_iter):
        optimizer.zero_grad()

        # Simulate
        y_sim, _ = model(u_train, (h0, c0))

        # Compute loss
        loss = loss_fn(y_sim, y_train)
        loss.backward()

        # Reporting
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Tradeoff Loss {loss:.4f}')

        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("../models"):
        os.makedirs("../models")

    model_name = "lstm"
    model_filename = f"{model_name}.pt"
    torch.save(model.state_dict(), os.path.join("../models", model_filename))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.suptitle("Train")
    ax[0].plot(y_train.detach().numpy()[1, :, 0], label='True')
    ax[0].plot(y_sim.detach().numpy()[1, :, 0], label='Fit')
    ax[0].legend()

    ax[1].plot(y_train.detach().numpy()[1, :, 1], label='True')
    ax[1].plot(y_sim.detach().numpy()[1, :, 1], label='Fit')
    ax[1].legend()


    # Test
    u_test = torch.tensor(np.load(os.path.join("../data", "cstr", "u_test.npy")).astype(np.float32))
    y_test = torch.tensor(np.load(os.path.join("../data", "cstr", "y_test.npy")).astype(np.float32))
    y_sim, _ = model(u_test)

    batch_idx = 10
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.suptitle("Test")
    ax[0].plot(y_test.detach().numpy()[batch_idx, :, 0], label='True')
    ax[0].plot(y_sim.detach().numpy()[batch_idx, :, 0], label='Fit')
    ax[0].legend()

    ax[1].plot(y_test.detach().numpy()[batch_idx, :, 1], label='True')
    ax[1].plot(y_sim.detach().numpy()[batch_idx, :, 1], label='Fit')
    ax[1].legend()


