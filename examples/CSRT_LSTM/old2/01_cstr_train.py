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

    num_iter = 2000  # gradient-based optimization steps
    lr = 1e-3  # learning rate
    n_skip = 64  # skip initial n_skip samples for training (ignore transient)
    test_freq = 10  # print a message every test_freq iterations

    u_train = torch.tensor(np.load(os.path.join("data", "cstr", "u_train.npy")).astype(np.float32))
    y_train = torch.tensor(np.load(os.path.join("data", "cstr", "y_train.npy")).astype(np.float32))

    model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    LOSS = []
    start_time = time.time()
    for itr in range(num_iter):
        optimizer.zero_grad()

        # Simulate
        y_sim, _ = model(u_train)

        # Compute loss
        loss = loss_fn(y_sim[:, n_skip:, :], y_train[:, n_skip:, :])
        loss.backward()

        # Reporting
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Train Loss {loss:.4f}')

        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_name = "lstm_old"
    model_filename = f"{model_name}.pt"
    torch.save(model.state_dict(), os.path.join("models", model_filename))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.suptitle("Train")
    batch_idx = 10
    ax[0].plot(y_train.detach().numpy()[batch_idx, :, 0], label='True')
    ax[0].plot(y_sim.detach().numpy()[batch_idx, :, 0], label='Fit')
    ax[0].legend()

    ax[1].plot(y_train.detach().numpy()[batch_idx, :, 1], label='True')
    ax[1].plot(y_sim.detach().numpy()[batch_idx, :, 1], label='Fit')
    ax[1].legend()
    plt.show()

    # Test
    u_test = torch.tensor(np.load(os.path.join("data", "cstr", "u_test.npy")).astype(np.float32))
    y_test = torch.tensor(np.load(os.path.join("data", "cstr", "y_test.npy")).astype(np.float32))
    y_sim, _ = model(u_test)

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.suptitle("Test")
    batch_idx = 10
    ax[0].plot(y_test.detach().numpy()[batch_idx, :, 0], label='True')
    ax[0].plot(y_sim.detach().numpy()[batch_idx, :, 0], label='Fit')
    ax[0].legend()

    ax[1].plot(y_test.detach().numpy()[batch_idx, :, 1], label='True')
    ax[1].plot(y_sim.detach().numpy()[batch_idx, :, 1], label='Fit')
    ax[1].legend()
    plt.show()


