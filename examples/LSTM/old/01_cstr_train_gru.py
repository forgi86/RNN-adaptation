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

    num_iter = 5000  # gradient-based optimization steps
    lr = 1e-3  # learning rate
    test_freq = 10

    u_train = torch.tensor(np.load(os.path.join("../data", "cstr", "u_train.npy")).astype(np.float32))
    y_train = torch.tensor(np.load(os.path.join("../data", "cstr", "y_train.npy")).astype(np.float32))
    #x_0 = np.array([0.692, 0.287])
    h_0 = y_train[:, 0, :]
    h_0 = h_0[None, :, :].clone()
    h_0.requires_grad_(True)


    model = nn.GRU(input_size=2, hidden_size=2, num_layers=1, batch_first=True)
    loss_fn = nn.MSELoss()

    # Setup optimizer
    params_net = list(model.parameters())
    params_init = [h_0]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_init, 'lr': 10*lr},
    ], lr=lr)

    LOSS = []
    start_time = time.time()
    for itr in range(num_iter):
        optimizer.zero_grad()

        # Simulate
        y_sim, _ = model(u_train, h_0)

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
    ax[0].plot(y_train.detach().numpy()[0, :, 0], label='True')
    ax[0].plot(y_sim.detach().numpy()[0, :, 0], label='Fit')
    ax[0].legend()

    ax[1].plot(y_train.detach().numpy()[0, :, 1], label='True')
    ax[1].plot(y_sim.detach().numpy()[0, :, 1], label='Fit')
    ax[1].legend()

    plt.figure()
    plt.plot(LOSS)
