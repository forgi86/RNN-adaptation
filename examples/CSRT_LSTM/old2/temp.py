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


    def detach_tensor(hd):
        hd[0].detach()
        hd[1].detach()
        return hd


    num_iter = 2000  # gradient-based optimization steps
    lr = 1e-3  # learning rate
    n_skip = 64  # skip initial n_skip samples for training (ignore transient)
    test_freq = 10  # print a message every test_freq iterations

    # u_train.size() == [64, 256, 2] {batch_size = 64, n_steps = 256}
    u_train = torch.tensor(np.load(os.path.join("data", "cstr", "u_train.npy")).astype(np.float32))
    y_train = torch.tensor(np.load(os.path.join("data", "cstr", "y_train.npy")).astype(np.float32))
    # y_train_bi = torch.cat((y_train, torch.flip(y_train, dims=[0, 1])), dim=2)

    # Projection layer:
    # https://stackoverflow.com/questions/37889914/what-is-a-projection-layer-in-the-context-of-neural-networks?rq=1
    model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    LOSS = []
    start_time = time.time()
    h0 = torch.zeros(1, u_train.size()[0], 2).requires_grad_()
    c0 = torch.zeros(1, u_train.size()[0], 16).requires_grad_()

    seq_length = 256

    for itr in range(num_iter):
        optimizer.zero_grad()
        """ # Truncated Backpropagation
        loss_list = []
        hidden = None
        # Open LSTM loop by sending in a few inputs at a time
        for i in range(int(u_train.size()[1]/seq_length)):
            optimizer.zero_grad()

            if hidden is None:
                print("1 here")
                y_sim, hidden = model(u_train[:, i*seq_length: (i+1)*seq_length, :].view(64, seq_length, 2))
            else:
                hidden = detach_tensor(hidden)
                y_sim, hidden = model(u_train[:, i*seq_length: (i+1)*seq_length, :].view(64, seq_length, 2), hidden)

            loss = loss_fn(y_sim, y_train[:, i*seq_length: (i+1)*seq_length, :].view(64, seq_length, 2))
            loss.backward()

            loss_list.append(loss.item())
            optimizer.step()

        # y_sim, _ = model(u_train)

        # Reporting
        LOSS.append(sum(loss_list) / len(loss_list))
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Train Loss {loss:.4f}')
        """

        # Open LSTM loop by sending in a inputs few at a time
        # Simulate using learned model
        y_sim = []
        # state = [(h0.detach(), c0.detach())]
        for i in range(int(u_train.size()[1] / seq_length)):
            y_s, (hn, cn) = model(u_train[:, i * seq_length: (i + 1) * seq_length, :].view(64, seq_length, 2),
                                  (h0.detach(), c0.detach()))
            # state.append((hn.detach(), cn.detach()))
            y_sim.append(y_s)

        # Compute loss
        y_sim = torch.stack(y_sim).reshape(64, 256, 2)
        loss = loss_fn(y_sim[:, n_skip:, :], y_train[:, n_skip:, :])
        # loss = loss_fn(y_sim, y_train)
        loss.backward(retain_graph=True)

        # Reporting
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Train Loss {loss:.4f}')

        optimizer.step()

    """ # Bi-directional LSTM
        for itr in range(num_iter):
            optimizer.zero_grad()

            # Simulate
            y_sim, _ = model(u_train)

            # Compute loss
            loss = loss_fn(y_sim[:, n_skip:, :], y_train_bi[:, n_skip:, :])
            loss.backward()
        """

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_name = "lstm"
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


