import os
import numpy as np
import torch
from torchid.dynonet.module.lti import SisoLinearDynamicalOperator
import matplotlib.pyplot as plt
import time
import loader

if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    model_name = 'IIR'
    add_noise = True
    lr = 1e-4
    num_iter = 30000
    test_freq = 100
    n_batch = 1
    n_b = 2
    n_a = 2

    # In[Load data]
    t, u, y, x = loader.rlc_loader("train")

    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_meas_torch = torch.tensor(y[None, ...], dtype=torch.float)

    # In[Second-order dynamical system custom defined]
    G = SisoLinearDynamicalOperator(n_b, n_a)

    # In[Setup optimizer]
    optimizer = torch.optim.Adam([
        {'params': G.parameters(),    'lr': lr},
    ], lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    y_hat = []
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        y_hat = G(u_torch)

        # Compute fit loss
        err_fit = y_meas_torch - y_hat
        loss_fit = torch.mean(err_fit**2)
        loss = loss_fit

        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss_fit:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    # In[Save model]

    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(G.state_dict(), os.path.join(model_folder, "model.pt"))
    # In[Detach and reshape]
    y_hat = y_hat.detach().numpy()[0, ...]
    y_nonoise = np.copy(x[:, [0]])

    # In[Plot]
    plt.figure()
    plt.plot(t, y_nonoise, 'k', label="$y$")
    plt.plot(t, y, 'r', label="$y_{noise}$")
    plt.plot(t, y_hat, 'b', label="$\hat y$")
    plt.legend()

    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)
    plt.show()


