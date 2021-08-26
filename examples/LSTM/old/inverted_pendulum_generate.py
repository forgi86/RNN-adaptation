import torch
import numpy as np

from worlds.world_inverted_pendulum import InvertedPendulumWorld

import matplotlib.pyplot as plt

if __name__ == "__main__":

    batch_size = 1
    n_x = 2
    n_u = 1
    n_steps = 1000

    m = 1.0
    l = 1.0
    inv_pend = InvertedPendulumWorld(mass=m, length=l)

    x_0 = torch.tensor([[np.pi/3, 0]])     # alpha, omega
    K = np.ones_like(x_0, shape=(n_u, n_x))  # need to find a good controller!

    x_seq = inv_pend.closed_loop_sim(x_0, K, n_steps)
    plt.plot(x_seq)
