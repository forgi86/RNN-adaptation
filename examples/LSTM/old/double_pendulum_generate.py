import torch
import numpy as np

from worlds.world_double_pendulum import DoublePendulumWorld
import matplotlib.pyplot as plt

if __name__ == "__main__":

    batch_size = 1
    n_x = 4
    n_u = 2
    n_steps = 100

    m = [1.0, 2.0]
    l = [1.0, 2.0]
    cm = [0.0, 1.0]
    double_pend = DoublePendulumWorld(mass=m, length=l, center_of_mass=cm)

    x_0 = torch.zeros(batch_size, n_x)
    K = np.ones_like(x_0, shape=(n_u, n_x)) #torch.ones(n_x)

    x_seq = double_pend.closed_loop_sim(x_0, K, n_steps)   # shape problem. Needs perhaps a specific closed_loop_sim