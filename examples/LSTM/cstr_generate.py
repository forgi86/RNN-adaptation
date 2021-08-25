import os
import numpy as np
from worlds.worlds import CSTR
import matplotlib.pyplot as plt


if __name__ == "__main__":
    system = CSTR()

    batch_size = 64
    n_steps = 256
    u_train, u_test, y_train, y_test = system.generate_data(batch_size, n_steps, flow_period=50)

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(u_train[0, :, 0])  # q
    ax[1].plot(u_train[0, :, 1])  # T
    ax[2].plot(y_train[0, :, 0])  # Ca
    ax[3].plot(y_train[0, :, 1])  # Cr

    if not os.path.exists(os.path.join("data", "cstr")):
        os.makedirs(os.path.join("data", "cstr"))

    np.save(os.path.join("data", "cstr", "u_train.npy"), u_train)
    np.save(os.path.join("data", "cstr", "u_test.npy"), u_test)
    np.save(os.path.join("data", "cstr", "y_train.npy"), y_train)
    np.save(os.path.join("data", "cstr", "y_test.npy"), y_test)