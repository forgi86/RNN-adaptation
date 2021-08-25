import os
import numpy as np
from worlds.worlds import CSTR
import matplotlib.pyplot as plt


if __name__ == "__main__":
    system = CSTR()

    # Generate train and test datasets
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

    # Generate transfer and evaluation datasets
    batch_size = 1
    n_steps = 1024
    u_transf, u_eval, y_transf, y_eval = system.generate_data(batch_size, n_steps, flow_period=50)
    np.save(os.path.join("data", "cstr", "u_transf.npy"), u_transf)
    np.save(os.path.join("data", "cstr", "u_eval.npy"), u_eval)
    np.save(os.path.join("data", "cstr", "y_transf.npy"), y_transf)
    np.save(os.path.join("data", "cstr", "y_eval.npy"), y_eval)
