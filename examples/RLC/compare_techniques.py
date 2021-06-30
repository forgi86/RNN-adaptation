import numpy as np
import matplotlib.pyplot as plt
import loader

if __name__ == "__main__":

    t_new, u_new, y_new, x_new = loader.rlc_loader("eval", noise_std=0.0)
    y_lin_parspace_naive = np.load("y_lin_parspace_naive.npy")
    y_lin_parspace_lazy = np.load("y_lin_parspace_lazy.npy")
    y_lin_gp_parspace = np.load("y_lin_gp_parspace.npy")

    plt.plot(y_new, 'k', label="True")
    plt.plot(y_lin_parspace_naive, 'b', label="Parspace_naive")
    plt.plot(y_lin_parspace_lazy, 'b--', label="Parspace_lazy")
    plt.plot(y_lin_gp_parspace, 'r', label="GP_parspace")
    plt.legend()
