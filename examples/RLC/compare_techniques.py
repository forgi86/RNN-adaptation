import numpy as np
import matplotlib.pyplot as plt
import loader

if __name__ == "__main__":

    t_new, u_new, y_new, x_new = loader.rlc_loader("eval", noise_std=0.0)
    y_sim = np.load("y_sim.npy")
    y_lin_parspace_naive = np.load("y_lin_parspace_naive.npy")
    y_lin_parspace_lazy = np.load("y_lin_parspace_lazy.npy")
    y_lin_gp_parspace = np.load("y_lin_gp_parspace.npy")
    y_lin_gp_funspace = np.load("y_lin_gp_funspace.npy")

    plt.plot(y_new, 'k', label="True")
    plt.plot(y_sim, 'r', label="Sim (no transfer)")
    plt.plot(y_lin_parspace_naive, 'b', label="parspace_naive")
    plt.plot(y_lin_parspace_lazy, 'b-*', label="parspace_lazy")
    plt.plot(y_lin_gp_parspace, 'b-o', label="GP_parspace")
    plt.plot(y_lin_gp_funspace, 'c', label="GP_funspace")
    plt.legend()
