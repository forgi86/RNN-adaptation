import numpy as np
import matplotlib.pyplot as plt
import loader

if __name__ == "__main__":

    t_new, u_new, y_new, x_new = loader.rlc_loader("eval", noise_std=0.0)
    y_lin_parspace = np.load("y_lin_parspace.npy")
    y_lin_gp_parspace = np.load("y_lin_gp_parspace.npy")

    plt.plot(y_new, 'k')
    plt.plot(y_lin_parspace, 'b')
    plt.plot(y_lin_gp_parspace, 'r')
