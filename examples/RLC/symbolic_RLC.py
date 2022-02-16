import numpy as np
import matplotlib.pyplot as plt
import os


# In[Substitution]
R_nom = 3
L_nom = 50e-6
C_nom = 270e-9
Td_nom = 1e-6


def saturation_formula(current_abs):
    sat_ratio = (1/np.pi*np.arctan(-1.0*(current_abs-5))+0.5)*0.9 + 0.1
    return sat_ratio


def fxu_ODE(t, x, u, params={}):
    C_val = params.get('C', C_nom)
    R_val = params.get('R', R_nom)
    L_val = params.get('L', L_nom)

    A = np.array([[0.0, 1.0/C_val],
                  [-1/L_val, -R_val/L_val]
                  ])
    B = np.array([[0.0], [1.0/L_val]])

    return A @ x + B @ u


def fxu_ODE_nl(t, x, u, params={}):

    C_val = params.get('C', C_nom)
    R_val = params.get('R', R_nom)
    L_val = params.get('L', L_nom)

    I_abs = np.abs(x[1])
    L_val_mod = L_val * saturation_formula(I_abs)
    R_val_mod = R_val
    C_val_mod = C_val

    A = np.array([[0.0, 1.0/C_val_mod],
                  [-1/(L_val_mod), -R_val_mod/L_val_mod]
                  ])
    B = np.array([[0.0], [1.0/L_val_mod]])
    return A @ x + B @ u


if __name__ == '__main__':

    x = np.zeros(2)
    u = np.zeros(1)
    dx = fxu_ODE_nl(0.0, x, u)

    I = np.arange(0., 20., 0.1)

    # Save model
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 3))
    ax.plot(I, L_nom * 1e6 * saturation_formula(I), 'k')
    ax.grid(True)
    ax.set_xlabel('Inductor current $i_L$ (A)', fontsize=14)
    ax.set_ylabel('Inductance $L$ ($\mu$H)', fontsize=14)
    fig.savefig(os.path.join("fig", "RLC_characteristics.pdf"), bbox_inches='tight')
