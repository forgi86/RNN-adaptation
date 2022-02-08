from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import control.matlab
import pandas as pd
import os
from symbolic_RLC import fxu_ODE, fxu_ODE_nl

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)

    # Input characteristics #
    len_sim = 5e-3
    Ts = 1e-6  # 5e-7
    omega_input = 80e3
    std_input = 80

    scale_u = 80.
    scale_x = [90., 3.]
    tau_input = 1 / omega_input
    Hu = control.TransferFunction([1], [1 / omega_input, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts)
    N_sim = int(len_sim // Ts)
    N_skip = int(20 * tau_input // Ts)  # skip initial samples to get a regime sample of d
    N_sim_u = N_sim + N_skip
    e = np.random.randn(N_sim_u)
    te = np.arange(N_sim_u) * Ts
    _, u = control.forced_response(Hu, te, e, return_x=False)
    u = u[N_skip:]
    u = u / np.std(u) * std_input

    t_sim = np.arange(N_sim) * Ts
    u_func = interp1d(t_sim, u, kind='zero', fill_value="extrapolate")


    def f_ODE(t, x):
        u = u_func(t).ravel()
        return fxu_ODE(t, x, u)

    def f_ODE_nl(t, x):
        u = u_func(t).ravel()
        return fxu_ODE_nl(t, x, u)


    x0 = np.zeros(2)
    f_ODE(0.0, x0)
    t_span = (t_sim[0], t_sim[-1])
    y1 = solve_ivp(f_ODE, t_span, x0, t_eval=t_sim)  # Linear
    y2 = solve_ivp(f_ODE_nl, t_span, x0, t_eval=t_sim)  # Non-Linear

    x1 = y1.y.T  # Linear
    x2 = y2.y.T  # Non-Linear

    x1 = x1 / scale_x  # Linear
    x2 = x2 / scale_x  # Non-Linear
    u = u / scale_u

    # In[plot]
    # System State x = [v_c, i_l].T
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t_sim, x1[:, 0], 'b')  # Linear
    ax[0].plot(t_sim, x2[:, 0], 'r')  # Non-Linear
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Capacitor voltage (V)')

    ax[1].plot(t_sim, x1[:, 1], 'b')
    ax[1].plot(t_sim, x2[:, 1], 'r')
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Inductor current (A)')

    ax[2].plot(t_sim, u, 'b')
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('Input voltage (V)')

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.show()

    # In[Save]
    if not os.path.exists("data"):
        os.makedirs("data")

    X = np.hstack((t_sim.reshape(-1, 1), x1, u.reshape(-1, 1), x1[:, 0].reshape(-1, 1)))
    print(f'Data shape LIN: {X.shape}')
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    COL = COL_T + COL_X + COL_U + COL_Y
    df_X = pd.DataFrame(X, columns=COL)
    df_X.to_csv(os.path.join("data", "RLC_data_train_lin.csv"), index=False)

    X = np.hstack((t_sim.reshape(-1, 1), x2, u.reshape(-1, 1), x2[:, 0].reshape(-1, 1)))
    print(f'Data shape NL: {X.shape}')
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    COL = COL_T + COL_X + COL_U + COL_Y
    df_X = pd.DataFrame(X, columns=COL)
    df_X.to_csv(os.path.join("data", "RLC_data_train_nl.csv"), index=False)
