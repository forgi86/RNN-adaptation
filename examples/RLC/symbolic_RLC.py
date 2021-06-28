from sympy import symbols, collect, cancel, init_printing, fraction
import numpy as np
import matplotlib.pyplot as plt
import os


# In[Symbols of the RLC circuit]

R = symbols('R')
L = symbols('L')
C = symbols('C')
s = symbols('s')

# In[Impedances]

ZR = R
ZL = s*L
ZC = 1/(s*C)

ZRL = ZR + ZL  # series R and L

G1 = 1/(ZRL)


G2 = ZC/(ZRL + ZC)
G2sym = 1/(L*C)/(s**2 + R/L*s + 1/(L*C))


# In[Impedances]
z = symbols('z')
Td = symbols('Td')

s_subs = 2/Td * (z-1)/(z+1)  # Tustin transform of the laplace variable s

G2d = G2.subs(s, s_subs)
G2d_simple = collect(cancel(G2d), z)


# In[Substitution]
R_nom = 3
L_nom = 50e-6
C_nom = 270e-9
Td_nom = 1e-6


def saturation_formula(current_abs):
    sat_ratio = (1/np.pi*np.arctan(-5*(current_abs-5))+0.5)*0.9 + 0.1 
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

    init_printing(use_unicode=True)

    x = np.zeros(2)
    u = np.zeros(1)
    dx = fxu_ODE_nl(0.0, x, u)

    sym = [R, L, C, Td]
    vals = [R_nom, L_nom, C_nom, Td_nom]

    G2d_val = G2d_simple.subs(zip(sym, vals))
    G2d_num, G2d_den = fraction(G2d_val)

    # In[Get coefficients]

    num_coeff = G2d_num.collect(z).as_coefficients_dict()
    den_coeff = G2d_den.collect(z).as_coefficients_dict()

    G2d_num = G2d_num / den_coeff[z**2]  # Monic numerator
    G2d_den = G2d_den / den_coeff[z**2]  # Monic denominator
    G2d_monic = G2d_num/G2d_den  # Monic trasnfer function

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
