import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import gpytorch
import finite_ntk
from open_lstm import OpenLSTM
from models import LSTMWrapperSingleOutput
from torchid import metrics

# Parameters
# output_idx = 1
batch_size = 1
context = 25

# Import evaluation data
u_eval = np.load(os.path.join("data", "cstr", "u_eval.npy")).astype(np.float32)[:batch_size, :, :]  # batch_size, seq_len, input_size
y_eval = np.load(os.path.join("data", "cstr", "y_eval.npy")).astype(np.float32)[:batch_size, :, :]
y_eval_0 = y_eval[..., [0]]  # Single output
y_eval_1 = y_eval[..., [1]]  # Single output

y_torch_new_0 = torch.tensor(y_eval_0[:, 1:, :].reshape(-1, 1), dtype=torch.float)  # Single output
y_context_0 = y_torch_new_0[1:context, :].detach().numpy()
y_torch_new_1 = torch.tensor(y_eval_1[:, 1:, :].reshape(-1, 1), dtype=torch.float)  # Single output
y_context_1 = y_torch_new_1[1:context, :].detach().numpy()

# Import evaluation results from parametric approach
y_sim_02 = np.load(os.path.join("data", "cstr", "02_cstr_eval.npy")).astype(np.float32)

# Import evaluation results after linearization
y_sim_04 = np.load(os.path.join("data", "cstr", "04_cstr_eval_sim.npy")).astype(np.float32)
y_lin_04 = np.load(os.path.join("data", "cstr", "04_cstr_eval_lin.npy")).astype(np.float32)

# Import evaluation results from Jacobian-vector product
y_sim_gd = np.load(os.path.join("data", "cstr", "cstr_eval_gd_sim.npy")).astype(np.float32)
y_lin_gd = np.load(os.path.join("data", "cstr", "cstr_eval_gd_lin.npy")).astype(np.float32)

# Import evaluation results from non-parametric approach
y_lin_gp_0 = np.load(os.path.join("data", "cstr", "GP_predict_0.npy")).astype(np.float32)
upper_conf_0 = np.load(os.path.join("data", "cstr", "GP_upper_conf_0.npy")).astype(np.float32)
lower_conf_0 = np.load(os.path.join("data", "cstr", "GP_lower_conf_0.npy")).astype(np.float32)

y_lin_gp_1 = np.load(os.path.join("data", "cstr", "GP_predict_1.npy")).astype(np.float32)
upper_conf_1 = np.load(os.path.join("data", "cstr", "GP_upper_conf_1.npy")).astype(np.float32)
lower_conf_1 = np.load(os.path.join("data", "cstr", "GP_lower_conf_1.npy")).astype(np.float32)

# Plot all output signals
fig, ax = plt.subplots(2, 1)
ax[0].plot(y_eval[0, 1:, 0], 'k', label="Ground truth")
ax[0].plot(y_sim_02[0, :, 0], 'b', label="LSTM")
# ax[0].plot(y_sim_02[0, :, 0], 'b', label="LSTM-02")
# ax[0].plot(y_sim_04[:, 0], 'y', label="LSTM-04")
ax[0].plot(y_lin_04[:, 0], '--r', label="BLR-LSTM")
# ax[0].plot(y_sim_gd[:, 0], 'm', label="LSTM-GD")
ax[0].plot(y_lin_gd[:, 0], '--c', label="JVP-LSTM")  # Jacobian-Vector Product
ax[0].plot(np.concatenate((y_context_0[:, 0], y_lin_gp_0[:, 0]), axis=0), '--k', label="GP-LSTM")
ax[0].axvline(context-1, color='k', linestyle='--', alpha=0.2)
ax[0].set_ylabel('Y')
ax[0].set_xlabel('X')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(y_eval[0, 1:, 1], 'k')
ax[1].plot(y_sim_02[0, :, 1], 'b')
# ax[1].plot(y_sim_02[0, :, 1], 'b')
# ax[1].plot(y_sim_04[:, 1], 'y')
ax[1].plot(y_lin_04[:, 1], '--r')
# ax[1].plot(y_sim_gd[:, 1], 'm')
ax[1].plot(y_lin_gd[:, 1], '--c')  # Jacobian-Vector Product
ax[1].plot(np.concatenate((y_context_1[:, 0], y_lin_gp_1[:, 0]), axis=0), '--k') # TODO: Run this for 0 intput
ax[1].axvline(context-1, color='k', linestyle='--', alpha=0.2)
ax[1].set_ylabel('Y')
ax[1].set_xlabel('X')
ax[1].grid(True)

plt.show()