import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from open_lstm import OpenLSTM
from diffutil.products import jvp, unflatten_like
from models import LSTMWrapper
from torchid import metrics


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    n_skip = 0 # skip initial n_skip samples for metrics (ignore transient)
    context = 25
    batch_size = 1

    model_name = "lstm"
    dataset_transf = "transf"
    dataset_eval = "eval"

    # In[Load dataset]
    u_tf = torch.from_numpy(
        np.load(os.path.join("data", "cstr", f"u_{dataset_transf}.npy")).astype(np.float32)[:batch_size, :, :])
    # seq_len, input_size
    y_tf = torch.from_numpy(
        np.load(os.path.join("data", "cstr", f"y_{dataset_transf}.npy")).astype(np.float32)[:batch_size, :, :])
    # seq_len, output_size

    u_new = torch.from_numpy(
        np.load(os.path.join("data", "cstr", f"u_{dataset_eval}.npy")).astype(np.float32)[:batch_size, :, :])
    # seq_len, input_size
    y_new = torch.from_numpy(
        np.load(os.path.join("data", "cstr", f"y_{dataset_eval}.npy")).astype(np.float32)[:batch_size, :, :])
    # seq_len, output_size

    # In[Check dimensions]
    _, seq_len, input_size = u_new.shape
    _, seq_len_, output_size = y_new.shape
    assert(seq_len == seq_len_)

    n_inputs = u_new.shape[-1]

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters
    model = OpenLSTM(context, n_inputs)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Model wrapping]
    model_wrapped = LSTMWrapper(model, seq_len, input_size,batch_s=batch_size)
    u_torch_new = torch.cat((u_new[:, 1:, :], y_new[:, :-1, :]), -1)
    y_torch_new = y_new[:, 1:, :]

    # In[Load theta_lin]
    theta_lin = np.load(os.path.join("models", "theta_lin_gd.npy"))  # closed-form
    # theta_lin = np.load(os.path.join("models", "theta_lin_gd.npy"))  # gradient descent
    # theta_lin = np.load(os.path.join("models", "theta_lin_lbfgs.npy"))  # L-BFGS
    theta_lin = torch.tensor(theta_lin)

    # In[Nominal model output]
    y_sim_new_f = model_wrapped(u_torch_new)
    y_sim_new = y_sim_new_f.reshape(seq_len-1, output_size).detach().numpy()

    # In[Linearized model output]
    theta_lin_f = unflatten_like(theta_lin, tensor_lst=list(model_wrapped.parameters()))
    time_jvp_start = time.time()
    y_lin_new_f = jvp(y_sim_new_f, model_wrapped.parameters(), theta_lin_f)[0]
    time_jvp = time.time() - time_jvp_start
    y_lin_new = y_lin_new_f.reshape((seq_len-1), output_size).detach().numpy()

    # Save output data
    np.save(os.path.join("data", "cstr", "cstr_eval_gd_sim.npy"), y_sim_new)
    np.save(os.path.join("data", "cstr", "cstr_eval_gd_lin.npy"), y_lin_new)

    # In[Plot]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(y_new[0, :, 0].detach().numpy(), 'k', label="Ground truth")
    ax[0].plot(y_tf[0, :, 0].detach().numpy(), 'g', label="Transfer data")
    ax[0].plot(y_sim_new[:, 0], 'b', label="LSTM")
    ax[0].plot(y_lin_new[:, 0], '--r', label="JVP-LSTM")
    ax[0].axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax[0].set_ylabel('Y')
    ax[0].set_xlabel('X')
    ax[0].legend()

    ax[1].plot(y_new[0, :, 1].detach().numpy(), 'k')
    ax[1].plot(y_tf[0, :, 1].detach().numpy(), 'g')
    ax[1].plot(y_sim_new[:, 1], 'b')
    ax[1].plot(y_lin_new[:, 1], '--r')
    ax[1].axvline(context-1, color='k', linestyle='--', alpha=0.2)
    ax[1].set_ylabel('Y')
    ax[1].set_xlabel('X')
    plt.show()

    # R-squared metrics
    R_sq_lin = metrics.r_squared(y_new[0, context+1:, :].detach().numpy(), y_lin_new[context:, :])
    print(f"R-squared linear model: {R_sq_lin}")

    R_sq_sim = metrics.r_squared(y_new[0, context+1:, :].detach().numpy(), y_sim_new[context:, :])
    print(f"R-squared nominal model: {R_sq_sim}")
