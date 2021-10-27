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


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, model, use_linearstrategy=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        print("Initial EGP")
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(model=model, use_linearstrategy=use_linearstrategy)
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.model = model
        self.use_linearstrategy = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def estimate_state(self, train_x, train_y, nstep):
        with torch.no_grad():
            self.model.estimate_state(train_x, train_y, nstep, output_size=n_outputs)
        self.__init__(self.train_x, self.train_y, self.likelihood, self.model, use_linearstrategy=False)
        self.covar_module = finite_ntk.lazy.NTK(model=self.model, use_linearstrategy=use_linearstrategy)
        self.eval()

if __name__ == '__main__':

    time_start = time.time()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    output_idx = 1  # must run the code twice for output 0/1
    use_linearstrategy = False
    sigma = 0.03
    n_skip = 0 # skip initial n_skip samples for transfer (ignore transient)
    context = 25
    model_name = "lstm"

    # In[Load dataset]
    u = torch.tensor(np.load(os.path.join("data", "cstr", "u_transf.npy")).astype(np.float32)[0, :, :])  # seq_len, input_size
    y = torch.tensor(np.load(os.path.join("data", "cstr", "y_transf.npy")).astype(np.float32)[0, :, :])  # seq_len, output_size
    # y = y[..., [output_idx]]

    # In[Check dimensions]
    batch_size = 1
    seq_len, input_size = u.shape
    seq_len_, output_size = y.shape
    assert (seq_len == seq_len_)

    n_inputs = u.shape[-1]
    n_outputs = y.shape[-1]

    u_t = u[1:, :]
    y_t = y[1:, [output_idx]]

    # In[Load LSTM model]
    # Setup neural model structure and load fitted model parameters

    model = OpenLSTM(context, n_inputs, is_estimator=False)
    model_filename = f"{model_name}.pt"
    model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # model_wrapped = LSTMWrapperSingleOutput(model, seq_len, input_size, output_idx)
    u_torch = torch.tensor(u_t, dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y_t, dtype=torch.float)
    #u_torch_f = torch.clone(u_torch.view((input_size * seq_len, 1)))  # [bsize*seq_len, n_in]
    #y_torch_f = torch.clone(y_torch.view(output_size * seq_len, 1))  # [bsize*seq_len, ]
    #y_f = y_torch_f.detach().numpy()

    # In[Adaptation in function/parameter space]
    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_lh.noise = sigma**2
    model_wrapped = LSTMWrapperSingleOutput(model, seq_len, input_size, output_idx)
    model_wrapped.estimate_state(u_t, y, nstep=25, output_size=n_outputs)
    gp_model = ExactGPModel(u_torch, y_torch.squeeze(),
                            gp_lh, model_wrapped, use_linearstrategy=use_linearstrategy)

    # No GP training (we consider the kernel (hyper)parameters fixed.
    # We may think of training the measurement noise by mll optimization...
    gp_model.eval()
    gp_lh.eval()

    # In[Evaluate the GP-like model on new data]
    u_new = torch.tensor(np.load(os.path.join("data", "cstr", "u_eval.npy")).astype(np.float32)[0, :, :])  # seq_len, input_size
    y_new = torch.tensor(np.load(os.path.join("data", "cstr", "y_eval.npy")).astype(np.float32)[0, :, :])  # seq_len, output_size
    # y_new = y_new[..., [output_idx]]

    u_new_ = torch.unsqueeze(u_new, 0)
    y_new_ = torch.unsqueeze(y_new, 0)

    u_torch_new = u_new_[:, 1:, :]
    y_torch_new = y_new_[:, 1:, [output_idx]]

    # u_torch_new = torch.tensor(u, dtype=torch.float)
    # y_torch_new = torch.tensor(y, dtype=torch.float)
#    u_torch_new_f = torch.clone(u_torch_new.view((input_size * seq_len, 1)))  # [bsize*seq_len, n_in]
#    y_torch_new_f = torch.clone(y_torch_new.view(output_size * seq_len, 1))  # [bsize*seq_len, ]

    with torch.no_grad():
        y_sim_new = model_wrapped(u_torch_new)
    y_sim_new = y_sim_new.detach().numpy()

    time_inference_start = time.time()
    with gpytorch.settings.fast_pred_var():  #, gpytorch.settings.max_cg_iterations(4000), gpytorch.settings.cg_tolerance(0.1):
        gp_model.estimate_state(u_torch_new, y_new_, nstep=25)
        predictive_dist = gp_model(u_torch_new)
        y_lin_new = predictive_dist.mean.data
    y_lin_new = y_lin_new[..., None].detach().numpy()
    time_inference = time.time() - time_inference_start

    # In[Plot]
    fig = plt.figure()
    plt.plot(y_new[:, 0], 'k', label="True")
    plt.plot(y_sim_new[:, 0], 'r', label="Sim")
    plt.plot(y_lin_new[:, 0], 'b', label="Lin")
    plt.legend()
    plt.grid()
    plt.show()

    # In[]
    # R-squared metrics
    R_sq_lin = metrics.r_squared(y_new[n_skip:, :], y_lin_new[n_skip:, :])
    print(f"R-squared linear model: {R_sq_lin}")

    R_sq_sim = metrics.r_squared(y_new[n_skip:, :], y_sim_new[n_skip:, :])
    print(f"R-squared nominal model: {R_sq_sim}")

    print(f"\nInference time: {time_inference:.2f}")

