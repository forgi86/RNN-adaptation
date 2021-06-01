import numpy as np
import torch
import gpytorch
import finite_ntk
import matplotlib.pyplot as plt

n_hidden = 100
n_in = 10
n_seq = 30
n_out = 1

data_transf = 0.1*torch.randn(n_seq, n_in)
response_transf = torch.clone(data_transf[:,0]) * 3.

# randomly initialize a neural network
model = torch.nn.Sequential(torch.nn.Linear(n_in, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, n_out))

class ExactGPModel(gpytorch.models.ExactGP):
    # exact RBF Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(model=model)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

gp_lh = gpytorch.likelihoods.GaussianLikelihood()
gp_model = ExactGPModel(data_transf, response_transf, gp_lh, model)
print("GP model ", gp_model)

# draw a sample from the GP with kernel given by Jacobian of model
# samples = gp_lh(gp_model(data_transf)).sample()
# gp_lh.eval()
# gp_model.eval()
# zeromean_pred = gp_lh(gp_model(data_transf))
zeromean_pred = gp_model(data_transf).mean
conf = gp_model(data_transf).stddev.mul_(3.)

f, ax = plt.subplots()
ax.plot(response_transf, "r", label="Data")
ax.plot(zeromean_pred.detach().numpy(), "b--", label="GP predictions")
ax.legend()
# Plot training data as black stars
# Shade between the lower and upper confidence bounds
ax.plot(zeromean_pred.detach().numpy() + conf.detach().numpy(), "g--")
ax.plot(zeromean_pred.detach().numpy() - conf.detach().numpy(), "g--")
ax.set_ylim([-6, 6])
ax.legend(['Observed Data', 'Mean', 'Confidence'])


plt.show()
