import numpy as np
import torch
import gpytorch
import finite_ntk
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

np.random.seed(1)
torch.manual_seed(1)

epochs = 1000
n_hidden = 100
n_in = 10
n_seq = 30
n_out = 1
loss_function = nn.MSELoss()

# Data
data_transf = 0.1*torch.randn(n_seq, n_in)
response_transf = torch.clone(data_transf[:, 0]) * 3.

# randomly initialize a neural network
model = torch.nn.Sequential(torch.nn.Linear(n_in, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, n_out))

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Very rough training loop
for epoch in range(epochs):
    print(epoch)
    for index, targets in enumerate(response_transf):
        model.zero_grad()

        out = model(data_transf[index])

        loss = loss_function(out, targets)
        loss.backward()
        optimizer.step()


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

# TODO: This term needs to be trained (with or after the network)
# using the GP likelihood
gp_model.likelihood.noise = 1E-4 

print("Noise term", gp_model.likelihood.noise.item())

gp_lh.eval()
gp_model.eval()
# zeromean_pred = gp_lh(gp_model(data_transf))
zeromean_pred = gp_lh(gp_model(data_transf)).mean
# conf = gp_lh(gp_model(data_transf)).stddev.mul_(2.)
lower, upper = gp_lh(gp_model(data_transf)).confidence_region()

f, ax = plt.subplots()
ax.plot(response_transf, "r", label="Data")
ax.plot(zeromean_pred.detach().numpy(), "b--", label="GP predictions")
ax.legend()
# Plot training data as black stars
# Shade between the lower and upper confidence bounds
ax.fill_between(range(n_seq), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)

# ax.set_ylim([-6, 6])
ax.legend(['Observed Data', 'Mean', 'Confidence'])

zeromean_pred = gp_lh(gp_model(data_transf)).mean
conf = gp_lh(gp_model(data_transf)).stddev.mul_(3.)

f1, ax1 = plt.subplots()
ax1.plot(response_transf, "r", label="Data")
ax1.plot(model(data_transf).detach().numpy(), "b--", label="NN predictions")
ax1.legend()

plt.show()
