import numpy as np
import torch
import gpytorch
import finite_ntk
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

np.random.seed(1)
torch.manual_seed(1)

epochs = 100
n_hidden = 100
n_in = 10
batch = 30
n_out = 1
loss_function = nn.MSELoss()

# Data

data_train = torch.sort(0.1 * torch.randn(batch, n_in), 0)[0]
response_train = torch.exp(torch.clone(data_train[:, 0]) * 3.)

data_transf = torch.sort(.2 * torch.randn(batch, n_in), 0)[0]
response_transf = torch.exp(torch.clone(data_transf[:, 0]) * 5.)

data_transf_test = torch.sort(.2 * torch.randn(batch, n_in), 0)[0]
response_transf_test = torch.exp(torch.clone(data_transf_test[:,  0]) * 5.)

# randomly initialize a neural network
model = torch.nn.Sequential(torch.nn.Linear(n_in, n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden, n_out))

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Very rough training loop
for epoch in range(epochs):
    print(epoch)
    for index, targets in enumerate(response_train):
        model.zero_grad()

        out = model(data_train[index])

        loss = loss_function(out, targets)
        loss.backward()
        optimizer.step()


class ExactGPModel(gpytorch.models.ExactGP):
    # exact RBF Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(model=model, use_linearstrategy=True)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

gp_lh = gpytorch.likelihoods.GaussianLikelihood()
gp_model = ExactGPModel(data_train, response_train, gp_lh, model)
print("GP model ", gp_model)

# draw a sample from the GP with kernel given by Jacobian of model
# samples = gp_lh(gp_model(data_transf)).sample()

# TODO: This term needs to be trained (with or after the network)
# using the GP likelihood
# with torch.no_grad():
#     mse = loss_function(model(data_train), response_train)
# gp_model.likelihood.noise = mse

print("Noise term", gp_model.likelihood.noise.item())

gp_lh.eval()
gp_model.eval()
print("mean fun")
# zeromean_pred = gp_lh(gp_model(data_transf))
zeromean_pred = gp_lh(gp_model(data_train)).mean
# conf = gp_lh(gp_model(data_transf)).stddev.mul_(2.)
print("confidence")

lower, upper = gp_lh(gp_model(data_train)).confidence_region()

f, ax = plt.subplots()
ax.plot(response_train, "r", label="Data")
ax.plot(zeromean_pred.detach().numpy(), "b--", label="GP predictions")
ax.legend()
# Plot training data as black stars
# Shade between the lower and upper confidence bounds
ax.fill_between(range(batch), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)

# ax.set_ylim([-6, 6])
ax.legend(['Observed Data', 'Mean', 'Confidence'])
exit()
# zeromean_pred = gp_lh(gp_model(data_train)).mean
# conf = gp_lh(gp_model(data_train)).stddev.mul_(3.)

f1, ax1 = plt.subplots()
ax1.plot(response_train, "r", label="Data")
ax1.plot(model(data_train).detach().numpy(), "b--", label="NN predictions")
ax1.legend()

# Doing the transfer
gp_model = ExactGPModel(torch.cat((data_train, data_transf), dim=0),
                        torch.cat((response_train, response_transf), dim=0), gp_lh, model)
# gp_model = ExactGPModel(data_transf,
#                         response_transf, gp_lh, model)
# print("GP model ", gp_model)

# draw a sample from the GP with kernel given by Jacobian of model
# samples = gp_lh(gp_model(data_transf)).sample()

# using the GP likelihood
with torch.no_grad():
    mse = loss_function(model(data_train), response_train)
gp_model.likelihood.noise = mse
gp_lh.eval()
gp_model.eval()

# zeromean_pred = gp_lh(gp_model(data_transf))
zeromean_pred = gp_lh(gp_model(data_transf_test)).mean
# conf = gp_lh(gp_model(data_transf)).stddev.mul_(2.)
lower, upper = gp_lh(gp_model(data_transf_test)).confidence_region()
exit()

#
f2, ax2 = plt.subplots()
ax2.set_title("transfer")
ax2.plot(response_transf_test, "r", label="Data")
ax2.plot(zeromean_pred.detach().numpy(), "b--", label="GP predictions")
ax2.legend()
# Plot training data as black stars
# Shade between the lower and upper confidence bounds
ax2.fill_between(range(batch), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)

# ax.set_ylim([-6, 6])
ax2.legend(['Observed Data', 'Mean', 'Confidence'])

# zeromean_pred = gp_lhgp_model(data_transf_test)).mean
# conf = gp_lh(gp_model(data_transf_test)).stddev.mul_(3.)

f3, ax3 = plt.subplots()
ax3.set_title("transfer")
ax3.plot(response_transf_test, "r", label="Data")
ax3.plot(model(data_transf_test).detach().numpy(), "b--", label="NN predictions")
ax3.legend()

f4, ax4 = plt.subplots()
ax4.set_title("new data")
ax4.plot(response_transf, "r", label="Data")
ax4.plot(model(data_transf).detach().numpy(), "b--", label="NN predictions")
ax4.legend()


zeromean_pred = gp_lh(gp_model(data_transf)).mean
# conf = gp_lh(gp_model(data_transf)).stddev.mul_(2.)
lower, upper = gp_lh(gp_model(data_transf)).confidence_region()


f5, ax5 = plt.subplots()
ax5.set_title("new data")
ax5.plot(response_transf, "r", label="Data")
ax5.plot(zeromean_pred.detach().numpy(), "b--", label="GP predictions")
ax5.legend()
# Plot training data as black stars
# Shade between the lower and upper confidence bounds
ax5.fill_between(range(batch), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)

# ax.set_ylim([-6, 6])
ax5.legend(['Observed Data', 'Mean', 'Confidence'])

plt.show()
