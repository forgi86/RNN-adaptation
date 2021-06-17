import torch
import gpytorch
import finite_ntk
import models
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(3)
torch.manual_seed(3)

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

loss_function = nn.MSELoss()


if __name__ == "__main__":

    epochs = 3000
    seq_len = 10
    bsize = 32
    n_in = 10
    n_hidden = 10
    n_out = 1

    # Data
    data_train = torch.from_numpy(0.1 * np.random.randn(bsize, seq_len, n_in))
    response_train = torch.exp(torch.clone(data_train[:, :, 0]) * 3.)

    data_transf = torch.from_numpy(0.2 * np.random.randn(1, seq_len, n_in))
    response_transf = torch.exp(torch.clone(data_transf[:, :, 0]) * 5.)

    data_transf_test = torch.from_numpy(0.2 * np.random.randn(1, seq_len, n_in))
    response_transf_test = torch.exp(torch.clone(data_transf_test[:, :, 0]) * 5.)


    # randomly initialize a neural network
    lstm = models.Model(input_size=n_in, num_layers=1, hidden_size=n_hidden, output_size=n_out).double()

    print(lstm)

    optimizer = optim.Adam(lstm.parameters(), lr=0.001)

    print(response_train.shape)

    # Very rough training loop
    for epoch in range(epochs):
        print(epoch)
        for index, targets in enumerate(response_train):
            lstm.zero_grad()
            out = lstm(data_train[index].unsqueeze(0))

            loss = loss_function(out, targets)
            loss.backward()
            optimizer.step()

    # now do the rashapes so that sequence collapsed onto batch
    model = models.LSTMWrapper(lstm, n_in, n_out,
                               seq_len, n_hidden).double()  # [bsize * seq_len, n_in] -> [bsize * seq_len, n_out]
    # reshape input & output data
    data_transf_ = data_transf.reshape(1 * seq_len, n_in)
    response_transf_ = response_transf.reshape(1 * seq_len, n_out) if n_out>1 else response_transf.reshape(1 * seq_len, )
    data_transf_test = data_transf_test.reshape(1 * seq_len, n_in)

    gp_lh = gpytorch.likelihoods.GaussianLikelihood()

    data_test = data_train[0].reshape(1 * seq_len, n_in)
    response_test = response_train[0].view(1 * seq_len, n_out)  if n_out>1 else response_train[0].view(1 * seq_len,)

    gp_model = ExactGPModel(data_test, response_test,  gp_lh, model)
    print("GP model ", gp_model)

    # using the GP likelihood
    with torch.no_grad():
        mse = loss_function(lstm(data_train), response_train.view(bsize, seq_len, n_out))
    gp_model.likelihood.noise = mse

    print("Noise term", gp_model.likelihood.noise.item())

    # zeromean_pred = gp_lh(gp_model(data_transf_)).sample()

    gp_lh.eval()
    gp_model.eval()


    print("Mean prediction")
    zeromean_pred = gp_lh(gp_model(data_test)).mean
    print("Confidence")
    lower, upper = gp_lh(gp_model(data_test)).confidence_region()
    lower = lower.reshape(1, seq_len, n_out)
    upper = upper.reshape(1, seq_len, n_out)

    f, ax = plt.subplots()
    ax.plot(response_train[0], "r", label="Data")
    ax.plot(zeromean_pred.reshape(1, seq_len, n_out)[0].detach().numpy(), "b--", label="GP predictions")
    ax.legend()
    # Plot training data as black stars
    # Shade between the lower and upper confidence bounds
    ax.fill_between(range(seq_len), lower[0].detach().squeeze().numpy(),
                    upper[0].detach().squeeze().numpy(), alpha=0.5)

    # ax.set_ylim([-6, 6])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title("Training data")


    # Transfer
    print("Transfer")
    gp_model = ExactGPModel(data_transf_, response_transf_, gp_lh, model)
    gp_model.likelihood.noise = mse

    gp_lh.eval()
    gp_model.eval()
    print("Mean prediction")
    zeromean_pred = gp_lh(gp_model(data_transf_)).mean
    print("Confidence")
    lower1, upper1 = gp_lh(gp_model(data_transf_)).confidence_region()
    lower1 = lower1.reshape(1, seq_len, n_out)
    upper1 = upper1.reshape(1, seq_len, n_out)

    f1, ax1 = plt.subplots()
    ax1.plot(response_transf.reshape(1, seq_len, n_out)[0], "r", label="Data")
    ax1.plot(zeromean_pred.reshape(1, seq_len, n_out)[0].detach().numpy(), "b--", label="GP predictions")
    ax1.legend()
    # Plot training data as black stars
    # Shade between the lower and upper confidence bounds
    ax1.fill_between(range(seq_len), lower1[0].detach().squeeze().numpy(),
                    upper1[0].detach().squeeze().numpy(), alpha=0.5)

    # ax.set_ylim([-6, 6])
    ax1.legend(['Observed Data', 'Mean', 'Confidence'])
    ax1.set_title("Transfer data")

    # Transfer on new data


    print("Mean prediction")
    zeromean_pred = gp_lh(gp_model(data_transf_test)).mean
    print("Confidence")
    lower2, upper2 = gp_lh(gp_model(data_transf_test)).confidence_region()
    lower2 = lower2.reshape(1, seq_len, n_out)
    upper2 = upper2.reshape(1, seq_len, n_out)

    f2, ax2 = plt.subplots()
    ax2.plot(response_transf_test.reshape(1, seq_len, n_out)[0], "r", label="Data")
    ax2.plot(zeromean_pred.reshape(1, seq_len, n_out)[0].detach().numpy(), "b--", label="GP predictions")
    ax2.legend()
    # Plot training data as black stars
    # Shade between the lower and upper confidence bounds
    ax2.fill_between(range(seq_len), lower2[0].detach().squeeze().numpy(),
                     upper2[0].detach().squeeze().numpy(), alpha=0.5)

    # ax.set_ylim([-6, 6])
    ax2.legend(['Observed Data', 'Mean', 'Confidence'])
    ax2.set_title("New data")

    plt.show()
