import torch
import gpytorch
import finite_ntk
import models
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


# GP Model with NN-based covariance matrix
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


if __name__ == "__main__":

    # Set seed
    np.random.seed(3)
    torch.manual_seed(3)

    # Parameters
    epochs = 200
    seq_len = 11
    bsize = 32
    n_in = 10
    n_hidden = 10
    n_out = 1

    # Data for neural network training
    x_train = torch.from_numpy(0.1 * np.random.randn(bsize, seq_len, n_in))  # [bsize, seq_len, n_in]
    y_train = torch.exp(torch.clone(x_train[:, :, 0]) * 3.)  # bsize, seq_len

    # Data for GP transfer
    x_transf = torch.from_numpy(0.2 * np.random.randn(1, seq_len, n_in))  # bsize is 1 here?
    y_transf = torch.exp(torch.clone(x_transf[:, :, 0]) * 5.)

    # Data to test the GP transfer
    x_eval = torch.from_numpy(0.2 * np.random.randn(1, seq_len, n_in))  # bsize is 1 here?
    y_eval = torch.exp(torch.clone(x_eval[:, :, 0]) * 5.)

    # Base model: lstm
    lstm = models.Model(input_size=n_in, num_layers=1, hidden_size=n_hidden, output_size=n_out).double()

    # Optimizer and loss function
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        print(epoch)
        for index, targets in enumerate(y_train):
            lstm.zero_grad()
            out = lstm(x_train[index].unsqueeze(0))

            loss = loss_function(out.squeeze(), targets)  # MF: fixed dimensions for loss computation
            loss.backward()
            optimizer.step()

    # now do the reshapes so that sequence collapsed onto batch
    lstm_wrapped = models.LSTMWrapper(lstm, n_in, n_out,
                                      seq_len, n_hidden).double()  # [bsize * seq_len, n_in] -> [bsize * seq_len, n_out]
    gp_lh = gpytorch.likelihoods.GaussianLikelihood()

    # 1 - GP model based on the 1st training sequence
    x_train0_f = x_train[0].reshape(1 * seq_len, n_in)
    y_train0_f = y_train[0].view(1 * seq_len, n_out) if n_out > 1 else y_train[0].view(1 * seq_len, )
    y_train0_lstm_f = lstm(x_train[[0], :, :]).squeeze()
    gp_model = ExactGPModel(x_train0_f, y_train0_f, gp_lh, lstm_wrapped)  # GP with NN Kernel
    print("GP model ", gp_model)

    # GP likelihood noise set to mse (as in paper)
    with torch.no_grad():
        mse = loss_function(lstm(x_train), y_train.view(bsize, seq_len, n_out))
    gp_model.likelihood.noise = 1e-4#mse
    print("Noise term", gp_model.likelihood.noise.item())

    gp_lh.eval()
    gp_model.eval()

    print("Mean prediction")
    zeromean_pred = gp_lh(gp_model(x_train0_f)).mean
    print("Confidence")
    lower, upper = gp_lh(gp_model(x_train0_f)).confidence_region()
    lower = lower.reshape(1, seq_len, n_out)
    upper = upper.reshape(1, seq_len, n_out)

    f, ax = plt.subplots()
    ax.plot(y_train[0], "r", label="Observed Data")  # true output
    ax.plot(zeromean_pred.reshape(1, seq_len, n_out)[0].detach().numpy(), "b--", label="GP mean")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(range(seq_len), lower[0].detach().squeeze().numpy(),
                    upper[0].detach().squeeze().numpy(), alpha=0.5, label="GP confidence")
    ax.plot(y_train0_lstm_f.detach().numpy(), "k", label="LSTM model")
#    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.legend()
    ax.set_title("Training data")

    # 2 - GP model based on the transfer sequence

    # reshape input & output data
    x_transf_f = x_transf.view(1 * seq_len, n_in)
    y_transf_f = y_transf.view(1 * seq_len, n_out) if n_out > 1 else y_transf.view(1 * seq_len)
    y_lstm_transf_f = lstm_wrapped(x_transf_f).squeeze()

    print("Transfer")
    gp_model.set_train_data(x_transf_f, y_transf_f, strict=False)

    print("Mean prediction")
    zeromean_pred = gp_lh(gp_model(x_transf_f)).mean
    print("Confidence")
    lower1, upper1 = gp_lh(gp_model(x_transf_f)).confidence_region()
    lower1 = lower1.reshape(1, seq_len, n_out)
    upper1 = upper1.reshape(1, seq_len, n_out)

    f1, ax1 = plt.subplots()
    ax1.plot(y_transf.reshape(1, seq_len, n_out)[0], "r", label="Data")
    ax1.plot(zeromean_pred.reshape(1, seq_len, n_out)[0].detach().numpy(), "b--", label="GP mean")
    ax1.legend()
    ax1.fill_between(range(seq_len), lower1[0].detach().squeeze().numpy(),
                    upper1[0].detach().squeeze().numpy(), alpha=0.5, label="Confidence")
    ax1.plot(y_lstm_transf_f.detach().numpy(), "k", label="LSTM model")
    ax1.legend()
    ax1.set_title("Transfer data")

    # 3 - Transfer on new data
    x_eval_f = x_eval.view(1 * seq_len, n_in)
    y_eval_f = y_eval.view(1 * seq_len, n_out) if n_out > 1 else y_eval.view(1 * seq_len)
    y_lstm_eval_f = lstm_wrapped(x_eval_f).squeeze()

    print("Mean prediction")
    zeromean_pred = gp_lh(gp_model(x_eval_f)).mean
    print("Confidence")
    lower2, upper2 = gp_lh(gp_model(x_eval_f)).confidence_region()
    lower2 = lower2.reshape(1, seq_len, n_out)
    upper2 = upper2.reshape(1, seq_len, n_out)

    f2, ax2 = plt.subplots()
    ax2.plot(y_eval.reshape(1, seq_len, n_out)[0], "r", label="Data")
    ax2.plot(zeromean_pred.reshape(1, seq_len, n_out)[0].detach().numpy(), "b--", label="GP mean")
    ax2.legend()
    # Plot training data as black stars
    # Shade between the lower and upper confidence bounds
    ax2.fill_between(range(seq_len), lower2[0].detach().squeeze().numpy(),
                     upper2[0].detach().squeeze().numpy(), alpha=0.5, label="Confidence")
    ax2.plot(y_lstm_eval_f.detach().numpy(), "k", label="LSTM model")
    ax2.legend()
    ax2.set_title("New data")

    plt.show()
