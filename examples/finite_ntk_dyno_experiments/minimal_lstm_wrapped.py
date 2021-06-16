import torch
import gpytorch
import finite_ntk
import models
import numpy as np

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
    seq_len = 20
    bsize = 32
    n_in = 1
    n_out = 1

    # Data
    data_train = torch.from_numpy(0.01 * np.random.randn(bsize, seq_len, n_in))
    response_train = torch.exp(torch.clone(data_train[:, :, 0]) * 3.)

    data_transf = torch.from_numpy(0.01 * np.random.randn(bsize, seq_len, n_in))
    response_transf = torch.exp(torch.clone(data_transf[:, :, 0]) * 5.)

    data_transf_test = torch.from_numpy(0.01 * np.random.randn(bsize, seq_len, n_in))
    response_transf_test = torch.exp(torch.clone(data_transf_test[:, :, 0]) * 5.)


    # randomly initialize a neural network
    lstm = torch.nn.LSTM(n_in, n_out, 1, batch_first=True)
    lstm_wrapped = models.LSTMWrapper(lstm, n_in, n_out, seq_len).double()  # [bsize, seq_len*n_in] -> [bsize, seq_len*n_out]
    # reshape input & output data
    data_transf_ = data_transf.reshape(bsize*seq_len, n_in)
    response_transf_ = response_transf.reshape(bsize*seq_len, n_out)
    data_transf_test = data_transf_test.reshape(bsize*seq_len, n_out)
    response_model_ = lstm_wrapped(data_transf_)

    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ExactGPModel(data_transf_, response_transf_, gp_lh, lstm_wrapped)

    zeromean_pred = gp_lh(gp_model(data_transf_)).sample()

