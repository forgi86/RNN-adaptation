import torch
import gpytorch
import finite_ntk
import models


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
    data_transf = torch.randn(bsize, seq_len, n_in)
    response_transf = torch.randn(bsize, seq_len, n_out)
    data_new = torch.randn(bsize, seq_len, n_in)

    # randomly initialize a neural network
    #rnn = torch.nn.LSTM(n_in, n_out, 1, batch_first=True)
    rnn = models.WHNet()  # [bsize, seq_len, n_in] -> [bsize, seq_len, n_out]
    rnn_wrapped = models.RNNWrapper(rnn, n_in, n_out)  # [bsize, seq_len*n_in] -> [bsize, seq_len*n_out]

    # reshape input & output data
    # note: underscore denotes "flattened" data: time and channel dimensions are united
    data_transf_ = data_transf.view(*data_transf.shape[:-2], -1)  # [bsize, seq_len*n_in]
    response_transf_ = response_transf.view(*data_transf.shape[:-2], -1)  # [bsize, seq_len*n_out]
    data_new_ = data_new.view(*data_new.shape[:-2], -1)
    response_model_ = rnn_wrapped(data_transf_)
    assert(response_model_.shape == response_transf_.shape)

    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ExactGPModel(data_transf_, response_transf_, gp_lh, rnn_wrapped)

    zeromean_pred = gp_lh(gp_model(data_transf_)).sample()
