import torch
import gpytorch
import finite_ntk
import models


class ExactGPModel(gpytorch.models.ExactGP):
    # exact RBF Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(model=model, use_linearstrategy=False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    seq_len = 64
    bsize = 1
    n_in = 1
    n_out = 1  # must be on (GP model is only scalar)

    x_transf = torch.randn(bsize, seq_len, n_in)
    y_transf = torch.randn(bsize, seq_len, n_out)
    x_new = torch.randn(bsize, seq_len, n_in)

    # randomly initialize a neural network
    dyno = models.WHNet()  # [bsize, seq_len, n_in] -> [bsize, seq_len, n_out]
    dyno_wrapped = models.DynoWrapper(dyno, n_in, n_out)  # [bsize*seq_len, n_in] -> [bsize*seq_len, n_out]

    # reshape input & output data
    # note: underscore denotes "flattened" data: time and channel dimensions are united
    x_transf_f = torch.clone(x_transf.view(1 * seq_len, n_in))  # [bsize*seq_len, n_in]
    y_transf_f = torch.clone(y_transf.view(1 * seq_len, ))  # [bsize*seq_len, ]
    x_new_ = torch.clone(x_new.view(1 * seq_len, n_in))
    response_model_ = dyno_wrapped(x_transf_f)
    assert(response_model_.shape == y_transf_f.shape)

    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ExactGPModel(x_transf_f, y_transf_f, gp_lh, dyno_wrapped)

    gp_model.eval()
    gp_lh.eval()
    zeromean_pred = gp_lh(gp_model(x_transf_f)).sample()
