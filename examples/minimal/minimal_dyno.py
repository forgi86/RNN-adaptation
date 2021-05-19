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
    N = 2
    bsize = 100
    data_transf = torch.randn(bsize, N)
    response_transf = torch.randn(bsize, N)
    data_new = torch.randn(bsize, N)

    # randomly initialize a neural network
    model = models.WHNet()

    response_model = model(data_transf)

    gp_lh = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ExactGPModel(data_transf, response_transf, gp_lh, model)

    # draw a sample from the GP with kernel given by Jacobian of model
    zeromean_pred = gp_lh(gp_model(data_transf)).sample()