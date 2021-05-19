import torch
import gpytorch
import finite_ntk

data_transf = torch.randn(300, 1)
response_transf = torch.randn(300, 1)
data_new = torch.randn(300, 1)

# randomly initialize a neural network
model = torch.nn.Sequential(torch.nn.Linear(1, 30),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(30),
                            torch.nn.Linear(30, 1))

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

# draw a sample from the GP with kernel given by Jacobian of model
zeromean_pred = gp_lh(gp_model(data_transf)).sample()