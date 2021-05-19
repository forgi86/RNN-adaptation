import numpy as np

import models
from torch import Tensor
import torch
from extract_util import extract_weights, load_weights


# idea based on https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240

if __name__ == "__main__":

    bsize = 1
    N = 1000

    u_in = torch.randn(bsize, N, 1)
    y_out = torch.randn(bsize, N, 1)

    model = models.WHNet()
    out_ = model(u_in)

    # extract parameters
    params, names = extract_weights(model)
    n_param = sum(map(torch.numel, params))

    # return the neural net output as a function of the parameters, in order to be able to take the jacobian
    # using the built-in (functional) method torch.autograd.functional.jacobian(mod2func, params)
    # idea based on https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240

    def mod2func(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)
        out = model(u_in)
        return out

    jacobians = torch.autograd.functional.jacobian(mod2func, params)
    jac_dict = dict(zip(names, jacobians))

    with torch.no_grad():
        y_out_1d = torch.ravel(y_out).detach().numpy()
        param_1d = list(map(torch.ravel, params))
        P = torch.cat(param_1d, axis=0).detach().numpy()  # parameters concatenated
        jac_2d = list(map(lambda x: x.reshape(N, -1), jacobians))
        J = torch.cat(jac_2d, dim=-1).detach().numpy()


    Ip = np.eye(n_param)
    sigma = 0.1
    A = J.transpose() @ J + sigma * Ip

    P_lin = np.linalg.solve(A, J.transpose() @ y_out_1d)