import models
from torch import Tensor
import torch
from extract_util import extract_weights, load_weights


# idea based on https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240

if __name__ == "__main__":

    bsize = 1
    N = 10000

    u_in = torch.randn(bsize, N, 1)
    y_out = torch.randn(bsize, N, 1)

    model = models.WHNet()

    out_ = model(u_in)
    params, names = extract_weights(model)

    def mod2func(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)
        out = model(u_in)
        return out

    J = torch.autograd.functional.jacobian(mod2func, params)
    J_dict = dict(zip(names, J))

