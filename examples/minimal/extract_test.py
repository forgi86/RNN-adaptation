from torch import Tensor
import torch
from extract_util import extract_weights, load_weights


if __name__ == "__main__":

    n_hidden = 16
    bsize = 32
    # randomly initialize a neural network
    model = torch.nn.Sequential(torch.nn.Linear(1, n_hidden),
                                torch.nn.ReLU(),
                                torch.nn.Linear(n_hidden, 1))

    inputs = torch.randn(bsize, 1)
    out_ = model(inputs)


    params, names = extract_weights(model)

    def mod2func(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)
        out = model(inputs)
        return out

    #out = mod2func(*params)

    J = torch.autograd.functional.jacobian(mod2func, params)