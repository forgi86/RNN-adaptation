import torch
from torch import nn, Tensor
import functools
import copy
from typing import List, Tuple, Dict, Union, Callable


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def f_par_mod_in(*new_params, param_names, module, inputs):
    load_weights(module, param_names, new_params)
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        return module(*inputs)
    else:
        return module(inputs)


def parameter_jacobian(model, input, vectorize=True, flatten=True):

    model = copy.deepcopy(model)  # do not touch original model
    # In[Parameter Jacobians]
    # extract the parameters from the model in order to be able to take jacobians using the convenient functional API
    # see the discussion in https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240
    params, names = extract_weights(model)
    # params_dict = dict(zip(names, params))
    # n_param = sum(map(torch.numel, params))
    # scalar_names = [f"{name}_{pos}" for name in params_dict for pos in range(params_dict[name].numel())]
    # [f"{names[i]}_{j}" for i in range(len(names)) for j in range(params[i].numel())]

    # from Pytorch module to function of the module parameters only
    f_par = functools.partial(f_par_mod_in, param_names=names, module=model, inputs=input)
    f_par(*params)

    jacs = torch.autograd.functional.jacobian(f_par, params, vectorize=vectorize) # vectorize not supported in pytorch 1.7.1

    if flatten:
        if len(input.shape) == 3:
            # NOTE: Use the below code snippet for LSTM
            # TODO: Refactor
            u = input[:, :, :model.input_size]
            batch_size, seq_len, inp = u.size()
            u_new = u.reshape((seq_len * inp, batch_size))
            n_data = u_new.squeeze().shape[0]
            jacs_2d = [jac.reshape(n_data, -1) for jac in jacs]
            J = torch.cat(jacs_2d, dim=-1)
        else:
            n_data = input.squeeze().shape[0]
            jacs_2d = [jac.reshape(n_data, -1) for jac in jacs]
            J = torch.cat(jacs_2d, dim=-1)
    else:
        J = jacs

    return J

#jac_dict = dict(zip(names, jacs))

#with torch.no_grad():
#    sim_y = model(input)
#    n_data = input.squeeze().shape[0]
#    y_out_1d = torch.ravel(sim_y).detach().numpy()
#    params_1d = list(map(torch.ravel, params))
#    theta = torch.cat(params_1d, axis=0).detach().numpy()  # parameters concatenated
#    jacs_2d = list(map(lambda x: x.reshape(n_data, -1), jacs))
#    J = torch.cat(jacs_2d, dim=-1).detach().numpy()

# load_weights(model, names, tuple(params))