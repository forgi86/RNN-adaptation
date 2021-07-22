import torch
import functools
from dynonet.utils.extract_util import extract_weights, load_weights, f_par_mod_in
import copy


def parameter_jacobian(model, input, vectorize=True):

    model = copy.deepcopy(model)  # do not touch original model
    # In[Parameter Jacobians]
    # extract the parameters from the model in order to be able to take jacobians using the convenient functional API
    # see the discussion in https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240
    params, names = extract_weights(model)
    params_dict = dict(zip(names, params))
    n_param = sum(map(torch.numel, params))
    scalar_names = [f"{name}_{pos}" for name in params_dict for pos in range(params_dict[name].numel())]
    # [f"{names[i]}_{j}" for i in range(len(names)) for j in range(params[i].numel())]

    # from Pytorch module to function of the module parameters only
    f_par = functools.partial(f_par_mod_in, param_names=names, module=model, inputs=input)
    f_par(*params)

    jacs = torch.autograd.functional.jacobian(f_par, params, vectorize=vectorize)
    jac_dict = dict(zip(names, jacs))
    with torch.no_grad():
        sim_y = model(input)
        n_data = input.squeeze().shape[0]
        y_out_1d = torch.ravel(sim_y).detach().numpy()
        params_1d = list(map(torch.ravel, params))
        theta = torch.cat(params_1d, axis=0).detach().numpy()  # parameters concatenated
        jacs_2d = list(map(lambda x: x.reshape(n_data, -1), jacs))
        J = torch.cat(jacs_2d, dim=-1).detach().numpy()

    # load_weights(model, names, tuple(params))
    return J
