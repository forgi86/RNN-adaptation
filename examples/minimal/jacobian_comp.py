import  sys, os, datetime, argparse
from typing import List, Tuple
import numpy as np
import matplotlib

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch, torch.nn
from torch import nn
from torch.nn import Sequential, Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

import copy

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

# from Optimization.BayesianGradients.src.DeterministicLayers import GradBatch_Linear as Linear


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

	'''
		Make params regular Tensors instead of nn.Parameter
	'''
	params = tuple(p.detach().requires_grad_() for p in orig_params)
	return params, names

def _set_nested_attr(obj: Module, names: List[str], value: Tensor) -> None:
	"""
	Set the attribute specified by the given list of names to value.
	For example, to set the attribute obj.conv.weight,
	use _del_nested_attr(obj, ['conv', 'weight'], value)
	"""
	if len(names) == 1:
		setattr(obj, names[0], value)
	else:
		_set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod: Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
	"""
	Reload a set of weights so that `mod` can be used again to perform a forward pass.
	Note that the `params` are regular Tensors (that can have history) and so are left
	as Tensors. This means that mod.parameters() will still be empty after this call.
	"""
	for name, p in zip(names, params):
		_set_nested_attr(mod, name.split("."), p)

def compute_jacobian(model, x):
	'''

	@param model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
	@param x: input since any gradients requires some input
	@return: either store jac directly in parameters or store them differently

	we'll be working on a copy of the model because we don't want to interfere with the optimizers and other functionality
	'''

	jac_model = copy.deepcopy(model) # because we're messing around with parameters (deleting, reinstating etc)
	all_params, all_names = extract_weights(jac_model) # "deparameterize weights"
	load_weights(jac_model, all_names, all_params) # reinstate all weights as plain tensors

	def param_as_input_func(model, x, param):
		load_weights(model, [name], [param]) # name is from the outer scope
		out = model(x)
		return out

	for i, (name, param) in enumerate(zip(all_names, all_params)):
		jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, x, param), param,
							 strict=True if i==0 else False, vectorize=False if i==0 else True)
		print(jac.shape)

	del jac_model # cleaning up

def compute_diag_Hessian(model, loss):
	'''
	Computing the diagonal Hessian layer wise and batches the computations over the layers
	@param model: model as a container for all the parameters
	@param loss: need to differentiate it
	@return:
	'''

	if not hasattr(model.parameters().__iter__(), 'grad') or model.parameters().__iter__().grad is None:
		loss.backward(create_graph=True, retain_graph=True)

	grad = [param.grad for param in model.parameters()]

	# iterate over precomputed gradient and the parameter in lockstep
	for grad, param in zip(grad, model.parameters()):
		gradgrad = torch.autograd.grad(outputs=grad, inputs=param, retain_graph=True, grad_outputs=torch.ones_like(grad), allow_unused=True)
		param.gradgrad = gradgrad  # store in conveniently in parameter


class Net(torch.nn.Module):

	def __init__(self):
		super().__init__()

		self.nn = Sequential(Linear(in_features=4, out_features=7, bias=True),
					 torch.nn.Tanh(),
					 Linear(in_features=7, out_features=3, bias=True),)

	def forward(self, x):
		return self.nn(x)

net = Net()
x = torch.randn(5,4).requires_grad_()
compute_jacobian(net, x)

out = net(x).sum()
compute_diag_Hessian(net, out)
