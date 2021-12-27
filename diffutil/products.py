import torch


def vjp(y, x, v):
    """Computes a vector-jacobian product v^T J, aka Lop (Left Operation).
    This is what reverse-mode automatic differentiation directly obtains.

    Arguments:
    y (torch.tensor): output of differentiated function
    x (torch.tensor): differentiated input
    v (torch.tensor): vector to be multiplied with Jacobian from the left
    """
    return torch.autograd.grad(y, x, v, retain_graph=True)


def jvp(y, x, v):
    """Computes a jacobian-vector product J v, aka Rop (Right Operation)
    This is what forward-mode automatic differentiation directly obtains.
    It can also be obtained via reverse-mode differentiation using the
    well-known trick below (source?)

    Arguments:
    y (torch.tensor): output of differentiated function
    x (torch.tensor): differentiated input
    v (torch.tensor): vector to be multiplied with Jacobian from the right
    from: https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
    """
    w = torch.ones_like(y, requires_grad=True)
    return torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, v)


def jvp_diff(y, x, v):
    """Computes a jacobian-vector product J v, aka Rop (Right Operation)
    This is what forward-mode automatic differentiation directly obtains.
    It can also be obtained via reverse-mode differentiation using the
    well-known trick below (source?)

    Arguments:
    y (torch.tensor): output of differentiated function
    x (torch.tensor): differentiated input
    v (torch.tensor): vector to be multiplied with Jacobian from the right
    from: https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
    """
    w = torch.ones_like(y, requires_grad=True)
    return torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, v, create_graph=True)


def unflatten_like(vector, tensor_lst):
    """
    Takes a flat torch.tensor and unflattens it to a list of torch.tensors
        shaped like tensor_lst
    Arguments:
    vector (torch.tensor): flat one dimensional tensor
    likeTensorList (list or iterable): list of tensors with same number of ele-
        ments as vector
    """
    outList = []
    i = 0
    for tensor in tensor_lst:
        n = tensor.numel()
        outList.append(vector[i: i + n].view(tensor.shape))
        i += n
    return outList