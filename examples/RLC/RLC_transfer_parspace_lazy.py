import torch
import numpy as np
import os
from dynonet.module.lti import SisoLinearDynamicalOperator
from finite_ntk.lazy.ntk_lazytensor import NeuralTangent
import loader


class DynoWrapper(torch.nn.Module):
    def __init__(self, dyno, n_in, n_out):
        super(DynoWrapper, self).__init__()
        self.dyno = dyno
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, u_in):
        u_in = u_in[None, :, :]  # [bsize, seq_len, n_in]
        y_out = self.dyno(u_in)  # [bsize, seq_len, n_out]
        n_out = y_out.shape[-1]
        y_out_ = y_out.reshape(-1, n_out)  #if n_out > 1 else y_out.reshape(-1, )
        # output size: [bsize*seq_len, n_out] or [bsize*seq_len, ]
        return y_out_


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    model_name = 'IIR'  # model to be loaded
    n_b = 2  # numerator coefficients
    n_a = 2  # denominator coefficients
    sigma = 10.0

    # In[Load dataset]
    t, u, y, x = loader.rlc_loader("transfer", noise_std=sigma)
    n_data = t.size

    # In[Second-order dynamical system custom defined]
    G = SisoLinearDynamicalOperator(n_b, n_a)
    model_folder = os.path.join("models", model_name)
    G.load_state_dict(torch.load(os.path.join(model_folder, "model.pt")))

    # In[Model wrapping]
    n_in = 1
    n_out = 1
    G_wrapped = DynoWrapper(G, n_in, n_out)
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)
    y_torch = torch.tensor(y[None, ...], dtype=torch.float)
    u_torch_f = torch.clone(u_torch.view((1 * n_data, n_in)))  # [bsize*seq_len, n_in]
    y_torch_f = torch.clone(y_torch.view(1 * n_data, n_out))  # [bsize*seq_len, ]

    # In[Adaptation in parameter space (lazy smart way)]
    K = NeuralTangent(model=G_wrapped, data=u_torch_f)
    JtJ = K.get_expansion(epsilon=1e-4)  # lazy J^T J using the Fisher matrix trick
    JtJ_hat = JtJ.add_jitter(sigma**2)  # lazy (J^T J + \sigma^2 I)
    Jt = K.get_root()  # or Jacobian(G_wrapped, u_torch_f, y_torch_f, num_outputs=1)
    theta_lin = JtJ_hat.inv_matmul(Jt.matmul(y_torch_f))  # (J^T J + \sigma^2 I)^-1 J^T y
