import torch
from torchid.module.lti import SisoLinearDynamicalOperator
from torchid.module.static import SisoStaticNonLinearity


class WHNet(torch.nn.Module):
    def __init__(self, nb_1=8, na_1=8, nb_2=8, na_2=8):
        super(WHNet, self).__init__()
        self.nb_1 = nb_1
        self.na_1 = na_1
        self.nb_2 = nb_2
        self.na_2 = na_2
        self.G1 = SisoLinearDynamicalOperator(n_b=self.nb_1, n_a=self.na_1, n_k=1)
        self.F_nl = SisoStaticNonLinearity(n_hidden=10, activation='tanh')
        self.G2 = SisoLinearDynamicalOperator(n_b=self.nb_2, n_a=self.na_2, n_k=0)

    def forward(self, u):
        y1_lin = self.G1(u)
        y1_nl = self.F_nl(y1_lin)  # B, T, C1
        y2_lin = self.G2(y1_nl)  # B, T, C2

        return y2_lin
