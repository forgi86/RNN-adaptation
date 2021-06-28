import torch
from dynonet.module.lti import SisoLinearDynamicalOperator
from dynonet.module.static import SisoStaticNonLinearity
import torch.nn as nn


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
        y_out_ = y_out.reshape(-1, n_out)  if n_out > 1 else y_out.reshape(-1, )
        # [bsize*seq_len, n_out] or [bsize*seq_len, ]
        return y_out_


class LSTMWrapper(torch.nn.Module):
    def __init__(self, lstm, n_in, n_out, seq_len, n_hidden):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm
        self.n_in = n_in
        self.n_out = n_out
        self.seq_len = seq_len
        self.n_hidden = n_hidden

    def forward(self, u_in):
        u_in = u_in.view(*u_in.shape[:-1], -1, self.n_in)
        y_out = self.lstm(u_in)
        y_out = y_out.reshape(-1, y_out.shape[-1]) #if y_out.shape[-1] > 1 else y_out.reshape(-1, )
        return y_out


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True).double()
        self.decoder = nn.Linear(hidden_size * (1 + int(bidirectional)), output_size).double()

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        return self.decoder(output)
