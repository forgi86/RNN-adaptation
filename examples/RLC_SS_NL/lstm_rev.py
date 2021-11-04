# Third Party
import torch
import torch.nn as nn


class LSTMFlippedStateEstimator(nn.Module):
    """ From u, y => x(N)"""
    def __init__(self, n_u=1, n_y=1, n_x=2, batch_first=False):
        super(LSTMFlippedStateEstimator, self).__init__()
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x

        self.lstm = nn.LSTM(input_size=n_y+n_u, hidden_size=16,
                            proj_size=n_x, batch_first=batch_first)

    def forward(self, u, y):
        uy = torch.cat((u, y), -1)
        uy_rev = uy.flip(0)
        x_rev, (x0, c0) = self.lstm(uy_rev)
        return x0
