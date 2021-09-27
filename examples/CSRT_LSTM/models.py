import numpy as np
import torch


class LSTMWrapper(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size, hidden_=None):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size
        if hidden_ is not None:
            self.h = hidden_[0]
            self.c = hidden_[1]
        else:
            with torch.no_grad():
                _, hid = lstm(torch.tensor(np.zeros((1, 1, self.input_size)), dtype=torch.float32))
            self.h = torch.nn.Parameter(hid[0])
            self.c = torch.nn.Parameter(hid[1])

    def forward(self, u_in_f):

        # print(u_in_f.shape)
        u_in = u_in_f.view(1, -1, self.input_size)
        y_out, _ = self.lstm(u_in, (self.h, self.c))
        return y_out.view(-1, 1)


class LSTMWrapperSingleOutput(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size, output_idx):
        super(LSTMWrapperSingleOutput, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_idx = output_idx

    def forward(self, u_in_f):

        # print(u_in_f.shape)
        u_in = u_in_f.view(1, -1, self.input_size)
        y_out, _ = self.lstm(u_in)
        return y_out[..., self.output_idx].view(-1, 1)

