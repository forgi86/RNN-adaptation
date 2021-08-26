import torch


class LSTMWrapper(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size

    def forward(self, u_in_f):

        print(u_in_f.shape)
        u_in = u_in_f.view(1, -1, self.input_size)
        y_out, _ = self.lstm(u_in)
        return y_out.view(-1, 1)
