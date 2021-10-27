import torch


class LSTMWrapper(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size

    def forward(self, u_in_f):
        u_in = u_in_f.view(1, -1, self.input_size)
        y_out = self.lstm(u_in)
        return y_out.view(-1, 1)


class LSTMWrapperSingleOutput(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size, output_idx):
        super(LSTMWrapperSingleOutput, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_idx = output_idx

    def forward(self, u_in_f):
        u_in = u_in_f.view(1, -1, self.input_size)
        y_out = self.lstm(u_in)
        return y_out[..., self.output_idx].view(-1, 1)

    def estimate_state(self, u_train, y_train, nstep, output_size):
        u_train = u_train.view(1, -1, self.input_size)
        y_train = y_train.view(1, -1, output_size)
        self.lstm.estimate_state(u_train, y_train, nstep)