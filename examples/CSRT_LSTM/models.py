import torch

class LSTMWrapper(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size, batch_s):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size
        self.batch_size = batch_s

    def forward(self, u_in_f):
        y_out = self.lstm(u_in_f)
        return y_out.view(-1, 1)


class LSTMWrapperSingleOutput(torch.nn.Module):
    def __init__(self, lstm, seq_len, input_size, output_idx, batch_size):
        super(LSTMWrapperSingleOutput, self).__init__()
        self.lstm = lstm
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_idx = output_idx
        self.batch_size = batch_size

    def set_batch_size(self, batch_s):
        self.batch_size = batch_s

    def forward(self, u_in_f):
        u_in = u_in_f.view(self.batch_size, -1, self.input_size)
        y_out = self.lstm(u_in)
        return y_out[..., self.output_idx].view(-1, 1)

    def estimate_state(self, u_train, y_train, nstep, output_size):
        # Call LSTM to initialize hidden state before EGP eval()
        # Add a dimension for 3d tensor for LSTM
        u_train_ = u_train.view(-1, self.seq_len-1, self.input_size)
        y_train_ = y_train.view(-1, self.seq_len-1, output_size)
        self.lstm.estimate_state(u_train_, y_train_, nstep)