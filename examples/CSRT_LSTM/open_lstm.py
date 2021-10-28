import torch
import torch.nn as nn

class OpenLSTM(nn.Module):
    def __init__(self, n_context, n_inputs, is_estimator=True):
        super(OpenLSTM, self).__init__()
        self.n_context = n_context # 64
        self.model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)
        self.n_inputs = n_inputs
        self.hn = None
        self.cn = None
        self.is_estimator = is_estimator

    def forward(self, u_train):
        if self.is_estimator:
            y1 = self.estimate_state(u_train[:, :, :self.n_inputs],
                                     u_train[:, :, self.n_inputs:], self.n_context)

            y2 = self.predict_state(u_train[:, :, :self.n_inputs], self.n_context)

            y_sim = torch.cat((y1, y2), dim=1)
        else:
            state = (self.hn, self.cn)
            print("forward openLSTM: ", u_train.size())
            y_sim, _ = self.model(u_train, state)
        return y_sim

    def estimate_state(self, u_train, y_train, nstep):
        y_est = []
        hn = torch.zeros(1, u_train.size()[0], 2).requires_grad_()
        cn = torch.zeros(1, u_train.size()[0], 16).requires_grad_()
        print("Open estimate_state: ", u_train.size(), y_train.size())

        for i in range(nstep):
            # Feed in the known output to estimate state
            # Hidden state (hn) stores the previous output
            # For state estimation, we feed in the known output value
            out, (hn, cn) = self.model(u_train[:, i, :].unsqueeze(1),
                                       (y_train[:, i, :].view(hn.shape), cn))
            y_est.append(out)

        y_sim = torch.cat(y_est, dim=1)
        self.hn, self.cn = (hn, cn)
        return y_sim

    def predict_state(self, u_train, nstep):
        state = (self.hn, self.cn)
        y_sim, _ = self.model(u_train[:, nstep:, :], state)
        return y_sim

    def get_model(self):
        return self.model

