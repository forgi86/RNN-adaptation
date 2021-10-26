import torch
import torch.nn as nn
import torch.optim as optim

class OpenLSTM(nn.Module):
    def __init__(self, nstep):
        super(OpenLSTM, self).__init__()
        self.nstep = nstep # 64
        self.model = nn.LSTM(input_size=2, hidden_size=16, proj_size=2, num_layers=1, batch_first=True)

    def forward(self, u_train, y_train):
        y1, (hn, cn) = self.estimate_state(u_train, y_train, self.nstep)
        y2 = self.predict_state(u_train, y_train, self.nstep, (hn, cn))
        y_sim = torch.cat((y1, y2), dim=1)
        return y_sim

    def estimate_state(self, u_train, y_train, nstep):
        y_est = []
        hn = torch.zeros(1, u_train.size()[0], 2).requires_grad_()
        cn = torch.zeros(1, u_train.size()[0], 16).requires_grad_()

        for i in range(nstep):
            # Feed in the known output to estimate state
            out, (hn, cn) = self.model(y_train[:, i, :].view(64, 1, 2))
            y_est.append(out)

        y_sim = torch.cat(y_est, dim=1)
        # loss = self.loss_fn(y_sim, y_train[:, :nstep, :])
        return y_sim, (hn, cn)

    def predict_state(self, u_train, y_train, nstep, state):
        y_sim, _ = self.model(u_train[:, nstep:, :], state)
        # loss = self.loss_fn(y_sim, y_train[:, nstep:, :])
        return y_sim

    def get_model(self):
        return self.model