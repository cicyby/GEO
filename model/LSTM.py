import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers, out_dim):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # input [seq_len, batch_size, dim=1]
        self.lstm1 = nn.LSTM(1, hidden_dim, n_layers)
        self.lstm2 = nn.LSTM(1, hidden_dim, n_layers)
        self.lstm3 = nn.LSTM(1, hidden_dim, n_layers)
        self.fc_nn = nn.Linear(hidden_dim*3, out_dim)

    def forward(self, rdispph, prf, rwe):
        # data [batch_size, seq_len] -> [seq_len, batch_size, 1]
        _, (rdispph_lstm, _) = self.lstm1(rdispph.unsqueeze(2).permute([1, 0, 2]))
        _, (prf_lstm, _) = self.lstm2(prf.unsqueeze(2).permute([1, 0, 2]))
        _, (rwe_lstm, _) = self.lstm3(rwe.unsqueeze(2).permute([1, 0, 2]))

        x = torch.cat([rdispph_lstm, prf_lstm, rwe_lstm], 2)
        out = self.fc_nn(x)

        return out

if __name__ == '__main__':
    lstm = LSTM(hidden_dim=128, n_layers=1, out_dim=51)
    x = torch.tensor(torch.rand(201, 2, 1))
    out, (_, _) = lstm(x)
    print(out.shape)


