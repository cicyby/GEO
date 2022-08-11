import torch
import torch.nn as nn
from .position_embedding import SinusoidalPositionalEmbedding


# embeding output (b, len, dim)
# input size (embedding size=len(signal),batch)
class LSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers, out_dim):
        super(LSTM, self).__init__()
        self.embad_dim = embed_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.positionEmbedding = SinusoidalPositionalEmbedding(embed_dim)
        self.lstm1 = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.lstm3 = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.fc_nn = nn.Linear(embed_dim*3, out_dim)

    def forward(self, rdispph, prf, rwe):
        rdispph_embed = self.positionEmbedding(rdispph)
        prf_embed = self.positionEmbedding(prf)
        rwe_embed = self.positionEmbedding(rwe)

        _, (rdispph_lstm, _) = self.lstm1(rdispph.unsqueeze(2))
        _, (prf_lstm, _) = self.lstm2(prf.unsqueeze(2))
        _, (rwe_lstm, _) = self.lstm3(rwe.unsqueeze(2))

        x = torch.cat([rdispph_lstm, prf_lstm, rwe_lstm], 2)
        out = self.fc_nn(x)

        return out

if __name__ == '__main__':
    lstm = LSTM(embed_dim=300, in_dim=[50, 201, 38], hidden_dim=300, n_layers=1)
    x = torch.tensor(torch.rand(3, 4, 201))
    out, (_, _) = lstm(x)
    print(out.shape)


