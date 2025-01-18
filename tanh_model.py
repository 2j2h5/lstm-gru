import torch
import torch.nn as nn

class TanhCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(TanhCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx = hidden

        hy = torch.tanh(self.x2h(x) + self.h2h(hx))
        return hy
    
class TanhModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(TanhModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.tanh_cell = TanhCell(input_dim, hidden_dim, bias=bias)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=device)

        hn = h0[0, :, :]

        outputs = []
        for seq in range(x.size(1)):
            hn = self.tanh_cell(x[:, seq, :], hn)
            outputs.append(hn)

        outputs = torch.stack(outputs, dim=1)
        out = self.fc(outputs)

        return out
