import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()

        self.num_params = 3 * ((input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim)# + (hidden_dim * output_dim + output_dim)

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, bias=bias, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.fc(output)

        return output
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()

        self.num_params = 4 * ((input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim)# + (hidden_dim * output_dim + output_dim)

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, bias=bias, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)

        return output
    
class TanhModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(TanhModel, self).__init__()

        self.num_params = ((input_dim * hidden_dim) + (hidden_dim * hidden_dim) + hidden_dim)# + (hidden_dim * output_dim + output_dim)

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, bias=bias, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)

        return output