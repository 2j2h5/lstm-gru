import torch
import matplotlib.pyplot as plt

from torch_models import tLSTMModel, tGRUModel, tTanhModel
from lstm_model import LSTMModel
from gru_model import GRUModel
from tanh_model import TanhModel
from trainer import Trainer
from data import load_data

# Params for models ===========================================
input_dim = 88
hidden_dims = {
    "lstm": 36,
    "gru": 46,
    "tanh": 100
}
layer_dim = 1
output_dim = 88
# -------------------------------------------------------------

torch_models = {
    'lstm': tLSTMModel(input_dim, hidden_dims['lstm'], layer_dim, output_dim),
    'gru': tGRUModel(input_dim, hidden_dims['gru'], layer_dim, output_dim),
    'tanh': tTanhModel(input_dim, hidden_dims['tanh'], layer_dim, output_dim)
}

models = {
    'lstm': LSTMModel(input_dim, hidden_dims['lstm'], layer_dim, output_dim),
    'gru': GRUModel(input_dim, hidden_dims['gru'], layer_dim, output_dim),
    'tanh': TanhModel(input_dim, hidden_dims['tanh'], layer_dim, output_dim)
}



# Params for data loader ======================================
batch_size = 32
max_length = 10000

# -------------------------------------------------------------

train_loader, valid_loader, test_loader = load_data('nottingham', batch_size=batch_size, max_length=max_length)



# Params for trainer ==========================================
num_epochs = 20
learning_rate = 0.001
# -------------------------------------------------------------

loss_list = {}
nll_list = {}

for name, model in models.items():
    print(f"Training with {name} model ...")
    #print(f"Number of parameters: {model.num_params}")

    trainer = Trainer(model, train_loader, valid_loader, test_loader,
                    criterion='BCELoss', optimizer='Adam',
                    learning_rate=learning_rate, num_epochs=num_epochs, output_dim=output_dim, patience=5)

    trainer.train(early_stop=False)
    loss_list[name] = (trainer.train_loss_list, trainer.valid_loss_list)
    nll_list[name] = (trainer.train_nll_list, trainer.valid_nll_list)

    test_loss, test_nll = trainer.test()
    print(f"Final Test Loss: {test_loss:.4f}, Final Test NLL: {test_nll:.4f}")
    


# Graph =======================================================
x = torch.arange(num_epochs)
plt.figure(figsize=(10, 6))

plt.plot(x, loss_list['lstm'][0], label='LSTM Train', linestyle='-', color='blue')
plt.plot(x, loss_list['lstm'][1], label='LSTM Valid', linestyle='--', color='blue')
plt.plot(x, loss_list['gru'][0], label='GRU Train', linestyle='-', color='orange')
plt.plot(x, loss_list['gru'][1], label='GRU Valid', linestyle='--', color='orange')
plt.plot(x, loss_list['tanh'][0], label='Tanh Train', linestyle='-', color='green')
plt.plot(x, loss_list['tanh'][1], label='Tanh Valid', linestyle='--', color='green')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves for LSTM, GRU, and Tanh")
plt.legend(loc='upper right')
plt.grid(True)

plt.show()


plt.figure(figsize=(10, 6))

plt.plot(x, nll_list['lstm'][0], label='LSTM Train', linestyle='-', color='blue')
plt.plot(x, nll_list['lstm'][1], label='LSTM Valid', linestyle='--', color='blue')
plt.plot(x, nll_list['gru'][0], label='GRU Train', linestyle='-', color='orange')
plt.plot(x, nll_list['gru'][1], label='GRU Valid', linestyle='--', color='orange')
plt.plot(x, nll_list['tanh'][0], label='Tanh Train', linestyle='-', color='green')
plt.plot(x, nll_list['tanh'][1], label='Tanh Valid', linestyle='--', color='green')


plt.xlabel("Epochs")
plt.ylabel("NLL")
plt.title("NLL Curves for LSTM, GRU, and Tanh")
plt.legend(loc='upper right')
plt.grid(True)

plt.show()