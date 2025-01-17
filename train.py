import pickle
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import JSBChoralesDataset, collate_fn
from torch_models import tLSTMModel, tGRUModel, tTanhModel
from lstm import LSTMModel
from gru import GRUModel
from trainer import Trainer

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
    'gru': GRUModel(input_dim, hidden_dims['gru'], layer_dim, output_dim)
}



# Params for data loader ======================================
batch_size = 32
pkl_file = "datasets/jsb-chorales/jsb-chorales-8th.pkl"
# -------------------------------------------------------------

with open(pkl_file, 'rb') as p:
    data = pickle.load(p, encoding='latin1')

max_train_seq_len = max([len(seq) for seq in data['train']])
max_valid_seq_len = max([len(seq) for seq in data['valid']])
max_test_seq_len = max([len(seq) for seq in data['test']])

train_loader = DataLoader(
    JSBChoralesDataset(data["train"], max_length=max_train_seq_len),
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(
    JSBChoralesDataset(data["valid"], max_length=max_valid_seq_len),
    batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(
    JSBChoralesDataset(data["test"], max_length=max_test_seq_len),
    batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


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

    trainer.train()
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