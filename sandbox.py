import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import JSBChoralesDataset, collate_fn
from lstm import LSTMModel
from gru import GRUModel


# Params for models ===========================================
input_dim = 88
hidden_dims = {
    "lstm": 36,
    "gru": 46,
    "tanh": 100
}
layer_dim = 1
output_dim = 88

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

train_set = JSBChoralesDataset(data["train"], max_length=max_train_seq_len)
valid_set = JSBChoralesDataset(data["valid"], max_length=max_valid_seq_len)
test_set = JSBChoralesDataset(data["test"], max_length=max_test_seq_len)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



# Params for trainer ==========================================
num_epochs = 20
learning_rate = 0.001
# -------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for name, model in models.items():
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_seq_dim = train_set.max_length

    model.to(device)

    train_loss = 0.0
    neg_log_prob_train = 0.0
    total_masks = 0.0

    train_loss_list = []
    train_nll_list = []

    for epoch in range(num_epochs):
        for i, (inputs, targets, masks) in enumerate(train_loader):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            outputs = model(inputs)
            print(f"Outputs shape: {outputs.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Masks shape: {masks.shape}")

            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1, outputs.size(-1))
            masks = masks.view(-1)

            loss = criterion(outputs, targets)
            loss = (loss * masks.unsqueeze(1)).sum() / masks.sum()

            log_probs = torch.log_softmax(outputs, dim=-1)
            neg_log_prob_train += -((targets * log_probs) * masks.unsqueeze(1)).sum().item()

            total_masks += masks.sum().item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            optimizer.step()

            train_loss += loss.item() * masks.sum().item()

        train_loss /= total_masks
        avg_neg_log_prob_train = neg_log_prob_train / total_masks
        train_loss_list.append(train_loss)
        train_nll_list.append(avg_neg_log_prob_train)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train NLL: {avg_neg_log_prob_train:.4f}")