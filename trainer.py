import torch
import torch.nn as nn
import torch.optim as optim
import math

class Trainer():
    def __init__(self, model, train_loader, valid_loader, test_loader,
                 criterion='CELoss', optimizer='SGD',
                 learning_rate=0.001, num_epochs=20, output_dim=88, patience=5):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        criterion_dict = {
            'BCELoss': nn.BCEWithLogitsLoss(reduction='none'),
            'CELoss': nn.CrossEntropyLoss(reduction='none')
            }
        self.criterion = criterion_dict[criterion]

        if optimizer == 'RMSProp':
            log_lr = torch.FloatTensor(1).uniform_(-12, -6).item()
            learning_rate = math.exp(log_lr)
            print(f"Selected Learning Rate: {learning_rate:.8f}")

        optimizer_dict = {
            'SGD': optim.SGD(self.model.parameters(), lr=learning_rate),
            'Adam': optim.Adam(self.model.parameters(), lr=learning_rate),
            'RMSProp': optim.RMSprop(self.model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-08)
            }
        self.optimizer = optimizer_dict[optimizer]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.output_dim = output_dim
        self.patience = patience

        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_nll_list = []
        self.valid_nll_list = []

    def train_step(self):
        self.model.train()

        train_loss = 0.0
        neg_log_prob_train = 0.0
        total_masks = 0.0

        for inputs, targets, masks in self.train_loader:
            inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)

            outputs = self.model(inputs)

            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1, outputs.size(-1))
            masks = masks.view(-1)

            loss = self.criterion(outputs, targets)
            loss = (loss * masks.unsqueeze(1)).sum() / masks.sum()

            log_probs = torch.log_softmax(outputs, dim=-1)
            neg_log_prob_train += -((targets * log_probs) * masks.unsqueeze(1)).sum().item()

            total_masks += masks.sum().item()

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            
            self.optimizer.step()

            train_loss += loss.item() * masks.sum().item()

        train_loss /= total_masks
        avg_neg_log_prob_train = neg_log_prob_train / total_masks
        self.train_loss_list.append(train_loss)
        self.train_nll_list.append(avg_neg_log_prob_train)

        return train_loss, avg_neg_log_prob_train

    def valid_step(self):
        self.model.eval()

        valid_loss = 0.0
        neg_log_prob_valid = 0.0
        total_masks = 0.0

        with torch.no_grad():
            for inputs, targets, masks in self.valid_loader:
                inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)

                outputs = self.model(inputs)

                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1, outputs.size(-1))
                masks = masks.view(-1)

                loss = self.criterion(outputs, targets)
                loss = (loss * masks.unsqueeze(1)).sum() / masks.sum()

                log_probs = torch.log_softmax(outputs, dim=-1)
                neg_log_prob_valid += -((targets * log_probs) * masks.unsqueeze(1)).sum().item()

                total_masks += masks.sum().item()

                valid_loss += loss.item() * masks.sum().item()

        valid_loss /= total_masks
        avg_neg_log_prob_valid = neg_log_prob_valid / total_masks
        self.valid_loss_list.append(valid_loss)
        self.valid_nll_list.append(avg_neg_log_prob_valid)

        return valid_loss, avg_neg_log_prob_valid

    def train(self, early_stop=False):
        self.model.to(self.device)

        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.075)

        best_valid_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            train_loss, avg_neg_log_prob_train = self.train_step()
            valid_loss, avg_neg_log_prob_valid = self.valid_step()

            print(f"Epoch [{epoch+1}/{self.num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
              f"Train NLL: {avg_neg_log_prob_train:.4f}, Valid NLL: {avg_neg_log_prob_valid:.4f}")
            
            if early_stop:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print("Early stopping triggered")
                        break

    def test(self):
        self.model.eval()

        test_loss = 0.0
        neg_log_prob_test = 0.0
        total_masks = 0.0

        with torch.no_grad():
            for inputs, targets, masks in self.test_loader:
                inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)

                outputs = self.model(inputs)

                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1, outputs.size(-1))
                masks = masks.view(-1)

                loss = self.criterion(outputs, targets)
                loss = (loss * masks.unsqueeze(1)).sum() / masks.sum()

                log_probs = torch.log_softmax(outputs, dim=-1)
                neg_log_prob_test += -((targets * log_probs) * masks.unsqueeze(1)).sum().item()

                total_masks += masks.sum().item()

                test_loss += loss.item() * masks.sum().item()

        test_loss /= total_masks
        avg_neg_log_prob_test = neg_log_prob_test / total_masks

        print(f"Test Loss: {test_loss:.4f}, Test NLL: {avg_neg_log_prob_test:.4f}")
        return test_loss, avg_neg_log_prob_test