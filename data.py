import pickle
import torch
from torch.utils.data import DataLoader

def to_binary_vector(vector, target_dim=88, offset=21):
    binary_vector = torch.zeros(target_dim, dtype=torch.float32)

    for value in vector:
        index = int(value - offset)
        if 0 <= index < target_dim:
            binary_vector[index] = 1
        else:
            raise ValueError(f"Value {value} is out of the valid range (21 ~ 108).")
    
    return binary_vector

def to_nonbinary_vector(vector, target_dim=4, offset=21):
    indices = (vector == 1).nonzero(as_tuple=True)[0]

    sorted_indices = torch.sort(indices, descending=True)[0]

    top_indices = sorted_indices[:target_dim].float()

    if top_indices.size(0) < target_dim:
        padding = torch.zeros(target_dim - top_indices.size(0))
        top_indices = torch.cat((top_indices, padding))

    top_indices += offset

    return top_indices

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_length = max(input.size(0) for input in inputs)
    feature_dim = inputs[0].size(1)

    input_padded = torch.zeros(len(inputs), max_length, feature_dim)
    target_padded = torch.zeros(len(targets), max_length, feature_dim)
    masks = torch.zeros(len(inputs), max_length, dtype=torch.float32)

    for i, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
        input_padded[i, :input_seq.size(0), :] = input_seq
        target_padded[i, :target_seq.size(0), :] = target_seq
        masks[i, :input_seq.size(0)] = 1

    return input_padded, target_padded, masks

def filter_empty_sequence(data):
    return [seq for seq in data if len(seq) > 0]

def load_data(dataset, batch_size=32, max_length=None):
    datasets = ['jsb-chorales', 'nottingham']
    if not dataset  in datasets:
        raise ValueError(f'There is no {dataset} dataset')
    if dataset == 'jsb-chorales':
        pkl_file = "datasets/jsb-chorales/jsb-chorales-8th.pkl"

        with open(pkl_file, 'rb') as p:
            data = pickle.load(p, encoding='latin1')

    elif dataset == 'nottingham':
        pkl_file = 'datasets/nottingham/nottingham.pkl'

        with open(pkl_file, 'rb') as p:
            data = pickle.load(p)

        if max_length:
            data['train'] = [seq[:max_length] for seq in data['train']]
            data['valid'] = [seq[:max_length] for seq in data['valid']]
            data['test'] = [seq[:max_length] for seq in data['test']]

            data['train'] = filter_empty_sequence(data['train'])
            data['valid'] = filter_empty_sequence(data['valid'])
            data['test'] = filter_empty_sequence(data['test'])

    max_train_seq_len = max([len(seq) for seq in data['train']])
    max_valid_seq_len = max([len(seq) for seq in data['valid']])
    max_test_seq_len = max([len(seq) for seq in data['test']])

    print(f"Max Sequence Length(train:valid:test) {max_train_seq_len}:{max_valid_seq_len}:{max_test_seq_len}")

    train_loader = DataLoader(
        Dataset(data["train"], max_length=max_train_seq_len),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(
        Dataset(data["valid"], max_length=max_valid_seq_len),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        Dataset(data["test"], max_length=max_test_seq_len),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
    return train_loader, valid_loader, test_loader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        binary_sequence = [to_binary_vector(vector) for vector in sequence]   
        sequence_tensor = torch.stack(binary_sequence, dim=0)

        input_seq = sequence_tensor[:-1]
        target_seq = sequence_tensor[1:]
        
        return input_seq, target_seq