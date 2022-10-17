from torch.utils import data
import torch

class HugDataset(data.Dataset):
    def __init__(self, X, y, tokenizer):
        self.X = X
        self.tokenizer = tokenizer
        self.y = y

    def __getitem__(self, idx):
        if self.tokenizer is not None:
            item = { 'input_ids': self.tokenizer.encode(self.X[idx], return_tensors='pt', padding=True, truncation=True, max_length=256) }
        else:
            item = { 'content': self.X[idx] }

        item['labels'] = torch.LongTensor( self.y[idx] )
        return item

    def __len__(self):
        return len(self.y)