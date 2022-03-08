from PreProcess import dataLoading
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, x, y):
        self.input, self.target = dataLoading(x, y)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        input, target = self.input[idx], self.target[idx]
        return (input, target)

