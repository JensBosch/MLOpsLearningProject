import polars as pl
from torch.utils.data import Dataset
from torch import tensor, float32


class TitanicDataset(Dataset):
    def __init__(self, filepath: str):
        df = pl.read_csv(filepath)
        self.inputs = tensor(df[:, 1:].to_numpy(), dtype=float32)
        self.targets = tensor(df[:, 0].to_numpy(), dtype=float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
