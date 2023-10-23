import torch
from torch.utils.data import Dataset, DataLoader

class HDData(Dataset):
  def __init__(self, X, y=None):
    self.X = X
    self.y = y
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, index: int):
    if self.y is None:
      return torch.tensor(self.X.iloc[index]).float()
    return torch.tensor(self.X.iloc[index]).float(), torch.tensor(self.y.iloc[index]).float()