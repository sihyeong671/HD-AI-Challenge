import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, bottle_neck_dim, output_dim):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.bottle_neck_dim = bottle_neck_dim
    self.output_dim = output_dim
    
    self.block1 = nn.Sequential(
      nn.Linear(self.input_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.3),
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
    )
    
    self.bottle_neck = nn.Linear(self.hidden_dim, self.bottle_neck_dim)
    
    self.block2 = nn.Sequential(
      nn.Linear(self.input_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.3),
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Linear(self.hidden_dim, self.output_dim)
    )
    
  def forward(self, x):
    x = self.block1(x)
    x = self.bottle_neck(x)
    x = self.block2(x)
    return x
    