import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.dropout_rate = dropout_rate
    
    self.block1 = nn.Sequential(
      nn.Linear(self.input_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=self.dropout_rate),
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
    )
    
    self.block2 = nn.Sequential(
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=self.dropout_rate),
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
    )

    self.regression_head = nn.Linear(self.hidden_dim, self.output_dim)
    
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.regression_head(x)
    return x

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.dropout_rate = dropout_rate
    
    self.block1 = nn.Sequential(
      nn.Linear(self.input_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=self.dropout_rate),
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
    )
    
    
    self.block2 = nn.Sequential(
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=self.dropout_rate),
      nn.Linear(self.hidden_dim, self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(inplace=True),
    )

    self.regression_head = nn.Linear(self.hidden_dim, self.output_dim)
    
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.regression_head(x)
    return x
    