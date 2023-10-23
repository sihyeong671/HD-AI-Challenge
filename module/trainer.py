import os
import sys
from copy import deepcopy


from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from module.utils import Config
from module.dataset import HDData
from module.model import MLP

class Trainer:
  def __init__(self, CONFIG: Config):
    self.config = CONFIG
  
  
  def setup(self):
    
    ## setup_data
    train_df = pd.read_parquet(self.config.TRAIN_DATAPATH) # 270
    
    
    # split
    X = train_df.drop(columns=["CI_HOUR"])
    y = train_df["CI_HOUR"]
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=self.config.SEED)

    train_dataset = HDData(train_x, train_y)
    val_dataset = HDData(val_x, val_y)
    
    self.train_dataloader = DataLoader(
      dataset=train_dataset,
      shuffle=True, 
      batch_size=self.config.BATCH_SIZE,
      num_workers=self.config.NUM_WORKERS
    )
    
    self.val_dataloader = DataLoader(
      dataset=val_dataset,
      shuffle=False,
      batch_size=self.config.BATCH_SIZE,
      num_workers=self.config.NUM_WORKERS
    )
    
    ## setup model & loss_fn & optimizer & lr_scheduler
    self.model = MLP(
      input_dim=270,
      hidden_dim=2048,
      bottle_neck_dim=64,
      output_dim=1
    )
    
    self.loss_fn = nn.MSELoss()
    
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer=self.optimizer,
      mode="min",
      factor=0.5,
      patience=10,
      min_lr=1e-5
    )
    
  def train(self):
    self.model.to(self.config.DEVICE)
    
    best_model = None
    best_val_loss = sys.maxsize
    
    for epoch in range(1, self.config.EPOCHS+1):
      train_loss = 0
      self.model.train()
      for inputs, labels in tqdm(self.train_dataloader):
        inputs = inputs.to(self.config.DEVICE)
        labels = labels.to(self.config.DEVICE)

        self.optimizer.zero_grad()
        
        outputs = self.model(inputs).squeeze(dim=-1)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        
        self.optimizer.step()
        
        train_loss += loss.item()
      
      train_loss /= len(self.train_dataloader)

      val_loss = self._valid()
      
      if self.scheduler is not None:
        self.scheduler.step(val_loss)

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(self.model)
    
    os.makedirs("ckpt/", exist_ok=True)
    torch.save(best_model, f"ckpt/{self.config.MODEL_NAME}_{self.config.DETAIL}.pth")
      
  def _valid(self):
    self.model.eval()
    with torch.no_grad():
      val_loss = 0
      for inputs, labels in tqdm(self.val_dataloader):
        inputs = inputs.to(self.config.DEVICE)
        labels = labels.to(self.config.DEVICE)
  
        outputs = self.model(inputs).squeeze(dim=-1)
        loss = self.loss_fn(outputs, labels)
        
        val_loss += loss.item()
      
      return val_loss / len(self.val_dataloader)
      