import sys
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.utils import Config
from module.model import MLP
from module.dataset import HDData


class Predictor:
  def __init__(self, CONFIG: Config):
    self.config = CONFIG
  
  
  def setup(self):
    
    # setup_data
    train_df = pd.read_parquet(self.config.TEST_DATAPATH) # 271
    
    self.submit = pd.read_csv(self.config.SUBMIT_DATAPATH)
    
    # split
    test_dataset = HDData(train_df)
    
    self.test_dataloader = DataLoader(
      dataset=test_dataset,
      shuffle=False,
      batch_size=self.config.BATCH_SIZE,
      num_workers=self.config.NUM_WORKERS
    )

    self.model = torch.load("ckpt/MLP_v0.pth")

  def predict(self):
    self.model.eval()
    preds = []
    with torch.no_grad():
      for inputs in tqdm(self.test_dataloader):
        inputs = inputs.to(self.config.DEVICE)
        outputs = self.model(inputs).squeeze(dim=-1)
        preds += outputs.cpu().numpy().tolist()
    
    self.submit["CI_HOUR"] = [i if i > 0 else 0.0 for i in preds]
    self.submit.to_csv("csv/autogluon_v2_mlp.csv", index=False)
    
    
    

      