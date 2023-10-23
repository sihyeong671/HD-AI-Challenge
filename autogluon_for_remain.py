import pandas as pd
import autogluon
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

train_df = pd.read_parquet("../data/HD_data/train_v2_remain.parquet")
test_df = pd.read_parquet("../data/HD_data/test_v2_remain.parquet")

train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)

label = "CI_HOUR"
eval_metric = "mean_absolute_error"

predictor = TabularPredictor(
    label=label, problem_type='regression', eval_metric=eval_metric,
).fit(train_data, presets="best_quality", num_gpus=4, num_bag_folds=7, num_stack_levels=1)

