import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df.to_parquet("data/train.parquet", index=False)
test_df.to_parquet("data/test.parquet", index=False)


