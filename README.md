# Setting

```sh
# autogluon setting
conda create -n dacon python=3.10
conda activate dacon

conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install lit cmake
pip install autogluon

conda install -y -c anaconda seaborn
conda install -y -c anaconda ipykernel

pip install pytorch_tabular
pip install optuna

# mlp setting
conda create -n hd python=3.8
conda activate hd
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tqdm pandas pyarrow scikit-learn

```

# EDA

## TODO

---

1. outlier 제거 실험
3. 