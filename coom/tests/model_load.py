import torch

from coom.lightning_model import EKALightningModel  # your model class

ckpt_path = "/projects/data/teams/coom_team/training/checkpoint_path/eka_train_prototype_13_11_25/epoch=0-step=100-EMA/rank_0/model.pt"

ckpt = torch.load(ckpt_path, map_location="cpu")

print(ckpt.keys())
