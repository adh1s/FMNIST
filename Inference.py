import torch
import torch.nn as nn
from Model import CNNclassifier

model_dir = ''

model = CNNclassifier(img_size=28, n_channels=1, n_classes=10)
model.load_state_dict(torch.load(model_dir))
model.eval()
