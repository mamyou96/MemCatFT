import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image

class read_cat(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()

        img_name = "/home/myounes9/memcat_prjct/Memnet-Pytorch/samples/vehicles/"+ self.root_dir[idx]
        image = PIL.Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image
