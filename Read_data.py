import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image

class Read_data(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.mem_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.mem_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()

        img_name = os.path.join(self.root_dir, self.mem_frame[idx][1],self.mem_frame[idx][2],self.mem_frame[idx][0])
        image = PIL.Image.open(img_name).convert("RGB")
        mem_score = self.mem_frame[idx][-1]
        target = float(mem_score)
        target = torch.tensor(target)
        if self.transform:
            image = self.transform(image)

        return image, target
