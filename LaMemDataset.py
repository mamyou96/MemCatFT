import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image

class LaMemEvalDataset(Dataset):

    def __init__(self, csv_file, root_dir, cat, transform=None):
        mem_frame = pd.read_csv(csv_file)
        mem_frame = mem_frame.iloc[(cat-1)*2000:cat*2000]
        self.mem_frame = mem_frame
        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.mem_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.mem_frame.iloc[idx, 2],self.mem_frame.iloc[idx, 3],self.mem_frame.iloc[idx, 1])
        image = PIL.Image.open(img_name).convert("RGB")
        mem_score = self.mem_frame.iloc[idx, 13]
        target = float(mem_score)
        target = torch.tensor(target)
        if self.transform:
            image = self.transform(image)

        return image, target
