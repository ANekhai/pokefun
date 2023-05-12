import os
import pandas as pd

import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image

class PokemonDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, 
                 get_mask=False, get_types=False):
        self.df = pd.read_csv(csv_file).fillna("None")
        self.root_dir = root_dir
        self.transform = transform
        self.get_mask = get_mask
        self.get_types = get_types
        self.mask_label = self.df.columns.get_loc("mask")
        self.type_labels = [self.df.columns.get_loc(t) 
                            for t in ["type_1", "type_2"]]
        
        # type logic
        self.types = sorted(self.df["type_2"].unique())

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        # image = read_image(img_path).type(torch.float32)
        image = Image.open(img_path)
        targets = []

        if self.get_mask:
            mask_path = os.path.join(self.root_dir, self.df.iloc[idx, self.mask_label])
            mask = Image.open(mask_path)
            targets.append(mask)
        if self.get_types:
            types = [self.df.iloc[idx, t] for t in self.type_labels]
            type_vec = torch.zeros(len(self.types))
            
            for type in types:
                type_vec[self.types.index(type)] = 1.

            targets.append(type_vec)

        if self.transform:
            image = self.transform(image)

        return image, targets


if __name__ == "__main__":
    img_dir = "data"
    annotations = "data/data.csv"

    dset = PokemonDataset(annotations, img_dir, get_types=True)
    
    print(len(dset))

    print(next(iter(dset)))