import os
import pandas as pd

import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image

class PokemonDataset(Dataset):

    def __init__(self, csv_file, root_dir, labels=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        # image = read_image(img_path).type(torch.float32)
        image = Image.open(img_path)

        if self.labels:
            label_idxs = [self.df.columns.get_loc(label) for label in self.labels]
            target = self.df.iloc[idx, label_idxs]
        else:
            target = self.df.iloc[idx, 1:]

        if self.transform:
            image = self.transform(image)

        return image, target


if __name__ == "__main__":
    img_dir = "data"
    annotations = "data/data.csv"

    dset = PokemonDataset(annotations, img_dir, labels=["type_1", "type_2", "is_legendary", "is_mythical"])
    
    print(len(dset))

    print(next(iter(dset)))