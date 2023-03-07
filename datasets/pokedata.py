import os
import pandas as pd

import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image

class PokemonDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, 
                 get_mask=False, get_types=False):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.get_mask = get_mask
        self.get_types = get_types
        self.mask_label = self.df.columns.get_loc("mask")
        self.type_labels = [self.df.columns.get_loc(t) 
                            for t in ["type_1", "type_2"]]

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        # image = read_image(img_path).type(torch.float32)
        image = Image.open(img_path)
        targets = []
        # TODO: think through how we can mesh disparate auxiliary data into one unpacking stream
        # EX: OHE types or egg_groups, open a mask image, etc. etc. 
        if self.get_mask:
            mask_path = os.path.join(self.root_dir, self.df.iloc[idx, self.mask_label])
            mask = Image.open(mask_path)
            targets.append(mask)
        if self.get_types:
            # need to one hot encode type labels and return the sum of the two labels we get
            pass

        if self.transform:
            image = self.transform(image)

        return image, targets


if __name__ == "__main__":
    img_dir = "data"
    annotations = "data/data.csv"

    dset = PokemonDataset(annotations, img_dir, labels=["type_1", "type_2", "is_legendary", "is_mythical"])
    
    print(len(dset))

    print(next(iter(dset)))