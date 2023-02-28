import os
import pandas as pd

import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image

class MaskedPokemonDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path).type(torch.float32)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    img_dir = "data"
    annotations = "data/masks.csv"

    dset = MaskedPokemonDataset(annotations, img_dir)

    print(len(dset))

