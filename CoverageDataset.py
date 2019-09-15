import torch
import torchvision
import model
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import shutil
from PIL import Image
import torch.optim as optim
import torch.nn as nn

# import torch.nn.functional as F

REGULAR = "./original_data/regular"
REPEAT = "./original_data/repeat"
CHIMERIC = "./original_data/chimeric"
NON_CHIMERIC = "./original_data/non-chimeric"


class CoverageDataset(Dataset):

    def __init__(self, dir_chimeric, dir_non_chimeric, transform=None):
        self.image_list = []
        self.label_list = []
        self.transform = transform
        for file in os.listdir(dir_chimeric):
            self.image_list.append(os.path.join(dir_chimeric, file))
            self.label_list.append(1)
        for file in os.listdir(dir_non_chimeric):
            self.image_list.append(os.path.join(dir_non_chimeric, file))
            self.label_list.append(0)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample


