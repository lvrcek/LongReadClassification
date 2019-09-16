import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CoverageDataset(Dataset):

    def __init__(self, dir_chimeric, dir_non_chimeric, transform=None):
        self.path_list = []
        self.label_list = []
        self.transform = transform
        for file in os.listdir(dir_chimeric):
            self.path_list.append(os.path.join(dir_chimeric, file))
            self.label_list.append(1)
        for file in os.listdir(dir_non_chimeric):
            self.path_list.append(os.path.join(dir_non_chimeric, file))
            self.label_list.append(0)

    def __len__(self):
        return 2 * len(self.path_list)

    def __getitem__(self, idx):
        if idx < len(self.path_list):
            image = Image.open(self.path_list[idx])
            label = self.label_list[idx]
            if self.transform:
                image = self.transform(image)
            sample = {'image': image, 'label': label}
            return sample
        else:
            horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
            image = Image.open(self.path_list[idx - len(self.path_list)])
            image = horizontal_flip(image)
            label = self.label_list[idx - len(self.path_list)]
            if self.transform:
                image = self.transform(image)
            sample = {'image': image, 'label': label}
            return sample
