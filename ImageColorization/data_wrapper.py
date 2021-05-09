from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
class Data(Dataset):
    def __init__(self, fileList, mode, size):
        super().__init__()
        self.fileList = fileList
        self.length = len(self.fileList)
        self.size = size
        self.__gray = Grayscale(3)
        self.__toTensor = ToTensor()
        self.__norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if mode == 'train':
            self.transform = Compose([
                Resize(size),
                ColorJitter(brightness=0.15, contrast=0.15),
                RandomHorizontalFlip(),
                RandomRotation(15)
            ])
        else:
            self.transform = Compose([
                Resize(size),
            ])
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rgb = Image.open(self.fileList[idx]).convert('RGB')
        rgb = self.transform(rgb)
        bw = self.__gray(rgb)
        return {'rgb': self.__norm(self.__toTensor(rgb)), 'bw': self.__norm(self.__toTensor(bw))}