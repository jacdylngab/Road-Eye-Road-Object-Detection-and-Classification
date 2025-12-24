import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path

plt.ion() # interactive modex

class BDD100KDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        Docstring for __init__
        
        :param self: Description
        :param images_dir: Path to the directory that contains images
        :param labels_dir: Path to the directory that contains the labels
        :param transform: Optional transform to be applied on a sample
        """
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass