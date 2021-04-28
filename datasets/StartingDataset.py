import torch
import pandas as pd
from PIL import Image
import torchvision


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, train=True):
        self.data = pd.read_csv('humback-whale-identifiction/train.csv')
        if train:
            self.data = self.data[0:int(len(self.data)*.8)]
        else:
            self.data = self.data[int(len(self.data)*.8)+1:len(self.data)]


    def __getitem__(self, index):
        image_name = self.data["Image"][index]
        id = self.data["Id"][index]
        image = Image.open("../../humpback-whale-identification/train/"+image_name)

        resize = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
            ])
        image = resize(image)
        # image = torchvision.transforms.functional.rgb_to_grayscale(image)
        # image = torchvision.transforms.functional.to_tensor(image)
        
        return image

    def __len__(self):
        return len(self.data)
