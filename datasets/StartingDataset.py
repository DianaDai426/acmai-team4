import torch
import pandas as pd
from PIL import Image
import torchvision


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).'../../../humback-whale-identifiction/train.csv'
    """

    def __init__(self, train=True):
        self.data = pd.read_csv('humback-whale-identifiction/train.csv')
        self.data = self.data[self.data["Id"] != "new_whale"]
        self.data = self.data[self.data.groupby('Id').Id.transform(len) > 10]
        self.corners = pd.read_csv('humback-whale-identifiction/corners.csv')
        if train:
            self.data = self.data[0:int(len(self.data)*.8)]
            self.data = self.data[0:2000]
        else:
            self.data = self.data[int(len(self.data)*.8)+1:len(self.data)]

    def __getitem__(self, index):
        image_name = self.data["Image"][index]
        id = self.data["Id"][index]
        image = Image.open("../../humpback-whale-identification/train/"+image_name)

        image_edits = torchvision.transforms.Compose([
            # torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
            ])
        image = image_edits(image)
        corners_for_image = self.corners[self.corners["Image"] == image_name]
        x0 = int(corners_for_image['x0'])
        y0 = int(corners_for_image['y0'])
        x1 = int(corners_for_image['x1'])
        y1 = int(corners_for_image['y1'])
        image = image[y0:y1+1, x0:x1+1]
        image = torchvision.transforms.functional.resize(image, [224, 224])

        # image = torchvision.transforms.functional.rgb_to_grayscale(image)
        # image = torchvision.transforms.functional.to_tensor(image)
        
        return image

    def __len__(self):
        return len(self.data)
