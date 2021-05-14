import torch
import pandas as pd
from PIL import Image
import torchvision


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).'../../../humback-whale-identifiction/train.csv'
    """

    def __init__(self, train=True):
        self.data = pd.read_csv('/content/train.csv')
        # self.data = self.data[self.data["Id"] != "new_whale"]
        # self.data = self.data[self.data.groupby('Id').Id.transform(len) > 10]
        self.corners = pd.read_csv('/content/acmai-team4/corners.csv')
        self.mapping = {}
        i = 0
        for label in self.data["Id"]:
            if (label not in self.mapping):
                self.mapping[label] = i
                i += 1
        # print(self.mapping)
        if train:
            # self.data = self.data[0:int(len(self.data)*.8)]
            self.data = self.data.iloc[0:2000]
        else:
            # self.data = self.data[int(len(self.data)*.8)+1:len(self.data)-1]
            self.data = self.data.iloc[2000: 3000]


    def __getitem__(self, index):
        image_name = self.data["Image"].iloc[index]
        id = self.data["Id"].iloc[index]
        image = Image.open("/content/train/"+image_name)
        # print(image.size)
        image_edits = torchvision.transforms.Compose([
            # torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
            ])
        image = image_edits(image)
        # print(image.shape)
        corners_for_image = self.corners[self.corners["Image"] == image_name]
        x0 = int(corners_for_image['x0'])
        y0 = int(corners_for_image['y0'])
        x1 = int(corners_for_image['x1'])
        y1 = int(corners_for_image['y1'])
        image = image[:,y0:y1+1, x0:x1+1]
        # print(image.shape)
        image = torchvision.transforms.functional.resize(image, [224, 224])

        # image = torchvision.transforms.functional.rgb_to_grayscale(image)
        # image = torchvision.transforms.functional.to_tensor(image)
        
        return self.mapping[id], image
    

    def __len__(self):
        return len(self.data)
