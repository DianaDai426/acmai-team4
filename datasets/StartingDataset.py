import torch
import pandas as pd
from PIL import Image
import torchvision
from random import random

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).'../../../humback-whale-identifiction/train.csv'
    """

    def __init__(self, train=True):
        self.data = pd.read_csv('/Users/cameronfiske/Desktop/AI_Project/humpback-whale-identification/train.csv')
        self.data = self.data[self.data["Id"] != "new_whale"]
        self.data = self.data[self.data.groupby('Id').Id.transform(len) > 3]
        self.corners = pd.read_csv('/Users/cameronfiske/Desktop/AI_Project/corners.csv')
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
        self.images_with_same_id = {}   #takes each ID and maps to the tensor of all the images
        # {self.mapping[id] => [image names]}
            
        for row in self.data:
            if(row["Id"] not in self.images_with_same_id):
                self.images_with_same_id[row["Id"]] = [row["Image"]]
            else:
                self.images_with_same_id[row["Id"]].append(row["Image"])


    def __getitem__(self, index):
        id = self.mapping[self.data["Id"].iloc[index]]
        #Make tensor of labels

        labels = [id, id, id, id]
        #Initialize tensor of images
        images = self.images_with_same_id[id]
        images_4 = random.sample(images,4) 

        for i in range(4):
                #change to get image from images_with_same_id
                image_name = images_4[i]
                #do the image manipulation for each image
                image = Image.open("/Users/cameronfiske/Desktop/AI_Project/humpback-whale-identification/train/"+image_name)
                image = image.convert('RGB')
                image_edits = torchvision.transforms.Compose([
                    # torchvision.transforms.Resize([224, 224]),
                    # torchvision.transforms.Grayscale(),
                    torchvision.transforms.ToTensor()
                    ])
                image = image_edits(image)
                corners_for_image = self.corners[self.corners["Image"] == image_name]
                
                x0 = int(corners_for_image['x0'])
                y0 = int(corners_for_image['y0'])
                x1 = int(corners_for_image['x1'])
                y1 = int(corners_for_image['y1'])
                image = image[:,y0:y1+1, x0:x1+1]
                image = torchvision.transforms.functional.resize(image, [224, 224])
                images_4[i] = image

        # image = torchvision.transforms.functional.rgb_to_grayscale(image)
        # image = torchvision.transforms.functional.to_tensor(image)
        
        return torch.tensor(labels), torch.tensor(images_4) #return tensor of 4 of the same labels and 4 random images
    

    def __len__(self):
        return len(self.data)
