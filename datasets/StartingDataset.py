import torch
import pandas as pd
from PIL import Image
import torchvision
import random as rand

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).'../../../humback-whale-identifiction/train.csv'
    """

    def __init__(self, train=True):
        self.data = pd.read_csv('/content/train.csv')
        self.data = self.data[self.data["Id"] != "new_whale"]
        self.data = self.data.sample(frac=1, random_state=1)
        self.corners = pd.read_csv('/content/acmai-team4/corners.csv')
        self.mapping = {}
        i = 0
        for label in self.data["Id"]:
            if (label not in self.mapping):
                self.mapping[label] = i
                i += 1
        # print(self.mapping)
        # print(self.data['Id'].value_counts(ascending=True))
        if train:
            # self.data = self.data[0:int(len(self.data)*.8)]
            self.data = self.data.iloc[0:2000]
        else:
            # self.data = self.data[int(len(self.data)*.8)+1:len(self.data)-1]
            self.data = self.data.iloc[2000: 3000]
        
        self.images_with_same_id = {}   #takes each ID and maps to the tensor of all the images
        # {self.mapping[id] => [image names]}
        self.data = self.data[self.data.groupby('Id').Id.transform(len) > 3]
        # print(self.data['Id'].value_counts(ascending=True))

        for index, row in self.data.iterrows():
            # print(row["Id"])
            if(row["Id"] not in self.images_with_same_id):
                self.images_with_same_id[row["Id"]] = [row["Image"]]
            else:
                self.images_with_same_id[row["Id"]].append(row["Image"])

        # print(self.images_with_same_id)


    def __getitem__(self, index):
        id = self.mapping[self.data["Id"].iloc[index]]
        #Make tensor of labels
        # print(f"Id: {id}")
        # print(f"Index: {index}")
        # print(self.data["Id"].iloc[index])
        labels = [id, id, id, id]
        #Initialize tensor of images
        images = self.images_with_same_id[self.data["Id"].iloc[index]]
        images_4 = rand.sample(images,4) 

        for i in range(4):
                #change to get image from images_with_same_id
                image_name = images_4[i]
                #do the image manipulation for each image
                image = Image.open("/content/train"+image_name)
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
        # print("1")
        # print(images_4)
        # print("2")
        # print(labels)
        # print("3")
        # print(torch.tensor(labels))
        # print("4")
        # print(torch.tensor(images_4))
        return labels, images_4
    

    def __len__(self):
        return len(self.data)
