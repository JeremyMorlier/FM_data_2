import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms import v2

from segment_anything.utils.transforms import ResizeLongestSide

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data import Sampler


def arrange_folder(root, root2) :
    files_name = [name for name in os.listdir(root) if (os.path.isfile(os.path.join(root, name)))]
    files_name.sort()

    for file in files_name :
        number = int(file.split(".")[0][3:])
        extension = file.split(".")[1]
        #print(number)
        # if number >= 2088 :
        #     number += -1
        name = "sa_" + str(number-1) + "." + extension
        os.rename(root+file, root2+name)

def find_missing(root) :
    files = [int(name[3:-4]) for name in os.listdir(root) if (os.path.isfile(os.path.join(root, name)) and name[-3:] == "jpg")]
    files.sort()

    for i in files :
        print(i)
    # cst = 1
    # for i in range(1, files[-1]) :
    #     if files[i] != i + cst:
    #         print(i + cst)
    #         cst += 1

class Segment_Anything_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, SAM_transform=ResizeLongestSide):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.SAM_transform = None
        if SAM_transform :
            self.SAM_transform = SAM_transform(1024)

    def __len__(self) :
         number = len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))])
         return int(number/2)
    
    def __getitem__(self, idx):
        file_name = "sa_" + str(idx)
        img_name = file_name + ".jpg"
        masks_name = file_name + ".json"

        img_path = os.path.join(self.root, img_name)
        masks_path = os.path.join(self.root, masks_name)
        if os.path.exists(img_path) :
            image = read_image(img_path)
            masks = json.load(open(masks_path, 'r'))
            
            if self.SAM_transform :    
                image = self.SAM_transform.apply_image_torch(image)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            return image
        else :
            print("error", idx)
        
        return None

def SAM_transform(size) :
    return  v2.Compose([
        v2.RandomCrop(size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    ])

if __name__ == "__main__" :
    transform_test = SAM_transform(1024)

    root = "/home/j20morli/Documents/Projects/02_mobile_CLIP/data/"
    folder_name = "segmentAnything00/"
    folder_name1 = "segmentAnything01/"
    arrange_folder(root + folder_name1, root + folder_name)
    # find_missing(root + "segmentAnything01/")
    
    dataset = Segment_Anything_Dataset(root + folder_name, transform_test, None, None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    images = next(iter(dataloader))
    print(images.size())
    # for i, (inputs) in enumerate(dataloader):
    #     print(i, images.size())

    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=root + "sam_vit_b_01ec64.pth")

    embeddings = sam.image_encoder(images)

    print(embeddings.size())
