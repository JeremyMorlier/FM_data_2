import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms import v2

from segment_anything.utils.transforms import ResizeLongestSide

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
        image = read_image(img_path)
        masks = json.load(open(masks_path, 'r'))
        
        if self.SAM_transform :    
            image = self.SAM_transform.apply_image_torch(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image

def SAM_transform(size) :
    return  v2.Compose([
        v2.RandomCrop(size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    ])

if __name__ == "__main__" :
    transform_test = SAM_transform(1024)

    root = "/home/j20morli/Documents/Projects/02_mobile_CLIP/data/"
    
    dataset = Segment_Anything_Dataset(root + "segmentAnything00", transform_test, None, None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    images = next(iter(dataloader))
    print(images.size())

    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=root + "sam_vit_b_01ec64.pth")

    embeddings = sam.image_encoder(images)

    print(embeddings.size())
