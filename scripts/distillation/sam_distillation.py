import sys
import PIL
import io

sys.path.append("/users/local/j20morli/mobile_clip/")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as transforms

from datasets import load_dataset

from dataset.distribution_datasets import Distribution_dataset, White_dataset
from segment_anything.utils.transforms import ResizeLongestSide
from dataset.SAM_dataset import Segment_Anything_Dataset

# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import segment_anything as sam
import mobile_sam as msam

from scripts.common import distillate

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "/nasbrain/datasets/sam_dataset/"
folder_name = "images/"

model_path = "/users/local/j20morli/data/checkpoints/"

# Teacher Model
teacher_name = "vit_b"
sam_teacher = sam.sam_model_registry[teacher_name](checkpoint=model_path + "/sam_vit_b_01ec64.pth")
sam_teacher.image_encoder.to(device)
teacher_dim = 256

# Parameters
# Dataset parameters
batch_size_train = 1

# Datasets

# Sam Dataset
sam_dataset = Segment_Anything_Dataset(data_path + folder_name, None, None, ResizeLongestSide, sam_teacher.image_encoder.img_size)
sam_dataloader = DataLoader(sam_dataset, batch_size=1, shuffle=False)

#   Subsets
print(len(sam_dataset))
indices1 = torch.randperm(len(sam_dataset))[:1000]
indices2 = torch.randperm(len(sam_dataset))[:5000]
indices3 = torch.randperm(len(sam_dataset))[:10000]

subset1 = Subset(sam_dataset, indices1)
subset2 = Subset(sam_dataset, indices2)
subset3 = Subset(sam_dataset, indices3)

sam_dataloader1 = DataLoader(subset1, batch_size=1, shuffle=False)
sam_dataloader2 = DataLoader(subset2, batch_size=1, shuffle=False)
sam_dataloader3 = DataLoader(subset3, batch_size=1, shuffle=False)
# All pixels to random values following the original dataset mean and standard deviation
size = torch.tensor([1024, 1024])
distribution = torch.distributions.normal.Normal(torch.tensor([123.675, 116.28, 103.53]), torch.tensor([58.395, 57.12, 57.375]))
random_dataset = Distribution_dataset(distribution, size, None)
random_dataloader = DataLoader(random_dataset, batch_size=batch_size_train, shuffle=False)

# Search Parameters
student_names = ["vit_t"]
student_dims = [256]
iterations = 55000
checkpoints = [10, 100, 1000, 5000, 10000, 20000, 30000, 40000, iterations]
# dataloaders = [sam_dataloader1, sam_dataloader2, sam_dataloader3, random_dataloader]
# data_names = ["sam1000", "sam5000", "sam10000", "random"]
dataloaders = [sam_dataloader1, sam_dataloader3]
data_names = ["sam1000", "sam10000"]
#dataloaders = [random_dataloader, white_dataloader]

results = []
results_names = []
for student_name, student_dim in zip(student_names, student_dims) :
    print(student_name)
    for dataloader, data_name in zip(dataloaders, data_names) : 
        print(data_name)
        student = msam.sam_model_registry[student_name]()
        student.image_encoder.to(device)
        
        # Loss 
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(student.parameters(), lr=0.001)

        name = teacher_name + "_" + student_name + "_" + data_name

        result = distillate(name, sam_teacher.image_encoder, student.image_encoder, criterion, optimizer, device, dataloader, data_name, batch_size_train, None, None, student_dim, teacher_dim, iterations, checkpoints)
        results.append(result)
        results_names.append(name)

