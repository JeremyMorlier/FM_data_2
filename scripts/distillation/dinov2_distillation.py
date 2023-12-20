import sys

sys.path.append("/users/local/j20morli/mobile_clip/")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as transforms

import open_clip
from datasets import load_dataset

from dataset.distribution_datasets import Distribution_dataset, White_dataset

import PIL
import io

from common import Adapter, distillate

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Teacher Model
teacher_name = 'RN50-quickgelu'
teacher_dim = 1024
pretraining = "yfcc15m"
teacher, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(teacher_name, pretrained=pretraining, cache_dir="data/")
teacher.visual.to(device)

# Parameters
# Dataset parameters
cache_path = "/nasbrain/j20morli/yfcc_openai/yfcc"
batch_size_train = 10
batch_size_test = 10

size = torch.tensor([224, 768])
size2 = (3, 224, 768)
# Datasets
# Yfcc
yfcc_dataset = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", streaming=True, split='train')
print("test")
yfcc_dataloader = DataLoader(yfcc_dataset, batch_size=batch_size_train, shuffle=False)

# All pixels to random values following the original dataset mean and standard deviation
distribution = torch.distributions.normal.Normal(torch.tensor([123.675, 116.28, 103.53]), torch.tensor([58.395, 57.12, 57.375]))
random_dataset = Distribution_dataset(distribution, size, None)
random_dataloader = DataLoader(random_dataset, batch_size=batch_size_train, shuffle=False)

# All pixels to full value
white_dataset = White_dataset(size2)
white_dataloader = DataLoader(white_dataset, batch_size=batch_size_train, shuffle=False)

# Search Parameters
student_names = ["ViT-S-32-alt", "RN50", "convnext_tiny", "ViT-S-16-alt"]
student_dims = [256, 1024, 1024, 256]
iterations = 10000
checkpoints = [10, 100, 1000, 5000, 10000, 20000, 30000, 40000, iterations]
dataloaders = [yfcc_dataloader, random_dataloader, white_dataloader]
data_names = ["yfcc", "random", "white"]
#dataloaders = [random_dataloader, white_dataloader]

results = []
results_names = []
for student_name, student_dim in zip(student_names, student_dims) :
    print(student_name)
    for dataloader, data_name in zip(dataloaders, data_names) : 
        print(data_name)
        student, student_preprocess_train, student_preprocess_eval = open_clip.create_model_and_transforms(student_name, cache_dir="data/")
        student.visual.to(device)
        
        # Loss 
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(student.parameters(), lr=0.001)

        name = teacher_name + "_" + student_name + "_" + data_name

        result = distillate(name, teacher.visual, student.visual, criterion, optimizer, device, dataloader, data_name, batch_size_train, preprocess_train, student_preprocess_train, student_dim, teacher_dim, iterations, checkpoints)
        results.append(result)
        results_names.append(name)

