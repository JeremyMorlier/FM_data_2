import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import open_clip
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def distillate(teacher, student, criterion, optimizer, dataloader, batch_size, teacher_preprocess, student_preprocess, iterations, checkpoints, name) :

    running_loss = 0.0
    for iteration in iterations :
        optimizer.zero_grad()
        
        inputs = next(iter(dataloader)).to(device)

        student_inputs = student_preprocess(inputs)
        teacher_inputs = teacher_preprocess(inputs)

        student_outputs = student(student_inputs)
        with torch.no_grad() :
            teacher_outputs = teacher(teacher_inputs)

        loss = criterion(student_outputs, teacher_outputs)
        optimizer.step()

        running_loss += loss
        sys.stdout.write(f'\r {name} : {iteration + 1}/{iterations} - loss {round(loss.item() / (batch_size), 3)} ' f' - running loss {round(running_loss.item() / (batch_size), 3)}                       ')

        if iteration in checkpoints :
            torch.save(student, name + "_" + str(iterations))
            running_loss = 0.0

# Parameters
# Dataset parameters
cache_path = "/nasbrain/datasets/yfcc"
batch_size_train = 10
batch_size_test = 10


# Datasets
# Yfcc
yfcc_dataset = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", cache_dir=cache_path)
# transforms ?
yfcc_dataloader = DataLoader(yfcc_dataset, batch_size=batch_size_train, shuffle=False)

# All pixels to random values following the original dataset mean and standard deviation
random_dataset = random_dataset(size, mean, std)
random_dataloader = DataLoader(random_dataset, batch_size=batch_size_train, shuffle=False)

# All pixels to full value
white_dataset = white_dataset(size)
white_dataloader = DataLoader(white_dataset, batch_size=batch_size_train, shuffle=False)

# Teacher Model
teacher_name = 'RN50-quickgelu'
pretraining = "yfcc15m"
teacher, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(teacher_name, pretrained=pretraining, cache_dir="data/")
teacher.visual.to(device)

# Search Parameters
student_names = ["ViT-S-32-alt", "RN-50", "convnext_tiny", "ViT-S-16-alt"]
iterations = 10000
checkpoints = [10, 100, 1000, 5000, 10000, 20000, 30000, 40000, iterations]
dataloaders = [yfcc_dataloader, random_dataloader, white_dataloader]



for student_name in student_names :
    for dataloader in dataloaders : 
        student, student_preprocess_train, student_preprocess_eval = open_clip.create_model_and_transforms(student_name, cache_dir="data/")
        student.visual.to(device)

        #TODO: add preprocess to distillate
        distillate(teacher.visual, student.visual, dataloader, batch_size_train, preprocess_train, student_preprocess_train, iterations, checkpoints)

