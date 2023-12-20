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
# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_yfcc = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

class Adapter(nn.Module) :
    def __init__(self, in_features, out_features) :
        super(Adapter, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=None)
    
    def forward(self, input) :
        return self.linear(input)
    
def distillate(name, teacher, student, criterion, optimizer, device, dataloader, batch_size, teacher_preprocess, student_preprocess, student_dim, teacher_dim, iterations, checkpoints=[]) :
    adaptor = None
    if student_dim != teacher_dim :
        adaptor = Adapter(student_dim, teacher_dim).to(device)

    running_loss = 0.0
    running_iterations = 0

    results = []

    for iteration in range(0, iterations) :
        optimizer.zero_grad()
        
        if data_name == "yfcc" :
            inputs = next(dataloader.__iter__())
            student_temp = []
            teacher_temp = []
            for image in inputs["img"] :
                temp = transform_yfcc(PIL.Image.open(io.BytesIO(image)))
                student_temp.append(student_preprocess(temp))
                teacher_temp.append(teacher_preprocess(temp))

            student_inputs = torch.stack((student_temp)).to(device)
            teacher_inputs = torch.stack((teacher_temp)).to(device)
        else :
            inputs = next(dataloader.__iter__()).to(device)
            student_inputs = student_preprocess(inputs)
            teacher_inputs = teacher_preprocess(inputs)

        student_outputs = student(student_inputs)
        if adaptor != None :
            student_outputs = adaptor(student_outputs)

        with torch.no_grad() :
            teacher_outputs = teacher(teacher_inputs)

        loss = criterion(student_outputs, teacher_outputs)
        running_loss += loss
        running_iterations += 1

        loss.backward()
        optimizer.step()
        
        sys.stdout.write(f'\r {name} : {iteration + 1}/{iterations} - loss {round(loss.item() / (batch_size), 3)} ' f' - running loss {round(running_loss.item() / ((running_iterations + 1) * batch_size), 3)}')

        if (iteration + 1) in checkpoints :
            print(" \r\nSave at " + str(iteration + 1) + " Iterations")
            torch.save(student, name + "_" + str(iteration-1) + ".pth")
            results.append(running_loss)
            running_loss = 0.0
            running_iterations = 0

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

        result = distillate(name, teacher.visual, student.visual, criterion, optimizer, device, dataloader, batch_size_train, preprocess_train, student_preprocess_train, student_dim, teacher_dim, iterations, checkpoints)
        results.append(result)
        results_names.append(name)

