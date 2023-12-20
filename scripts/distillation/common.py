
import sys

sys.path.append("/users/local/j20morli/mobile_clip/")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as transforms

import PIL
import io


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


def distillate(name, teacher, student, criterion, optimizer, device, dataloader, data_name, batch_size, teacher_preprocess, student_preprocess, student_dim, teacher_dim, iterations, checkpoints=[]) :
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
            
    return results