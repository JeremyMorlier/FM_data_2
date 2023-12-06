# Import Libraries
import numpy as np
import sys
import csv
import os
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchinfo import summary


# Distillation run epoch
def run_epoch_distillation(loader, batchsize, teacher, student, criterion, optimizer, device, scheduler=None, mode="train") :
    
    # mode evaluation
    if mode == "train" :
        student.train()
        name = "Train"
    elif mode == "eval" :
        student.eval()
        name = "Eval"
    else :
        return 0
    running_loss = 0.0
    accuracy = 0

    # forward pass
    with torch.set_grad_enabled(mode == "train") :
        for i, (inputs, labels) in enumerate(loader) :
            inputs = inputs.to(device)

            optimizer.zero_grad()
            teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)

            loss = criterion

# Train method
def train( n_epochs, net, trainLoader, testLoader, batchsize_Train, batchsize_Test, criterion, optimizer,device, scheduler=None) : 
    accuracies_train = ["Training_Accuracy"]
    running_losses_train = ["Training_Running_Loss"]
    accuracies_test = ["Validation_Accuracy"]
    running_losses_test = ["Validation_Running_Loss"]

    for epoch in range(0, n_epochs) :
        print(f'\nEpoch {epoch + 1}/{n_epochs}')
        # Train
        result = run_epoch(trainLoader, batchsize_Train, net, criterion, optimizer, device, scheduler, mode="train")
        accuracies_train.append(result[0])
        running_losses_train.append(result[1])

        # Validation 
        result = run_epoch(testLoader, batchsize_Test, net, criterion, optimizer, device, scheduler, mode="eval")
        accuracies_test.append(result[0])
        running_losses_test.append(result[1])

    return [accuracies_train, running_losses_train, accuracies_test, running_losses_test]


# Train, eval mode
def run_epoch(loader, batchsize, model, criterion, optimizer, device, scheduler=None, mode="train"):
   
    # mode evaluation
    if mode == "train" :
        model.train()
        name = "Train"
    elif mode == "eval" :
        model.eval()
        name = "Eval"
    else :
        return 0
    running_loss = 0.0
    accuracy = 0

    # forward pass
    with torch.set_grad_enabled(mode == "train"):
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            if mode == "train" :
                loss.backward()
                optimizer.step()

            # Running Loss and Accuracy 
            running_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(labels.view_as(pred)).sum().item()

            # Affichage
            sys.stdout.write(f'\r {name} : {i + 1}/{len(loader)} - acc {round(accuracy / ((1 + i) * batchsize), 3)} ' f' - loss {round(running_loss / ((1 + i) * batchsize), 3)}                       ')

    print()
    return [round(accuracy / ((1 + i) * batchsize), 3), round(running_loss / ((1 + i) * batchsize), 3)]