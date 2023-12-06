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

# Import Networks
#from networks.resnet import ResNet18
from networks.densenet import DenseNet121, DenseNet169, DenseNet121_2, DenseNet121_3 

# Import tools
from tools.pytorch_helper import print_size_of_model

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

def calibrate(loader, batchsize, model, device):
   
    model.eval()

    # forward pass
    with torch.set_grad_enabled(False):
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Affichage
            sys.stdout.write(f'\r Calibrate : {i + 1}/{len(loader)} ')

            outputs = model(inputs)

    return 0

# Save method
def save(name, description, path, results, net=None, net_class=None) :
    full_path = path + name
    if (os.path.isdir(full_path)) :
        print("Repertory exists")
    else :
        os.mkdir(full_path)
    
    # save description
    if description != None :
        try :
            file = open(full_path + "/description.txt", "w+")
            # Save Network Size
            if net != None :
                file.write(print_size_of_model(net))
            file.write(description)
            file.close()
            print("Description Saved")
        except :
            print("Description Save Error")

    # save the weights of the network
    if net != None:
        torch.save(net.state_dict(), full_path + "/" + name + ".pth")
        print("Weights Saved")

    # save torchinfo results
    if net_class != None :
        model_stats = str(summary(net_class, verbose=0))
        try :
            file = open(full_path + "/torchinfo.txt", "w+")
            file.write(model_stats)
            file.close()
            print("Torchinfo Saved")
        except :
            print("Torchinfo save error")

    # save results 
    with open(full_path +"/result.csv", "w+") as saveFile :
            w = csv.writer(saveFile)
            w.writerow(results[0])
            w.writerow(results[1])
            w.writerow(results[2])
            w.writerow(results[3])
    
    return True

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


if __name__ == "__main__" : 
# Exemple :
    experiment_Name = "DenseNet6"
    description = "DensetNet 121 with Cross Entropy Loss, SGD (lr=0.1, momentum=0.9), lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)"
    model = DenseNet121()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    # Entrainement et 
    results = train(n_epochs, net, train_loader, test_loader, batchSizeTrain, batchSizeTest, criterion, optimizer, device, scheduler)

    # sauvegarde des résultats
    save(experimentName, description, resultsPath, results, model, DenseNet121())

def run_epoch_half_precision(loader, batchsize, model, criterion, optimizer, device, scheduler=None, mode="train"):
   
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
            inputs = inputs.type(torch.half)

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

# Function to use to mixup data

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Function to use to mixup data

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Function to use to run an epoch with mixup data
def run_epoch_mixup(loader, batchsize, model, criterion, optimizer, device, scheduler=None, mode="train"):
   
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

            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, use_cuda=True)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            optimizer.zero_grad()
            outputs = model(inputs)
          
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

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


# Function to use to train on mixup data

def train_mixup( n_epochs, net, trainLoader, testLoader, batchsize_Train, batchsize_Test, criterion, optimizer,device, scheduler=None) : 
    accuracies_train = ["Training Accuracy"]
    running_losses_train = ["Training Running Loss"]
    accuracies_test = ["Validation Accuracy"]
    running_losses_test = ["Validation Running Loss"]

    for epoch in range(0, n_epochs) :
        print(f'\nEpoch {epoch + 1}/{n_epochs}')
        # Train
        result = run_epoch_mixup(trainLoader, batchsize_Train, net, criterion, optimizer, device, scheduler, mode="train")
        accuracies_train.append(result[0])
        running_losses_train.append(result[1])

        # Validation 
        result = run_epoch(testLoader, batchsize_Test, net, criterion, optimizer, device, scheduler, mode="eval")
        accuracies_test.append(result[0])
        running_losses_test.append(result[1])

    return [accuracies_train, running_losses_train, accuracies_test, running_losses_test]

# To perform structure pruning on a model
def structured_pruning(model, pruning_threshold, first_load, cpt=None) :
    if first_load :
        model.load_state_dict(cpt)

    #  List of parameters we want to prune
    parameters_to_prune = []
    for name, layer in model.named_modules() :
        if isinstance(layer, torch.nn.Conv2d) :
            temp = name.split(sep='.')
            if (len(temp) == 3) :
                parameters_to_prune.append((getattr(getattr(model, temp[0])[int(temp[1])], temp[2]), 'weight'))
                prune.ln_structured(getattr(getattr(model, temp[0])[int(temp[1])], temp[2]), 'weight', amount=pruning_threshold, n=float("inf"), dim=1)
    