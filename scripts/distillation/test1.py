import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import open_clip

from scripts.distillation.training import run_epoch
# Parameters
# Dataset parameters
data_path = "/nasbrain/datasets/imagenet"
batch_size_train = 10
batch_size_test = 10

# Training parameters
n_epochs = 20

# Models parameters
teacher_name = 'ViT-B-32'
student_name = 'ViT-S-32-alt'

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose(
        [ transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225)),
        transforms.RandomCrop(224)
        ]) 

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225)),
        transforms.RandomCrop(224)
        ])

# Train and Test Loader
train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet(root=data_path, train=True, download=True, transform=transform_train),
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet(root=data_path, train=False, download=True, transform=transform_test),
        batch_size=batch_size_test,
        shuffle=False, num_workers=4)



# teacher
teacher_model, _, teacher_preprocess = open_clip.create_model_and_transforms(teacher_name, pretrained='laion2b_s34b_b79k')
teacher_tokenizer = open_clip.get_tokenizer(teacher_name)

teacher_text = text = teacher_tokenizer(["a diagram", "a dog", "a cat"])
# student 
student_model, _, student_preprocess = open_clip.create_model_and_transforms(student_name)
student_tokenizer = open_clip.get_tokenizer(student_name)


criterion = nn.MSELoss(reduction='sum')

for epoch in n_epochs :
    for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model.encode_image(inputs)
            student_outputs = student_model.encode_image(inputs)
            print(teacher_outputs.size(), student_outputs.size())

            loss = criterion(student_outputs, teacher_outputs)
            loss.backward()
            
