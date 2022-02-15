import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_dataset = torchvision.datasets.ImageFolder(root='/workspace/kli/ILSVRC2012/train', transform=data_transform)
train_dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

exit()
# train_dataset = torchvision.datasets.ImageFolder(root='ILSVRC2012/val', transform=data_transform)
# train_dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)