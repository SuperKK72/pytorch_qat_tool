import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time
import copy
import numpy as np

from torch_models.resnet import resnet18
from torch_models.mobilenetv1 import MobileNet
from torch_models.mobilenet_v2 import *
from data_loader.val_data_loader import *

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform)

    # train_set = torchvision.datasets.ImageNet(root="data", split='train', download=False, transform=train_transform)
    # test_set = torchvision.datasets.ImageNet( root="data", split='test', download=False, transform=test_transform)
    # exit()

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader


def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)
    running_loss = 0
    running_corrects = 0
    eval_batch_idx = 0
    batch_num = int(len(test_loader.dataset.images) / test_loader.batch_size)
    for inputs, labels in test_loader:
        if eval_batch_idx%100==0:
            print("eval batch {}/{} at engine {}...".format(eval_batch_idx, batch_num, torch.backends.quantized.engine))
        eval_batch_idx += 1
        inputs = inputs.to(device)
        labels = labels.squeeze(dim=1)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0
        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    return eval_loss, eval_accuracy


def train_model(model, train_loader, test_loader, device, learning_rate=1e-4, num_epochs=20):

    # The training configurations were not carefully selected.
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0
        running_corrects = 0
        train_batch_idx = 0
        batch_num = int(train_loader.batch_sampler.sampler.num_samples/train_loader.batch_size)
        for inputs, labels in train_loader:
            if train_batch_idx%100==0:
                print("Epoch {}/{}\tbatch {}/{} at engine {}...".format(epoch, num_epochs, train_batch_idx, batch_num,
                                                                        torch.backends.quantized.engine))
            train_batch_idx += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)
        #revised by kli: fuse relevant parameters
        if epoch > 3:
            model.apply(torch.quantization.disable_observer)
        if epoch > 2:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(
            model=model, test_loader=test_loader, device=device, criterion=criterion)
        # Set learning rate scheduler
        lr_scheduler.step()
        r'''display model loss and accuracy per epoch'''
        print("Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
            epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))
    return model


def calibrate_model(model, loader, device=torch.device("cpu:0")):
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def measure_inference_latency(model, device, input_size=(1, 3, 32, 32), num_samples=100):
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    start_time = time.time()
    for _ in range(num_samples):
        _ = model(x)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1, 3, 32, 32)):
    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False
    return True

def display_named_modules(model):
    for name, mod in enumerate(model.named_modules()):
        print(name)
        print(mod)
        print("---------------------------------------------")
    print("-----------------------END------------------------")

def display_named_children(model):
    for name, mod in enumerate(model.named_children()):
        print(name)
        print(mod)
        print("---------------------------------------------")
    print("-----------------------END------------------------")