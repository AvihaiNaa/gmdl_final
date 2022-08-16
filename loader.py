import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import model as DVAE
import torch.nn.functional as F
import copy
from torchvision.datasets import ImageFolder
import os


def load_fmnist(dataroot, imageSize, batch_size, workers):

    dataset_fmnist = dset.FashionMNIST(root=dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize((imageSize)),
                                transforms.ToTensor(),
                            ]))
    test_loader_fmnist = torch.utils.data.DataLoader(dataset_fmnist, batch_size=batch_size,
                                            shuffle=True, num_workers=int(workers))
    return test_loader_fmnist

def load_cifar(dataroot, imageSize, batch_size, workers):

    dataset_cifar = dset.CIFAR10(root=dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize((imageSize)),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ]))
    test_loader_cifar = torch.utils.data.DataLoader(dataset_cifar, batch_size=batch_size,
                                            shuffle=True, num_workers=int(workers))
    return test_loader_cifar

def load_svhn(dataroot, imageSize, batch_size, workers):
    dataset_svhn = dset.SVHN(root=dataroot, split = "test", download=True, transform=transforms.Compose([
                                transforms.Resize((imageSize)),
                                transforms.ToTensor(),
                            ]))
    test_loader_svhn = torch.utils.data.DataLoader(dataset_svhn, batch_size=batch_size,
                                            shuffle=True, num_workers=int(workers))
    return test_loader_svhn


def load_mnist_test(dataroot, imageSize, batch_size, workers):
    dataset_mnist_test = dset.MNIST(root=dataroot, train=False, download=True, transform=transforms.Compose([
                            transforms.Resize((imageSize)),
                            transforms.ToTensor(),
                        ]))
    test_loader_mnist= torch.utils.data.DataLoader(dataset_mnist_test, batch_size=batch_size,
                                        shuffle=True, num_workers=int(workers))
    return test_loader_mnist


def load_emnist_letters(dataroot, imageSize, batch_size, workers):
    
    dataset_emnist_letters = dset.EMNIST(root=dataroot, split= "letters", train=False, download=True, transform=transforms.Compose([
                            transforms.Resize((imageSize)),
                            transforms.ToTensor(),
                        ]))
    test_loader_emnist_letters= torch.utils.data.DataLoader(dataset_emnist_letters, batch_size=batch_size,
                                        shuffle=True, num_workers=int(workers))
    return test_loader_emnist_letters