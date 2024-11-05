import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_data = datasets.CIFAR10(
    root='../../data/cifar10_data', 
    train=True, 
    transform=transforms.ToTensor(),
    download=True)

validation_data = datasets.CIFAR10(
    root='../../data/cifar10_data', 
    train=False, 
    transform=transforms.ToTensor(),
    download=True)

labels_dict ={
    0: "plane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
