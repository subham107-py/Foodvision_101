import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision
from PIL import Image
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(model:torchvision.models=torchvision.models.efficientnet_b4,weights:torchvision.models=torchvision.models.EfficientNet_B4_Weights.DEFAULT,num_classes:int=101,
                 seed:int=42):
    """Creates an feature extractor model and transforms.
    
        Args:
            model (torchvision.models, optional): model architecture. Defaults to torchvision.models.efficientnet_b4.
            weights (torchvision.models, optional): pretrained weights for the model. Defaults to torchvision.models
            num_classes (int, optional): number of classes in the classifier head. 
                Defaults to 101.
            seed (int, optional): random seed value. Defaults to 42.
    
        Returns:
            model (torch.nn.Module): feature extractor model. 
            transforms (torchvision.transforms): image transforms.
        """
    model = model(weights=weights).to(device) 
    transform = weights.transforms() 
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1792, out_features=num_classes),
        )
        
    return model, transform


model, transform = create_model(
    model=torchvision.models.efficientnet_b4,
    weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT,
    num_classes=101,
    seed=42,
)


# ============================================================
# FOOD101 CLASS NAMES
# ============================================================

from torchvision import datasets
train_data = datasets.Food101(
    root="data",
    split="train",
    transform=transform,
    download=True
) # Training data

test_data = datasets.Food101(
    root="data",
    split="test",
    transform=transform,
    download=True
) # Testing data
class_names = train_data.classes
from pathlib import Path
image_path = Path("data/food-101/images")

def create_dataloaders(train_data,
                       test_data,
                       tranasforms:torchvision.transforms.Compose,
                       batch_size:int=32,
                       num_workers:int=8):
    """Creates dataloaders for training and testing datasets.
    
        Args:
            train_directory (str): path to the training dataset.
            test_directory (str): path to the testing dataset.
            transforms (torchvision.transforms.Compose): image transforms.
            batch_size (int, optional): number of samples per batch. Defaults to 32.
            num_workers (int, optional): number of subprocesses to use for data loading. Defaults to 8.
        Returns:
        It will give tuple of train_dataloader, test_dataloader and class_names.
            train_dataloader (torch.utils.data.DataLoader): dataloader for training dataset.
            test_dataloader (torch.utils.data.DataLoader): dataloader for testing dataset.
            class_names (list): list of class names in the dataset.
            Example:
                train_dataloader, test_dataloader, class_names = create_dataloaders(train_directory="data/food-101/images/train",
                                                                                     test_directory="data/food-101/images/test",
                                                                                     transforms=transform,
                                                                                     batch_size=32,
                                                                                     num_workers=4)
        """
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    ) # Training dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    ) # Testing dataloader
    return train_dataloader, test_dataloader, train_data.classes
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_data=train_data,
    test_data=test_data,
    tranasforms=transform,
    batch_size=32,
    num_workers=4
)