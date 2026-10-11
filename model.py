import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision
from PIL import Image
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(model:torchvision.models=torchvision.models.convnext_tiny,
                 weights:torchvision.models=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
                 freeze_backbone:bool=False):
    """Creates an feature extractor model and transforms.
    
        Args:
            model (torchvision.models, optional): model architecture. Defaults to torchvision.models.convnext_tiny.
            weights (torchvision.models, optional): pretrained weights for the model. Defaults to torchvision.models.ConvNext_Tiny_Weights.DEFAULT.
            freeze_backbone(bool): It is for freezing backbone of the model which reduces the training parameters upto 80-90%
    
        Returns:
            model (torch.nn.Module): feature extractor model.
            train_transform: training transforms to training images
            transforms (torchvision.transforms): image transforms.
        """
    model = model(weights=weights).to(device) # 1. Load pretrained model with weights
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    from torchvision import transforms

    transform = weights.transforms()
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform.mean, std=transform.std),
    ])
    
        
    return model, train_transform,transform


model, train_transform, transform = create_model(model=torchvision.models.convnext_tiny,
                               weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
                               freeze_backbone=False)


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
                       batch_size:int=32,
                       num_workers:int=8):
    """Creates dataloaders for training and testing datasets.
    
        Args:
            train_directory (str): path to the training dataset.
            test_directory (str): path to the testing dataset.
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
    batch_size=32,
    num_workers=4
)