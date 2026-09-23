import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision
from PIL import Image
from timeit import default_timer as timer
from effnet_model import class_names

def predict(image, model, transform, device, class_names):

    start_time = timer()

    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    # Get top 5 predictions
    top_probs, top_indices = torch.topk(
        probabilities,
        k=5
    )

    predictions = {}

    for prob, index in zip(
        top_probs[0],
        top_indices[0]
    ):
        predictions[
            class_names[index.item()]
        ] = prob.item()

    prediction_time = round(
        timer() - start_time,
        5
    )

    return predictions, prediction_time