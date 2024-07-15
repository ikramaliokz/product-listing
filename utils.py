import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import openai


# Pre-trained ResNet50 model setup
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last layer
model.eval()  # Set the model to evaluation mode

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # No need to compute gradients
        features = model(image)
    return features.flatten().numpy()


# openai embedding function
def embed(docs: list[str]) -> list[list[float]]:
    res = openai.embeddings.create(
        input=docs,
        model="text-embedding-ada-002"
    )
    doc_embeds = [r.embedding for r in res.data]
    return doc_embeds

