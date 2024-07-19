import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import openai

class ImageFeatureExtractor:
    def __init__(self):
        self.model = self._setup_model()
        self.preprocess = self._setup_preprocessing()

    def _setup_model(self):
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last layer
        model.eval()  # Set the model to evaluation mode
        return model

    def _setup_preprocessing(self):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess

    def extract_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # No need to compute gradients
            features = self.model(image)
        return features.flatten().numpy()

class OpenAIEmbedder:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name

    def embed(self, docs: list[str]) -> list[list[float]]:
        res = openai.embeddings.create(
            input=docs,
            model=self.model_name
        )
        doc_embeds = [r.embedding for r in res.data]
        return doc_embeds

# Example usage
if __name__ == "__main__":
    # Image feature extraction
    image_feature_extractor = ImageFeatureExtractor()
    features = image_feature_extractor.extract_features("path/to/image.jpg")
    print("Extracted Features:", features)

    # OpenAI embedding
    openai_embedder = OpenAIEmbedder()
    doc_embeddings = openai_embedder.embed(["This is a sample document."])
    print("Document Embeddings:", doc_embeddings)
