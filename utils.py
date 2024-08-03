# import torch
# from torchvision import models, transforms
from PIL import Image
import openai
import os
import base64
import io
import requests
import json
from dotenv import load_dotenv

load_dotenv()

open_api_key = os.environ.get('OPENAI_API_KEY')

openai.api_key=open_api_key


class ImageFeatureExtractor:
    def __init__(self):
        # self.model = self._setup_model()
        # self.preprocess = self._setup_preprocessing()
        self.api_url = "https://f783tgqxjb.execute-api.eu-north-1.amazonaws.com/dev"

    # def _setup_model(self):
    #     model = models.resnet50(pretrained=True)
    #     model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last layer
    #     model.eval()  # Set the model to evaluation mode
    #     return model

    # def _setup_preprocessing(self):
    #     preprocess = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     return preprocess
    
    # def _encode_image_to_base64(self,image_path):
    #     """ Encodes an image to a base64 string. """
    #     with Image.open(image_path) as image:
    #         buffered = io.BytesIO()
    #         image.save(buffered, format="JPEG")  # You can change JPEG to PNG if necessary
    #         return base64.b64encode(buffered.getvalue()).decode()
        

    def _encode_image_to_base64(self,image):
        """ Encodes an image to a base64 string. """
        buffered = io.BytesIO()
        image_format = image.format if image.format else "JPEG"  # Default to JPEG if format is None

        image.save(buffered, format=image_format)  # You can change JPEG to PNG if necessary
        return base64.b64encode(buffered.getvalue()).decode()   
    
    def _invoke_api_with_image(self,api_url, base64_string):
        """ Sends a POST request to the API with the base64-encoded image. """
        headers = {'Content-Type': 'application/json'}
        payload = {'body': base64_string}
        response = requests.post(api_url, json=payload, headers=headers)
        return response.json()

    # def extract_features(self, image_path):
    #     image = Image.open(image_path).convert('RGB')
    #     image = self.preprocess(image).unsqueeze(0)  # Add batch dimension
    #     with torch.no_grad():  # No need to compute gradients
    #         features = self.model(image)
    #     return features.flatten().numpy()
    
    def extract_features_lambda(self, image):
        base64_encoded_image = self._encode_image_to_base64(image)
        feats = self._invoke_api_with_image(self.api_url, base64_encoded_image)
        embeddings=json.loads(feats['body'])['embeddings']
        return embeddings


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
