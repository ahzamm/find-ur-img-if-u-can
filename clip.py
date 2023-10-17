import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_model(model_id="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    return processor, model


def encode_images(image_batch):
    processor, model = initialize_model()
    images = processor(
        text=None,
        images=image_batch,
        return_tensors='pt'
    )['pixel_values'].to(device)

    image_emb = model.get_image_features(images)
    image_emb = image_emb.detach().cpu().numpy()
    image_emb = image_emb.T / np.linalg.norm(image_emb, axis=1)
    image_emb = image_emb.T
    return image_emb


if __name__ == '__main__':
    ...