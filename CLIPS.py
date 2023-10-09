import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_model(model_id="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    return processor, model


processor, model = initialize_model()


def encode_images(image_batch):
    images = processor(
        text=None,
        images=image_batch,
        return_tensors='pt'
    )['pixel_values'].to(device)

    img_emb = model.get_image_features(images)
    img_emb = img_emb.detach().cpu().numpy()
    img_emb = img_emb.T / np.linalg.norm(img_emb, axis=1)
    img_emb = img_emb.T
    return img_emb
    

if __name__ == '__main__':
    ...
