import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from vendi_score import data_utils, vendi
from vendi_score.data_utils import Example
import clip  # OpenAI's CLIP library


# Define CLIP transform
def clip_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for CLIP
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # CLIP's mean and std
    ])

# Get embeddings using CLIP
def get_embeddings(images, model=None, device="cpu", batch_size=32):
    if type(device) == str:
        device = torch.device(device)
    model, _ = clip.load(model, device=device)

    transform = clip_transforms()
    embeddings = []
    for batch in data_utils.to_batches(images, batch_size):
        inputs = torch.stack([transform(img) for img in batch]).to(device)

        with torch.no_grad():
            output = model.encode_image(inputs).cpu().numpy()
        embeddings.append(output)
    return np.concatenate(embeddings, 0)

# Vendi score for embeddings
def embedding_vendi_score(images, batch_size=32, device="cpu", model=None):
    X = get_embeddings(images, batch_size=batch_size, device=device, model=model)
    n, d = X.shape
    if n < d:
        return vendi.score_X(X)
    return vendi.score_dual(X)

# Plot images
def plot_images(images, cols=None, ax=None):
    if cols is None:
        cols = len(images)
    if ax is None:
        fig, ax = plt.subplots()
    rows = data_utils.to_batches([np.array(x) for x in images], cols)
    shape = rows[0][0].shape
    while len(rows[-1]) < cols:
        rows[-1].append(np.zeros(shape))
    rows = [np.concatenate(row, 1) for row in rows]
    ax.imshow(np.concatenate(rows, 0))
    ax.set_xticks([])
    ax.set_yticks([])