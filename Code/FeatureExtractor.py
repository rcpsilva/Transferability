import torch
from tqdm import tqdm
import numpy as np

def extract_features(loader, model):
    
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Move the model to the specified device
    model.to(device)

    
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)  # Move images to the device (GPU or CPU)
            outputs = model(images)
            features.append(outputs.cpu())  # Move outputs back to CPU for aggregation
            labels.extend(targets.cpu().numpy())  # Move labels to CPU and convert to numpy

    features = torch.cat(features).numpy()
    return features, np.array(labels)