import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path, num_classes, device=None):
    """
    Load a ResNet50 model for inference.
    Handles both state_dict and full model checkpoints.
    """

    # Auto device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Try loading checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
            # Case 1: Only state_dict was saved
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(checkpoint)

        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # Case 2: Saved with trainer wrapper (common in PyTorch Lightning)
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(checkpoint["state_dict"])

        else:
            # Case 3: Entire model was saved with torch.save(model, ...)
            model = checkpoint

        model.to(device)
        model.eval()
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")