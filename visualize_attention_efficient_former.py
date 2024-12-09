import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_loader import PKLDatasetSquare
from efficient_former.efficientformer_models import efficientformerv2_s1
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the original pretrained model
pretrained_model_path = "checkpoints/2024-12-04_11-44-01/vit_model_105.pt" 
pretrained_state_dict = torch.load(pretrained_model_path)

PATCH_SIZE = 16
GOAL_SIZE = 3

# Initialize the new model
visualizable_model = efficientformerv2_s1(pretrained=False, resolution=128, distillation = False, visualize_attention=True).to(device)

# Load the weights into the new model
visualizable_model.load_state_dict(pretrained_state_dict)
visualizable_model.eval()

# Load data
eval_data_dir = os.path.join(os.path.dirname(__file__), "data/eval_crossings")
eval_dataset = PKLDatasetSquare(eval_data_dir)
eval_data_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True)

# Get a batch of data
for image_data, non_image_data, actions in eval_data_loader:
    
        image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
        goal = non_image_data[:, -3:]

        with torch.no_grad():
            embeddings = visualizable_model.forward_omnidir(image_data, goal)
    
        plt.close()
        utils.visualize_attention(image_data, visualizable_model)

