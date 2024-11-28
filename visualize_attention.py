import torch
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_loader import PKLDataset
from got_embedder.goal_oriented_transformer import GoTWithAttentionMaps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the original pretrained model
pretrained_model_path = "checkpoints/2024-11-25_14-47-10/vit_model_182.pt" 
pretrained_state_dict = torch.load(pretrained_model_path)

# Initialize the new model
visualizable_model = GoTWithAttentionMaps(
    image_size=(128, 128),
    patch_size=(16, 16),
    num_classes=2,
    dim=32,
    depth=2,
    heads=4,
    goal_size=3,
    mlp_dim=2048,
    channels=2,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
).to(device)

# Load the weights into the new model
visualizable_model.load_state_dict(pretrained_state_dict)
visualizable_model.eval()  # Disable dropout and other training-specific behaviors

# Load data
eval_data_dir = os.path.join(os.path.dirname(__file__), "data/eval")
eval_dataset = PKLDataset(eval_data_dir)
eval_data_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True)

# Get a batch of data
for image_data, non_image_data, actions in eval_data_loader:
    plt.close()
    with torch.no_grad():
        image_data = image_data[:, :2].to(device)
        goal = non_image_data[:, -3:].to(device)
        output, attention_maps = visualizable_model(image_data, goal)


    batch_idx = 0
    layer_idx = 1
    second_layer_attention = attention_maps[layer_idx][batch_idx]  # Shape: [heads, n, n]

    # Visualize attention across all the heads
    attn_map = second_layer_attention.mean(dim=0)  # Shape: [n, n]

    # Select the CLS token's attention
    goal_attention = attn_map[0, 1:]  # Exclude the CLS token itself
    goal_attention = goal_attention.view(8, 8)  # Reshape to 2D (8x8 for a 128x128 input with 16x16 patches)

    mean_attention = attn_map[:, 1:].mean(dim=0).view(8, 8)

    # Interpolate the attention map to the input image resolution
    goal_attention_resized = F.interpolate(
            goal_attention.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
    mean_attention_resized = F.interpolate(
            mean_attention.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
    
    # goal_attention_resized = np.kron(goal_attention.cpu(), np.ones((16, 16)))
    # mean_attention_resized = np.kron(mean_attention.cpu(), np.ones((16, 16)))

    # Convert the image tensor to a displayable format
    img_to_display = image_data[0, 0].cpu().numpy()  # Assuming a single-channel image

    print("Max values: Goal: ", goal_attention_resized.max(), " Mean: ", mean_attention_resized.max())

    # Normalize for visualization
    MAX_NORMALIZATION = 0.5
    max_goal_attn = goal_attention_resized.max()
    goal_attention_resized /= max_goal_attn
    mean_attention_resized /= mean_attention_resized.max()
    img_to_display = (img_to_display - img_to_display.min()) / (img_to_display.max() - img_to_display.min())

    # Determine goal direction
    goal_x, goal_y = goal[0, 0].item(), goal[0, 1].item()  # Extract goal x and y values
    distance = math.sqrt(goal_x ** 2 + goal_y ** 2)
    goal_x_label = "Front" if goal_x > 0 else "Back"
    goal_y_label = "Left" if goal_y > 0 else "Right"
    goal_text = f"Goal: {goal_x_label} {goal_y_label}   Max Attn: {max_goal_attn:.2f}"
    # Get goal angle vector
    goal_angle = math.atan2(goal_y, goal_x) * -1  # Go clockwise from front
    # Map to location on image
    front_offset = 32
    goal_angle_in_pixels = (goal_angle / np.pi) * 128 # Range from -128 to 128
    goal_x_pixel = int(goal_angle_in_pixels + front_offset)
    goal_y_pixel = 32
    if goal_x_pixel < 0:
        goal_x_pixel += 128
        goal_y_pixel = 64 + goal_y_pixel
    elif goal_x_pixel >= 128:
        goal_x_pixel -= 128
        goal_y_pixel = 64 + goal_y_pixel

    # Plot Goal Attention Overlay
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_to_display, cmap="gray")
    plt.imshow(goal_attention_resized, cmap="jet", alpha=0.3, vmin=0, vmax=1.0)
    plt.colorbar(label="Goal Attention")
    plt.title("Goal Token Attention")

    # Add labels for quadrants
    plt.text(32, 32, "Front", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    plt.text(96, 32, "Right", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    plt.text(32, 96, "Back", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    plt.text(96, 96, "Left", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)

    # Plot goal location
    plt.scatter(goal_x_pixel, goal_y_pixel, color="cyan", s=144, marker="x", zorder=10)

    # Plot Total Attention Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_to_display, cmap="gray")
    plt.imshow(mean_attention_resized, cmap="jet", alpha=0.3, vmin=0, vmax=1.0)
    plt.colorbar(label="Total Attention")
    plt.title("Full Cross-Attention")

    # Add labels for quadrants
    plt.text(32, 32, "Front", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7))
    plt.text(96, 32, "Right", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7))
    plt.text(32, 96, "Back", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7))
    plt.text(96, 96, "Left", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7))

    plt.figtext(0.5, 0.01, goal_text, ha="center", fontsize=14, bbox=dict(facecolor="white", edgecolor="black"))

    # Display the plots
    plt.tight_layout()
    plt.show()