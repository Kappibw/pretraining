import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_loader import PKLDatasetStrip
from got_embedder.goal_oriented_transformer import GoTWithAttentionMaps
import utils


def token_to_patch_coords(token_index, patch_height, patch_width, image_height, image_width):
    """
    Maps a token index to the coordinates of the square it annotates in the original image.

    Args:
        token_index (int): The index of the token in the flattened sequence.
        patch_height (int): The height of each patch.
        patch_width (int): The width of each patch.
        image_height (int): The height of the original image.
        image_width (int): The width of the original image.

    Returns:
        tuple: ((y_start, y_end), (x_start, x_end)) - The top-left and bottom-right pixel
               coordinates of the patch in the original image.
    """
    # Compute the number of patches along height and width
    num_patches_height = image_height // patch_height

    # Get the row and column of the patch
    col = token_index // num_patches_height
    row = token_index % num_patches_height

    # Calculate the pixel coordinates in the original image
    y_start = row * patch_height
    y_end = y_start + patch_height
    x_start = col * patch_width
    x_end = x_start + patch_width

    return (y_start, y_end), (x_start, x_end)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the original pretrained model
pretrained_model_path = "checkpoints/2024-11-25_14-47-10/vit_model_182.pt" 
pretrained_state_dict = torch.load(pretrained_model_path)

PATCH_SIZE = 16
GOAL_SIZE = 3
TEST_IMAGE = False

# Initialize the new model
visualizable_model = GoTWithAttentionMaps(
    image_size=(64, 256),
    patch_size=(PATCH_SIZE, PATCH_SIZE),
    num_classes=2,
    dim=32,
    depth=2,
    heads=4,
    goal_size=GOAL_SIZE,
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
eval_dataset = PKLDatasetStrip(eval_data_dir)
eval_data_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True)

patch_idx = 10

# Get a batch of data
for image_data, non_image_data, actions in eval_data_loader:

    # Load test data
    if TEST_IMAGE:
        image_data = torch.load("data/test/cylinder_validation.pt").to(device)
        image_data = image_data.permute(0, 3, 1, 2)
        goal = torch.tensor([[7.0, 0.0, 0.6]]).to(device)
        if GOAL_SIZE == 2:
            goal = torch.tensor([[10.0, 0.0]]).to(device)
    else:
        image_data = image_data[:, :2].to(device)
        goal = non_image_data[:, -3:].to(device)
        if GOAL_SIZE == 2:
            goal = utils.cartesian_to_polar(goal)

    plt.close()
    with torch.no_grad():
        output, attention_maps = visualizable_model(image_data, goal)

    batch_idx = 0
    layer_idx = 0
    second_layer_attention = attention_maps[layer_idx][batch_idx]  # Shape: [heads, n, n]

    # Visualize attention across all the heads
    attn_map = second_layer_attention.mean(dim=0)  # Shape: [n, n]

    # Select the CLS token's attention
    goal_attention = attn_map[0, 1:]  # Exclude the CLS token itself
    goal_attention = goal_attention.view(4, 16)  # Reshape to 2D

    patch_idx = patch_idx + 1
    patch_attention = attn_map[patch_idx, 1:].view(4, 16)

    (y_start, y_end), (x_start, x_end) = token_to_patch_coords(
        patch_idx, PATCH_SIZE, PATCH_SIZE, 64, 256
    )

    # Interpolate the attention map to the input image resolution
    goal_attention_resized = F.interpolate(
            goal_attention.unsqueeze(0).unsqueeze(0), size=(64, 256), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
    patch_attention_resized = F.interpolate(
            patch_attention.unsqueeze(0).unsqueeze(0), size=(64, 256), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
    
    # goal_attention_resized = np.kron(goal_attention.cpu(), np.ones((PATCH_SIZE, PATCH_SIZE)))
    # patch_attention_resized = np.kron(patch_attention.cpu(), np.ones((PATCH_SIZE, PATCH_SIZE)))

    # Convert the image tensor to a displayable format
    img_to_display = image_data[0, 0].cpu().numpy()  # Assuming a single-channel image

    print("Max values: Goal: ", goal_attention_resized.max(), " Mean: ", patch_attention_resized.max())

    # Normalize for visualization
    MAX_NORMALIZATION = 0.02
    max_goal_attn = goal_attention_resized.max()
    goal_attention_resized /= max_goal_attn
    patch_attention_resized /= MAX_NORMALIZATION# patch_attention_resized.max()

    # Determine goal direction
    goal_x, goal_y = goal[0, 0].item(), goal[0, 1].item()  # Extract goal x and y values
    distance = math.sqrt(goal_x ** 2 + goal_y ** 2)
    goal_x_label = "Front" if goal_x > 0 else "Back"
    goal_y_label = "Left" if goal_y > 0 else "Right"
    goal_text = f"Goal: {goal_x_label} {goal_y_label} ({goal_x:.2f}, {goal_y:.2f})   Max Attn: {max_goal_attn:.2f}"
    # Get goal angle vector
    goal_angle = math.atan2(goal_y, goal_x) * -1  # Go clockwise from front
    # Map to location on image
    front_offset = 128
    goal_angle_in_pixels = (goal_angle / np.pi) * 128 # Range from -128 to 128
    goal_x_pixel = int(goal_angle_in_pixels + front_offset)
    goal_y_pixel = 32

    # Plot Goal Attention Overlay
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(img_to_display, cmap="gray")
    plt.imshow(goal_attention_resized, cmap="jet", alpha=0.3, vmin=0, vmax=1.0)
    plt.colorbar(label="Goal Attention")
    plt.title("Goal Token Attention")

    # Add labels for quadrants
    plt.text(128, 32, "Front", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    plt.text(192, 32, "Right", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    # plt.text(32, 96, "Back", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    plt.text(64, 32, "Left", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)

    # Plot goal location
    plt.scatter(goal_x_pixel, goal_y_pixel, color="cyan", s=144, marker="x", zorder=10)

    # Plot Total Attention Overlay
    plt.subplot(2, 1, 2)
    plt.imshow(img_to_display, cmap="gray")
    plt.imshow(patch_attention_resized, cmap="jet", alpha=0.3, vmin=0, vmax=1.0)
    plt.colorbar(label="Patch Attention")
    plt.title("Patch Attention")

    rect = patches.Rectangle(
        (x_start, y_start),  # Bottom-left corner of the rectangle
        PATCH_SIZE,         # Width of the rectangle
        PATCH_SIZE,        # Height of the rectangle
        linewidth=2,         # Line width
        edgecolor="red",     # Color of the rectangle edge
        facecolor="none"     # No fill
    )
    plt.gca().add_patch(rect)

    # Add labels for quadrants
    plt.text(128, 32, "Front", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    plt.text(192, 32, "Right", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    # plt.text(32, 96, "Back", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)
    plt.text(64, 32, "Left", color="white", fontsize=12, ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), zorder=5)

    plt.figtext(0.5, 0.01, goal_text, ha="center", fontsize=14, bbox=dict(facecolor="white", edgecolor="black"))

    # Display the plots
    plt.tight_layout()
    plt.show()