import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import os
import numpy as np

from efficient_former.efficientformer_models import AttnFFN, Attention4D

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

###########################
##### TRAINING UTILS ######
###########################

def save(vit_model, actor_critic, save_dir, idx):
    vit_model_path = os.path.join(save_dir, f"vit_model_{idx}.pt")
    actor_critic_path = os.path.join(save_dir, f"actor_critic_{idx}.pt")

    # Save the models' state dictionaries
    torch.save(vit_model.state_dict(), vit_model_path)
    torch.save(actor_critic.state_dict(), actor_critic_path)

    print(f"Model idx {idx} saved to {save_dir}")

def cartesian_to_polar(cartesian_coords: torch.Tensor) -> torch.Tensor:
    # Radius
    r = torch.norm(cartesian_coords, dim=-1)
    # Azimuth (angle in the xy-plane)
    phi = torch.atan2(cartesian_coords[..., 1], cartesian_coords[..., 0])
    # Return a tensor of shape [batch, 2] with r and phi
    return torch.stack((r, phi), dim=1)


##############################
#### VISUALIZATION UTILS #####
##############################

def log_attention_mask(input_image, model, epoch, batch_idx):
    attention_maps = []
    for module in model.network:
        if isinstance(module, torch.nn.Sequential):
            for block in module:
                if isinstance(block, AttnFFN) and isinstance(block.token_mixer, Attention4D):
                    attention_maps.append(block.token_mixer.attention_map)

    if attention_maps:
        attn_map = attention_maps[-1][0]  # Last layer's attention map, first sample in the batch
        num_heads = attn_map.shape[0]

        total_attention = torch.sum(attn_map, dim=0)
        # Normalize attention between 0 and 1
        total_attention = (total_attention - torch.min(total_attention)) / (
            torch.max(total_attention) - torch.min(total_attention)
        )

        # Upsample the attention map to the input image size
        total_attention = F.interpolate(
            total_attention.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze()

        # Ensure input_image is in the right range for display
        normalized_image = torch.where(input_image == 0.0, 40.0, input_image)

        fig, ax = plt.subplots()
        ax.imshow(normalized_image[0, 0].cpu().numpy(), cmap="gray")
        ax.imshow(total_attention, cmap="coolwarm", alpha=0.3)
        ax.set_title(f"Sum Attention for all {num_heads} heads")
        # Add colorbar
        cbar = plt.colorbar(ax.imshow(total_attention, cmap="coolwarm", alpha=0.5))

        # Capture the figure as an image to log to wandb
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        wandb_img = wandb.Image(img, caption=f"Epoch {epoch+1}, Batch {batch_idx+1}")
        plt.close(fig)  # Close figure to release memory

        return wandb_img
    
def log_got_attention_map(image, attentions, epoch, batch_idx, img_idx=0):
    """
    Visualizes the averaged attention for the cls token across all heads from multiple layers 
    as heatmaps over the original image.

    :param image: torch.Tensor of shape [3, H, W] representing the original image.
    :param attentions: List of attention maps from each transformer layer. Each attention map is
                       a tensor of shape [batch, heads, patches], focused on cls token attention.
    :param epoch: Current training epoch.
    :param batch_idx: Current batch index.
    :param img_idx: Index of the image within the batch to visualize.
    """
    # Assume attentions list is structured as [layer_attention] with shape [1, heads, num_patches]
    num_layers = len(attentions)
    patch_size = 16  # Patch size used in the model

    # Prepare plot for each layer's averaged attention
    fig, axes = plt.subplots(num_layers, 1, figsize=(5, num_layers * 5))
    if num_layers == 1:
        axes = [axes]  # Ensure axes is iterable for single layer case

    for layer_idx, layer_attn in enumerate(attentions):
        # Average attention across all heads
        avg_attn_map = layer_attn[img_idx].mean(dim=0)  # Shape: [num_patches]

        # Reshape attention to grid shape based on image and patch dimensions
        grid_size = int(image.shape[-1] / patch_size)  # e.g., 128 / 16 = 8
        avg_attn_map = avg_attn_map.reshape(grid_size, grid_size).detach().cpu()

        # Upsample the attention map to match the image dimensions
        avg_attn_map = F.interpolate(
            avg_attn_map.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze()

        # Normalize attention map for better contrast
        avg_attn_map = (avg_attn_map - avg_attn_map.min()) / (avg_attn_map.max() - avg_attn_map.min())

        normalized_image = torch.where(image == 0.0, 40.0, image)

        # Display the attention map overlayed on the original image
        ax = axes[layer_idx]
        ax.imshow(normalized_image[img_idx, 0].cpu().numpy(), cmap="gray")
        ax.imshow(avg_attn_map.cpu().numpy(), cmap="coolwarm", alpha=0.5)
        ax.axis("off")
        ax.set_title(f"Layer {layer_idx + 1}")

    plt.tight_layout()
    # plt.show()

    # Convert the plot to an image for wandb
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    wandb_img = wandb.Image(img, caption=f"Epoch {epoch+1}, Batch {batch_idx+1}")
    # plt.close(fig)  # Close figure to release memory

    return wandb_img

   
def log_got_attention_map_separate_heads(image, attentions, epoch, batch_idx, img_idx=0):
    """
    Visualizes the attention for the cls token from multiple layers as heatmaps over the original image.
    
    :param image: torch.Tensor of shape [3, H, W] representing the original image.
    :param attentions: List of attention maps from each transformer layer. Each attention map is
                       a tensor of shape [batch, heads, patches], focused on cls token attention.
    :param patch_size: Size of each image patch.
    """
    # Assume attentions list is structured as [layer_attention] with shape [1, heads, num_patches]
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    patch_size = 16  # Patch size used in the model

    # Plot each layer's attention as a heatmap
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))
    for layer_idx, layer_attn in enumerate(attentions):
        for head_idx in range(num_heads):
            attn_map = layer_attn[img_idx, head_idx]  # Take attention from this layer and head

            # Reshape attention to grid shape based on image and patch dimensions
            grid_size = int(image.shape[-1] / patch_size)  # e.g., 128 / 16 = 8
            attn_map = attn_map.reshape(grid_size, grid_size).detach().cpu()

            # Upsample the attention map to match the image dimensions
            attn_map = F.interpolate(
                attn_map.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
            ).squeeze()

            # Normalize attention map for better contrast
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

            normalized_image = torch.where(image == 0.0, 40.0, image)

            # Display the attention map overlayed on the original image
            ax = axes[layer_idx, head_idx] if num_layers > 1 else axes[head_idx]
            ax.imshow(normalized_image[img_idx, 0].cpu().numpy(), cmap="gray")
            ax.imshow(attn_map.cpu().detach().numpy(), cmap='coolwarm', alpha=0.5)
            ax.axis('off')
            ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
    
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    wandb_img = wandb.Image(img, caption=f"Epoch {epoch+1}, Batch {batch_idx+1}")
    # plt.close(fig)  # Close figure to release memory

    return wandb_img

def visualize_attention(input_image, model):
    # Loop through model layers to capture attention maps from `Attention4D` modules
    attention_maps = []
    for module in model.network:
        if isinstance(module, torch.nn.Sequential):
            for block in module:
                if isinstance(block, AttnFFN) and isinstance(block.token_mixer, Attention4D):
                    attention_maps.append(block.token_mixer.attention_map)  # Retrieve the saved attention map

    # Visualize the attention maps
    # Example visualization for the last attention map captured
    if attention_maps:
        attn_map = attention_maps[-1][0]  # Last layer's attention map, first sample in the batch
        num_heads = attn_map.shape[0]

        # Upsample each head's attention to 128x128 and overlay on the input image
        fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
        for i in range(num_heads):
            ax = axes[i]
            # Reshape and upsample
            grid_size = 16
            spatial_attention_map = attn_map[i].reshape(grid_size, grid_size)  # (e.g., 4x4)
            spatial_attention_map = F.interpolate(
                spatial_attention_map.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
            ).squeeze()

            # Overlay on original image
            ax.imshow(input_image[0, 0].cpu().numpy(), cmap="gray")
            ax.imshow(spatial_attention_map, cmap="jet", alpha=0.5)  # Overlay attention with transparency
            ax.set_title(f"Head {i+1}")
            ax.axis("off")
        plt.show()


def visualize_spatial_heatmaps(output, batch_size=16):
    # Select an example from the batch
    example_index = 0
    example_output = output[example_index]  # Shape: (224, 4, 4)

    # Plot the first few channels as heatmaps
    num_channels_to_plot = 16  # Adjust this number based on how many you want to visualize
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_channels_to_plot:
            channel_data = example_output[i].detach().cpu().numpy()
            ax.imshow(channel_data, cmap="viridis")  # Use a color map like 'viridis' for clarity
            ax.set_title(f"Channel {i+1}")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_cube_sphere(cube_tensor, camera_data=None):
    """
    Visualize a cube-sphere tensor with depth and semantic data for each face.

    Args:
        cube_tensor (torch.Tensor): A tensor of shape (6, 64, 64, 2), where the last dimension is
                                    (depth, semantic_class).
    """
    # Set up the layout of faces for the cube representation
    face_layout = [
        [None, 4, None, None],  # top
        [2, 0, 3, 1],  # middle row (left, front, right, back)
        [None, 5, None, None],  # bottom
    ]

    # Prepare figure for visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    plt.suptitle("Cube Sphere Visualization - Depth")

    vmin, vmax = 0, 20  # Adjust based on your depth data range

    # Display the depth and semantic data separately
    for row in range(3):
        for col in range(4):
            # Find which face corresponds to this grid cell
            face_index = face_layout[row][col]

            if face_index is None:
                # No face for this slot; hide this subplot
                axes[row, col].axis("off")
            else:
                # Display depth data
                depth_img = cube_tensor[face_index, :, :, 0].cpu().numpy()
                img = axes[row, col].imshow(depth_img.T, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[row, col].set_title(f"Face {face_index} - Depth")

                # Add colorbar for this subplot
                fig.colorbar(img, ax=axes[row, col], orientation="vertical", fraction=0.046, pad=0.04)
                axes[row, col].axis("off")

    # Create a new figure for semantic data
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    plt.suptitle("Cube Sphere Visualization - Semantic Classes")

    for row in range(3):
        for col in range(4):
            face_index = face_layout[row][col]
            if face_index is None:
                axes[row, col].axis("off")
            else:
                # Display semantic class data
                semantic_img = cube_tensor[face_index, :, :, 1].cpu().numpy()
                axes[row, col].imshow(semantic_img.T, cmap="tab20")
                axes[row, col].set_title(f"Face {face_index} - Semantic")
                axes[row, col].axis("off")

    if camera_data is not None:
        # Create a new figure for original depth images
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        plt.suptitle("Original Depth Images")
        i = 0
        for cam_name, cam in camera_data.items():
            depth_img = torch.load(cam["file_path"]).cpu().numpy()
            axes[i].imshow(depth_img, cmap="viridis")
            axes[i].set_title(f"{cam_name} - Depth Image")
            axes[i].axis("off")
            i += 1

    plt.show()
