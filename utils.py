import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import numpy as np

from efficient_former.models import AttnFFN, Attention4D


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
    

    def visualize_goal_attention(input_image, model, epoch, batch_idx):
        """
        Visualizes the attention from the goal (CLS token) to all patches across all heads.

        Args:
            attention_map (torch.Tensor): Stored attention map of shape (B, num_heads, N, N).
            input_image (torch.Tensor): The input image tensor for overlay, shape (B, C, H, W).
        """
        attention_maps = []
        for module in model.network:
            if isinstance(module, torch.nn.Sequential):
                for block in module:
                    if isinstance(block, AttnFFN) and isinstance(block.token_mixer, Attention4D):
                        attention_maps.append(block.token_mixer.attention_map)

        attention_map = attention_maps[-1]  # Last layer's attention map

        # Attention map shape: (B, num_heads, N, N)
        B, num_heads, N, _ = attention_map.shape
        
        # Sum attention across heads for the CLS token (assuming CLS is the first token)
        cls_attention_all_heads = attention_map[:, :, 0, 1:].sum(dim=1)  # Shape: (B, N-1)
        
        # Normalize across the patch dimension for a clearer visualization
        cls_attention_all_heads = (cls_attention_all_heads - cls_attention_all_heads.min()) / (
            cls_attention_all_heads.max() - cls_attention_all_heads.min()
        )
        
        # Reshape to match patch grid (assume patches are square-rootable for simplicity)
        num_patches = int((N - 1) ** 0.5)  # Assuming square grid of patches
        cls_attention_all_heads = cls_attention_all_heads.view(B, num_patches, num_patches).cpu().numpy()

        # Upsample attention to match input image size
        upsampled_attention = F.interpolate(
            torch.tensor(cls_attention_all_heads).unsqueeze(1), size=(input_image.shape[-2], input_image.shape[-1]),
            mode="bilinear", align_corners=False
        ).squeeze()

        # Overlay the attention map on the input image
        fig, ax = plt.subplots()
        ax.imshow(input_image[0].permute(1, 2, 0).cpu().numpy(), cmap="gray")  # Adjust for RGB if needed
        ax.imshow(upsampled_attention[0], cmap="coolwarm", alpha=0.3)
        plt.colorbar(ax.imshow(upsampled_attention[0], cmap="coolwarm", alpha=0.5))
        ax.set_title("Attention from Goal (CLS) Token to All Patches Across Heads")
        cbar = plt.colorbar(ax.imshow(total_attention, cmap="coolwarm", alpha=0.5))

        plt.show()

        # Capture the figure as an image to log to wandb
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        wandb_img = wandb.Image(img, caption=f"Epoch {epoch+1}, Batch {batch_idx+1}")
        plt.close(fig)  # Close figure to release memory

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
