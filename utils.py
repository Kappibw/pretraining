import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Need to run in forward pass
def visualize_attention(input_image):
    # Loop through model layers to capture attention maps from `Attention4D` modules
    attention_maps = []
    for module in self.network:
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
            grid_size=16
            spatial_attention_map = attn_map[i].reshape(grid_size, grid_size)  # (e.g., 4x4)
            spatial_attention_map = F.interpolate(spatial_attention_map.unsqueeze(0).unsqueeze(0), 
                                                size=(128, 128), mode='bilinear', align_corners=False).squeeze()

            # Overlay on original image
            ax.imshow(input_image[0].permute(1, 2, 0).cpu().numpy())  # Assuming input_image is (B, C, H, W)
            ax.imshow(spatial_attention_map, cmap='jet', alpha=0.5)  # Overlay attention with transparency
            ax.set_title(f'Head {i+1}')
            ax.axis('off')
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
