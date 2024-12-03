import os
from torch.utils.data import DataLoader
from data_loader import PKLDatasetStripHistory
import matplotlib.pyplot as plt

VIT_EMBEDDING_SIZE = 128

NUM_EPOCHS = 200
BATCH_SIZE = 32

# Set up the data loaders
training_data_dir = os.path.join(os.path.dirname(__file__), "data/with_history")

training_dataset = PKLDatasetStripHistory(training_data_dir)
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

for image_data, non_image_data, actions in training_data_loader:
    plt.close()

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Display the first image in the first subplot
    axes[0].imshow(image_data[0, 0].cpu(), cmap='viridis')
    axes[0].set_title("Current Image")  # Optional: Set a title for the first image
    axes[0].axis('off')  # Optional: Turn off the axes

    # Display the second image in the second subplot
    axes[1].imshow(image_data[0, 1].cpu(), cmap='viridis')
    axes[1].set_title("Previous Image")  # Optional: Set a title for the second image
    axes[1].axis('off')  # Optional: Turn off the axes

    # Display the plot
    plt.tight_layout()
    plt.show()
