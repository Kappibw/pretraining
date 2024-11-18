import os
import pickle
import torch
from torch.utils.data import Dataset

SPHERE_IMAGE_HEIGHT = 64
SPHERE_IMAGE_SIDES = 6
SPHERE_IMAGE_WIDTH = SPHERE_IMAGE_HEIGHT * 4
IMAGE_START_IDX = 33
INPUT_IMAGE_SIZE = SPHERE_IMAGE_HEIGHT * 2

class PKLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".pkl")]
        self.transform = transform
        self.samples = []
        self.batch_load_size = 5000

        # Load all data from each pickle file once and store it in `self.samples`
        for file in self.data_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
                # Process in batches of `batch_load_size`
                total_samples = data["observations"].shape[0]
                
                for start_idx in range(0, total_samples, self.batch_load_size):
                    end_idx = min(start_idx + self.batch_load_size, total_samples)
                    
                    # Slice batch
                    batch_image_data = data["observations"][start_idx:end_idx, IMAGE_START_IDX:]
                    reshaped_image_data = batch_image_data.view(
                        batch_image_data.shape[0], SPHERE_IMAGE_SIDES, SPHERE_IMAGE_HEIGHT, SPHERE_IMAGE_HEIGHT, 2
                    )
                    
                    # Reshape and process each batch
                    reshaped_image_data = reshaped_image_data[:, :4, :, :, :]
                    reshaped_image_data = reshaped_image_data[:, [0, 3, 1, 2], :, :]
                    
                    depth_channel = reshaped_image_data[..., 0:1].view(
                        batch_image_data.shape[0], SPHERE_IMAGE_WIDTH, SPHERE_IMAGE_HEIGHT, 1
                    )
                    semantics_channel = reshaped_image_data[..., 1:2].view(
                        batch_image_data.shape[0], SPHERE_IMAGE_WIDTH, SPHERE_IMAGE_HEIGHT, 1
                    )
                    
                    # TODO: Use semantics once they aren't junk.
                    batch_image_data = torch.cat((depth_channel, depth_channel, depth_channel), dim=-1)
                    batch_image_data = batch_image_data.view(
                        batch_image_data.shape[0], 2, int(SPHERE_IMAGE_WIDTH / 2), SPHERE_IMAGE_HEIGHT, 3
                    )
                    batch_image_data = batch_image_data.permute(0, 4, 2, 1, 3).reshape(
                        batch_image_data.shape[0], 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE
                    )
                    
                    # Rotate 90 degrees clockwise
                    batch_image_data = torch.rot90(batch_image_data, k=-1, dims=(2, 3))
                    batch_non_image_data = data["observations"][start_idx:end_idx, :IMAGE_START_IDX]
                    batch_actions = data["actions"][start_idx:end_idx]

                    # Add the batch to the samples
                    self.samples.extend(zip(batch_image_data, batch_non_image_data, batch_actions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]