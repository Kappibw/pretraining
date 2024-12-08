import os
import pickle
import torch
import utils
from torch.utils.data import Dataset

SPHERE_IMAGE_HEIGHT = 48
SPHERE_IMAGE_SIDES = 6
SPHERE_IMAGE_WIDTH = SPHERE_IMAGE_HEIGHT * 4
IMAGE_START_IDX = 33
INPUT_IMAGE_SIZE = SPHERE_IMAGE_HEIGHT * 2

class PKLDatasetSquare(Dataset):
    def __init__(self, data_dir, include_paths=False, transform=None):
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".pkl")]
        self.transform = transform
        self.samples = []
        self.batch_load_size = 5000
        weights = torch.linspace(0, 1, steps=5).unsqueeze(0) 
        self.include_paths = include_paths

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

                    front = reshaped_image_data[:, 0]  # Shape: [batch, 64, 64]
                    back = reshaped_image_data[:, 1]  # Shape: [batch, 64, 64]
                    left = reshaped_image_data[:, 2]  # Shape: [batch, 64, 64]
                    right = reshaped_image_data[:, 3]  # Shape: [batch, 64, 64]

                    # Step 2: Arrange the images in the desired order
                    # Combine the images in the top row and bottom row
                    top_row = torch.cat((front, right), dim=2)  # Shape: [batch, 64, 128]
                    bottom_row = torch.cat((back, left), dim=2)  # Shape: [batch, 64, 128]

                    del front, left, back, right

                    # Step 3: Stack the rows vertically
                    image_data = torch.cat((top_row, bottom_row), dim=1)

                    # Add a third channel
                    # TODO(Kappi): Just train with 2
                    image_data = torch.cat((image_data, image_data[...,0:1]), dim=-1)
                    # Make the image data 3 channels first.
                    batch_image_data = image_data.permute(0, 3, 1, 2)

                    batch_non_image_data = data["observations"][start_idx:end_idx, :IMAGE_START_IDX]
                    batch_actions = data["actions"][start_idx:end_idx]

                    # Extract paths
                    if self.include_paths:
                        paths = data["waypoints"][start_idx:end_idx]
                        max_indices = torch.tensor([torch.unique(path, dim=0).shape[0] for path in paths])
                        end_indices = torch.argmin(torch.norm(paths - batch_non_image_data[:, -3:].unsqueeze(1), dim=2), dim=1)
                        end_distance = torch.min(torch.norm(paths - batch_non_image_data[:, -3:].unsqueeze(1), dim=2), dim=1)

                        end_indices = torch.min(end_indices, max_indices)
                        start_indices = torch.argmin(torch.norm(paths, dim=2), dim=1)
                        start_distance = torch.min(torch.norm(paths, dim=2), dim=1)
                        
                        indices = (weights * (end_indices - start_indices).unsqueeze(1)).round().long() + start_indices.unsqueeze(1)
                        indices = indices.unsqueeze(-1).expand(-1, -1, 3)
                        sampled_waypoints = torch.gather(paths, dim=1, index=indices)

                        # for x in range(0, batch_image_data.shape[0]):
                        #     print(x)
                        #     utils.visualize_training_data_torch(batch_non_image_data, batch_image_data, batch_actions, sampled_waypoints, sampled_waypoints, x)

                        self.samples.extend(zip(batch_image_data, batch_non_image_data, batch_actions, sampled_waypoints))

                    else:
                        # Add the batch to the samples
                        self.samples.extend(zip(batch_image_data, batch_non_image_data, batch_actions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class PKLDatasetStrip(Dataset):
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

                    front = reshaped_image_data[:, 0]  # Shape: [batch, 64, 64]
                    back = reshaped_image_data[:, 1]  # Shape: [batch, 64, 64]
                    left = reshaped_image_data[:, 2]  # Shape: [batch, 64, 64]
                    right = reshaped_image_data[:, 3]  # Shape: [batch, 64, 64]

                    # Step 2: Place front in the middle, with left and right to either side, and back split in half
                    # on either side of them.
                    back_right = back[:, :, :32]
                    back_left = back[:, :, 32:]
                    image_data = torch.cat((back_left, left, front, right, back_right), dim=2)

                    del front, left, back, right, back_left, back_right

                    batch_image_data = image_data.permute(0, 3, 1, 2)

                    batch_non_image_data = data["observations"][start_idx:end_idx, :IMAGE_START_IDX]
                    batch_actions = data["actions"][start_idx:end_idx]

                    # Add the batch to the samples
                    self.samples.extend(zip(batch_image_data, batch_non_image_data, batch_actions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    


class PKLDatasetStripHistory(Dataset):
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
                    print(f"Loading batch {start_idx} to {min(total_samples, start_idx + self.batch_load_size)} of file {file}")
                    end_idx = min(start_idx + self.batch_load_size, total_samples)
                    
                    # Slice batch
                    batch_image_data = data["observations"][start_idx:end_idx, IMAGE_START_IDX:]
                    image_data = self._get_image_data(batch_image_data)
                    prior_image_data = data["prev_observations"][start_idx:end_idx, IMAGE_START_IDX:]
                    prior_image_data = self._get_image_data(prior_image_data)

                    # Make the second channel be the previous image
                    image_data = torch.cat((image_data[...,0:1], prior_image_data[...,0:1]), dim=-1)

                    batch_image_data = image_data.permute(0, 3, 1, 2)

                    batch_non_image_data = data["observations"][start_idx:end_idx, :IMAGE_START_IDX]
                    batch_actions = data["actions"][start_idx:end_idx]

                    # Add the batch to the samples
                    self.samples.extend(zip(batch_image_data, batch_non_image_data, batch_actions))


    def _get_image_data(self, image_data):
        reshaped_image_data = image_data.view(
            image_data.shape[0], SPHERE_IMAGE_SIDES, SPHERE_IMAGE_HEIGHT, SPHERE_IMAGE_HEIGHT, 2
        )
        
        # Reshape and process each batch
        reshaped_image_data = reshaped_image_data[:, :4, :, :, :]

        front = reshaped_image_data[:, 0]  # Shape: [batch, 64, 64]
        back = reshaped_image_data[:, 1]  # Shape: [batch, 64, 64]
        left = reshaped_image_data[:, 2]  # Shape: [batch, 64, 64]
        right = reshaped_image_data[:, 3]  # Shape: [batch, 64, 64]

        # Step 2: Place front in the middle, with left and right to either side, and back split in half
        # on either side of them.
        back_right = back[:, :, :32]
        back_left = back[:, :, 32:]
        image_data = torch.cat((back_left, left, front, right, back_right), dim=2)

        return image_data


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]