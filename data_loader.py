import os
import pickle
import torch
from torch.utils.data import Dataset

SPHERE_IMAGE_HEIGHT = 64
SPHERE_IMAGE_SIDES = 6
SPHERE_IMAGE_WIDTH = SPHERE_IMAGE_HEIGHT * 4
IMAGE_START_IDX = 33
INPUT_IMAGE_SIZE = SPHERE_IMAGE_HEIGHT * 2

class PKLDatasetSquare(Dataset):
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

                    # Step 2: Arrange the images in the desired order
                    # Combine the images in the top row and bottom row
                    top_row = torch.cat((front, right), dim=2)  # Shape: [batch, 64, 128]
                    bottom_row = torch.cat((back, left), dim=2)  # Shape: [batch, 64, 128]

                    del front, left, back, right

                    # Step 3: Stack the rows vertically
                    image_data = torch.cat((top_row, bottom_row), dim=1)

                    # Normalize the image data between 0 and 1
                    image_data[image_data > 40.0] = 40.0
                    image_data = image_data / 40.0

                    # Step 4: Overwrite values of 0.0 with -1.0
                    # TODO(kappi): Remove when the data is collected again.
                    image_data[image_data == 0.0] = -1.0

                    # Add a third channel
                    # TODO(Kappi): Just train with 2
                    image_data = torch.cat((image_data, image_data[...,0:1]), dim=-1)
                    # Make the image data 3 channels first.
                    batch_image_data = image_data.permute(0, 3, 1, 2)

                    batch_non_image_data = data["observations"][start_idx:end_idx, :IMAGE_START_IDX]
                    batch_actions = data["actions"][start_idx:end_idx]

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