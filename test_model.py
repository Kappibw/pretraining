import os
import pickle
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torchvision import transforms
from rsl_rl.modules import ActorCritic
from torch.utils.data import DataLoader, Dataset
from efficient_former.models import efficientformerv2_s1
import utils

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

        # Load all data from each pickle file once and store it in `self.samples`
        for file in self.data_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
                image_data = data["observations"][:, IMAGE_START_IDX:]
                reshaped_image_data = image_data.view(
                    image_data.shape[0], SPHERE_IMAGE_SIDES, SPHERE_IMAGE_HEIGHT, SPHERE_IMAGE_HEIGHT, 2
                )
                # Keep only front, back, left, and right sides, and remove the top and bottom sides
                reshaped_image_data = reshaped_image_data[:, :4, :, :, :]

                depth_channel = reshaped_image_data[..., 0:1].view(
                    image_data.shape[0], SPHERE_IMAGE_WIDTH, SPHERE_IMAGE_HEIGHT, 1
                )
                semantics_channel = reshaped_image_data[..., 1:2].view(
                    image_data.shape[0], SPHERE_IMAGE_WIDTH, SPHERE_IMAGE_HEIGHT, 1
                )

                # Stack into 3 channels, which the vit expects
                image_data = torch.cat((depth_channel, depth_channel, semantics_channel), dim=-1)

                image_data = image_data.view(
                    image_data.shape[0], 2, int(SPHERE_IMAGE_WIDTH / 2), SPHERE_IMAGE_HEIGHT, 3
                )
                # Step 3: Permute and reshape to [num_samples, 128, 128, 3]
                image_data = image_data.permute(0, 4, 2, 1, 3).reshape(
                    image_data.shape[0], 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE
                )
                non_image_data = data["observations"][:, :IMAGE_START_IDX]
                actions = data["actions"]

                # Store each sample as a tuple
                self.samples.extend(zip(image_data, non_image_data, actions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return a single sample (image_data, non_image_data, action)
        return self.samples[idx]


def extract_goal(obs):
    return obs["policy"][:, 30:33]  # Shape: (num_envs, 3)


# Define Goal Embedding Layer to match EfficientFormer embedding dimension
class GoalEmbedding(nn.Module):
    def __init__(self, input_dim=4, embed_dim=384):
        super(GoalEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, goal):
        return self.fc(goal)  # Shape: (batch_size, embed_dim)


# Load data from the directory two above this one
data_dir = os.path.join(os.path.dirname(__file__), "data")

# Example transformation to resize and normalize images
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Custom image size
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
transform = None  # No transformations
dataset = PKLDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize models
vit_model = efficientformerv2_s1(pretrained=False, resolution=128)  # Training from scratch

actor_policy_cfg = {
    "init_noise_std": 1.0,
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
}

# Load a single batch from the DataLoader to inspect the observation sizes
for image_data, non_image_data, actions in data_loader:
    print("Image Data Shape:", image_data.shape)  # Should show (batch_size, 3, 128, 128) 
    print("Non-Image Data Shape:", non_image_data.shape)  # Should show (batch_size, IMAGE_START_IDX)
    print("Actions Shape:", actions.shape)  # Should show (batch_size, action_dim)
    break  # Only need one batch to inspect the shapes

# Initialize the ActorCritic model using the inspected sizes
num_actor_obs = image_data.shape[1] * image_data.shape[2] * image_data.shape[3] + non_image_data.shape[1]
num_critic_obs = num_actor_obs
num_actions = actions.shape[1]  # Assuming the action dimension is consistent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_critic = ActorCritic(num_actor_obs=num_actor_obs, num_critic_obs=num_critic_obs, num_actions=num_actions, **actor_policy_cfg).to(device)


# Optimizer and loss
optimizer = Adam(list(vit_model.parameters()) + list(actor_critic.actor.parameters()), lr=1e-4)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (image_data, non_image_data, actions) in enumerate(data_loader):
        # Forward pass through EfficientFormer to get embeddings
        goal = non_image_data[:,-3:]
        embeddings = vit_model.forward_omnidir(image_data, goal)
        embeddings = embeddings[-1] if vit_model.fork_feat else embeddings  # Use final layer embeddings

        # Combine embeddings with other observation data
        combined_input = torch.cat((embeddings, non_image_data), dim=-1)

        # Forward pass through the actor network to predict actions
        actions_pred = actor_critic.actor(combined_input)

        # Compute loss and backpropagate
        loss = loss_fn(actions_pred, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item()}")

print("Training completed.")
