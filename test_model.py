import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from rsl_rl.modules import ActorCritic
from torch.utils.data import DataLoader, Dataset
from efficient_former.models import efficientformerv2_s1
import utils
import wandb

SPHERE_IMAGE_HEIGHT = 64
SPHERE_IMAGE_SIDES = 6
SPHERE_IMAGE_WIDTH = SPHERE_IMAGE_HEIGHT * 4
IMAGE_START_IDX = 33
INPUT_IMAGE_SIZE = SPHERE_IMAGE_HEIGHT * 2

# Initialize wandb
wandb.init(project="pretraining_test")  # Replace with your wandb project name

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
                reshaped_image_data = reshaped_image_data[:, :4, :, :, :]

                depth_channel = reshaped_image_data[..., 0:1].view(
                    image_data.shape[0], SPHERE_IMAGE_WIDTH, SPHERE_IMAGE_HEIGHT, 1
                )
                semantics_channel = reshaped_image_data[..., 1:2].view(
                    image_data.shape[0], SPHERE_IMAGE_WIDTH, SPHERE_IMAGE_HEIGHT, 1
                )

                image_data = torch.cat((depth_channel, depth_channel, semantics_channel), dim=-1)
                image_data = image_data.view(
                    image_data.shape[0], 2, int(SPHERE_IMAGE_WIDTH / 2), SPHERE_IMAGE_HEIGHT, 3
                )
                image_data = image_data.permute(0, 4, 2, 1, 3).reshape(
                    image_data.shape[0], 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE
                )
                non_image_data = data["observations"][:, :IMAGE_START_IDX]
                actions = data["actions"]

                self.samples.extend(zip(image_data, non_image_data, actions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(os.path.dirname(__file__), "data")

transform = None
dataset = PKLDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

vit_model = efficientformerv2_s1(pretrained=False, resolution=128).to(device)
actor_policy_cfg = {
    "init_noise_std": 1.0,
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
}

EMBEDDING_SIZE = 3584
for image_data, non_image_data, actions in data_loader:
    break

num_actor_obs = EMBEDDING_SIZE + non_image_data.shape[1]
num_critic_obs = num_actor_obs
num_actions = actions.shape[1]
actor_critic = ActorCritic(num_actor_obs=num_actor_obs, num_critic_obs=num_critic_obs, num_actions=num_actions, **actor_policy_cfg).to(device)

# Optimizer and loss
optimizer = Adam(list(vit_model.parameters()) + list(actor_critic.actor.parameters()), lr=1e-4)
loss_fn = nn.MSELoss()

# Log the configuration to wandb
wandb.config.update({
    "num_epochs": 10,
    "batch_size": 8,
    "learning_rate": 1e-4,
})

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (image_data, non_image_data, actions) in enumerate(data_loader):
        image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
        goal = non_image_data[:, -3:]
        embeddings = vit_model.forward_omnidir(image_data, goal)
        embeddings = embeddings[-1] if vit_model.fork_feat else embeddings

        combined_input = torch.cat((embeddings, non_image_data), dim=-1)
        actions_pred = actor_critic.actor(combined_input)

        loss = loss_fn(actions_pred, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item()}")
            wandb.log({"epoch": epoch+1, "batch": batch_idx+1, "loss": loss.item()})

print("Training completed.")
wandb.finish()
