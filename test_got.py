import datetime
import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from rsl_rl.modules import ActorCritic
from torch.utils.data import DataLoader, Dataset
from got_nav.catkin_ws.src.gtrl.scripts.SAC.got_sac_network import GoTPolicy
import utils
import wandb

SPHERE_IMAGE_HEIGHT = 64
SPHERE_IMAGE_SIDES = 6
SPHERE_IMAGE_WIDTH = SPHERE_IMAGE_HEIGHT * 4
IMAGE_START_IDX = 33
INPUT_IMAGE_SIZE = SPHERE_IMAGE_HEIGHT * 2
VIT_EMBEDDING_SIZE = 128

NUM_EPOCHS = 200
BATCH_SIZE = 8

def save(vit_model, actor_critic, save_dir, idx):
    vit_model_path = os.path.join(save_dir, f"vit_model_{idx}.pt")
    actor_critic_path = os.path.join(save_dir, f"actor_critic_{idx}.pt")

    # Save the models' state dictionaries
    torch.save(vit_model.state_dict(), vit_model_path)
    torch.save(actor_critic.state_dict(), actor_critic_path)

    print(f"Model idx {idx} saved to {save_dir}")

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

def evaluate(vit_model, actor_critic, data_loader):
    vit_model.eval()
    actor_critic.eval()
    
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for image_data, non_image_data, actions in data_loader:
            image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
            goal = non_image_data[:, -3:]
            embeddings = vit_model.sample_omnidir([image_data, goal])
            combined_input = torch.cat((embeddings, non_image_data), dim=-1)
            actions_pred = actor_critic.actor(combined_input)

            loss = torch.sqrt(torch.pow(actions_pred - actions, 2).mean())
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

# Initialize wandb
wandb.init(project="got_test")

# Create a directory to save the models
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(os.path.dirname(__file__), "checkpoints", time)
os.makedirs(save_dir, exist_ok=True)

# Set up the data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_dir = os.path.join(os.path.dirname(__file__), "data/training")
eval_data_dir = os.path.join(os.path.dirname(__file__), "data/eval")

training_dataset = PKLDataset(training_data_dir)
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataset = PKLDataset(eval_data_dir)
eval_data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate the actor-critic model
actor_policy_cfg = {
    "init_noise_std": 1.0,
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
}
for image_data, non_image_data, actions in training_data_loader:
    break
num_actor_obs = VIT_EMBEDDING_SIZE + non_image_data.shape[1]
num_critic_obs = num_actor_obs
num_actions = actions.shape[1]
actor_critic = ActorCritic(num_actor_obs=num_actor_obs, num_critic_obs=num_critic_obs, num_actions=num_actions, **actor_policy_cfg).to(device)

# Instantiate the VIT model
# vit_model = efficientformerv2_s1(pretrained=False, resolution=128).to(device)
goal_size = 3
block = 2
head = 4 # TODO(kappi): idk what these are
vit_model = GoTPolicy(num_actions, goal_size, block, head).to(device)

# Optimizer and loss
optimizer = Adam(list(vit_model.parameters()) + list(actor_critic.actor.parameters()), lr=1e-4)

# Log the configuration to wandb
wandb.config.update({
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": 1e-4,
})

# Load weights
vit_model.load_state_dict(torch.load("checkpoints/2024-11-12_22-38-22/vit_model_199.pt"))

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (image_data, non_image_data, actions) in enumerate(training_data_loader):
        image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
        goal = non_image_data[:, -3:]

        # Dist  = torch.minimum(goal[:,0]/15, torch.tensor(1.0))
        # heading = goal[:, 1] / np.pi
        # goal_normalized = torch.stack((Dist, heading), dim=1)
        # predict, log_prob, mean = vit_model.sample([image_data, goal])
        embeddings, attention_map = vit_model.sample_omnidir([image_data, goal], return_attention=True)
        combined_input = torch.cat((embeddings, non_image_data), dim=-1)
        actions_pred = actor_critic.actor(combined_input)

        optimizer.zero_grad()
        loss = torch.sqrt(torch.pow(actions_pred - actions, 2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vit_model.parameters(), 10) # TODO(kappi): what is this
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(training_data_loader)}], Loss: {loss.item()}")
            if batch_idx % 1000 == 0:
                image = utils.log_got_attention_map(image_data, attention_map, epoch, batch_idx)
                wandb.log({"epoch": epoch+1, "batch": batch_idx+1, "loss": loss.item(), "attention_map": image})
            else:
                wandb.log({"epoch": epoch+1, "batch": batch_idx+1, "loss": loss.item()})
    save(vit_model, actor_critic, save_dir, epoch)

    # Run evaluation
    val_loss = evaluate(vit_model, actor_critic, eval_data_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss}")
    wandb.log({"epoch": epoch+1, "val_loss": val_loss})

print("Training completed.")
wandb.finish()