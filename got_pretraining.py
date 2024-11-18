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
from data_loader import PKLDataset
import wandb


VIT_EMBEDDING_SIZE = 128

NUM_EPOCHS = 200
BATCH_SIZE = 8


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
wandb.init(project="got_pretraining")

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
head = 4 # TODO(kappi): see what params they used in the paper
vit_model = GoTPolicy(num_actions, goal_size, block, head).to(device)

# Optimizer and loss
optimizer = Adam(list(vit_model.parameters()) + list(actor_critic.actor.parameters()), lr=1e-4)

# Log the configuration to wandb
wandb.config.update({
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": 1e-4,
})

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
    utils.save(vit_model, actor_critic, save_dir, epoch)

    # Run evaluation
    val_loss = evaluate(vit_model, actor_critic, eval_data_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss}")
    wandb.log({"epoch": epoch+1, "val_loss": val_loss})

print("Training completed.")
wandb.finish()