import datetime
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from rsl_rl.modules import ActorCritic
from torch.utils.data import DataLoader
from efficient_former.efficientformer_models import efficientformerv2_s1
import utils
from data_loader import PKLDataset
import wandb

VIT_EMBEDDING_SIZE = 3584

NUM_EPOCHS = 10
BATCH_SIZE = 8

# Create a directory to save the models
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(os.path.dirname(__file__), "checkpoints", time)
os.makedirs(save_dir, exist_ok=True)

# Initialize wandb
wandb.init(project="omnidir_nav_pretraining", id=f"efficient_former_{time}")

# Set up the data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_dir = os.path.join(os.path.dirname(__file__), "data/training")
eval_data_dir = os.path.join(os.path.dirname(__file__), "data/eval")

training_dataset = PKLDataset(training_data_dir)
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataset = PKLDataset(eval_data_dir)
eval_data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate the VIT model
vit_model = efficientformerv2_s1(pretrained=False, resolution=128).to(device)

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

# Optimizer and loss
optimizer = Adam(list(vit_model.parameters()) + list(actor_critic.actor.parameters()), lr=1e-4)
loss_fn = nn.MSELoss()

# Log the configuration to wandb
wandb.config.update({
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": 1e-4,
})

def evaluate(vit_model, actor_critic, data_loader, loss_fn):
    vit_model.eval()
    actor_critic.eval()
    
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for image_data, non_image_data, actions in data_loader:
            image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
            goal = non_image_data[:, -3:]
            embeddings = vit_model.forward_omnidir(image_data, goal)

            embeddings = embeddings[-1] if vit_model.fork_feat else embeddings
            combined_input = torch.cat((non_image_data, embeddings), dim=-1)
            actions_pred = actor_critic.actor(combined_input)

            loss = loss_fn(actions_pred, actions)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

# Training loop
for epoch in range(NUM_EPOCHS):
    vit_model.train()
    actor_critic.train()
    for batch_idx, (image_data, non_image_data, actions) in enumerate(training_data_loader):
        image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
        goal = non_image_data[:, -3:]
        embeddings = vit_model.forward_omnidir(image_data, goal)

        embeddings = embeddings[-1] if vit_model.fork_feat else embeddings

        combined_input = torch.cat((non_image_data, embeddings), dim=-1)
        actions_pred = actor_critic.actor(combined_input)

        loss = loss_fn(actions_pred, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(training_data_loader)}], Loss: {loss.item()}")
            if batch_idx % 1000 == 0:
                image = utils.log_attention_mask(image_data, vit_model, epoch, batch_idx)
                wandb.log({"epoch": epoch+1, "batch": batch_idx+1, "loss": loss.item(), "attention_map": image})
            else:
                wandb.log({"epoch": epoch+1, "batch": batch_idx+1, "loss": loss.item()})
    utils.save(vit_model, actor_critic, save_dir, epoch)

    # Run evaluation
    val_loss = evaluate(vit_model, actor_critic, eval_data_loader, loss_fn)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss}")
    wandb.log({"epoch": epoch+1, "val_loss": val_loss})

print("Training completed.")
wandb.finish()