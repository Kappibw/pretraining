import datetime
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from rsl_rl.modules import ActorCritic
from torch.utils.data import DataLoader
from got_embedder.goal_oriented_transformer import GoT
import utils
from data_loader import PKLDatasetStrip, PKLDatasetStripHistory
import wandb
import torch.nn.functional as F


VIT_EMBEDDING_SIZE = 128

NUM_EPOCHS = 200
BATCH_SIZE = 32

# Create a directory to save the models
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(os.path.dirname(__file__), "checkpoints", time)
os.makedirs(save_dir, exist_ok=True)

# Initialize wandb
wandb.init(project="omnidir_nav_pretraining", id=f"omnidir_vit_{time}")

# Set up the data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_dir = os.path.join(os.path.dirname(__file__), "data/training")
eval_data_dir = os.path.join(os.path.dirname(__file__), "data/eval")

training_dataset = PKLDatasetStrip(training_data_dir)
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataset = PKLDatasetStrip(eval_data_dir)
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
actor_critic = ActorCritic(
    num_actor_obs=num_actor_obs, num_critic_obs=num_critic_obs, num_actions=num_actions, **actor_policy_cfg
).to(device)


# Define goal type enums
class GoalType:
    POLAR = 1
    EUCLIDEAN = 2
    POLAR_NORMALIZED = 3


GOAL_TYPE = GoalType.EUCLIDEAN
PATCH_SIZE = 16

GOAL_NORMALIZATION_CONSTANT = 10.0  # Meters

# Instantiate the VIT model
vit_model = GoT(
    goal_size=2 if (GOAL_TYPE == GoalType.POLAR or GOAL_TYPE == GoalType.POLAR_NORMALIZED) else 3,
    patch_size=(PATCH_SIZE, PATCH_SIZE),
    dropout=0.1,
    channels=2,
    emb_dropout=0.1,
    mean_pool=False,
).to(device)

# Optimizer and loss
optimizer = Adam(list(vit_model.parameters()) + list(actor_critic.actor.parameters()), lr=1e-4)

# Log the configuration to wandb
wandb.config.update(
    {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-4,
    }
)

def get_goal(goal_type, goal):
    if goal_type == GoalType.POLAR:
        return utils.cartesian_to_polar(goal)
    elif goal_type == GoalType.POLAR_NORMALIZED:
        goal = utils.cartesian_to_polar(goal)
        goal[:, 0] = torch.min(goal[:, 0] / GOAL_NORMALIZATION_CONSTANT, torch.tensor([1.0]).to(device))
        return goal
    return goal


def evaluate(vit_model, actor_critic, data_loader):
    vit_model.eval()
    actor_critic.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for image_data, non_image_data, actions in data_loader:
            image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
            goal = non_image_data[:, -3:]
            goal = get_goal(GOAL_TYPE, goal)
            embeddings = vit_model(image_data, goal)
            combined_input = torch.cat((non_image_data, embeddings), dim=-1)
            actions_pred = actor_critic.actor(combined_input)

            loss = torch.sqrt(torch.pow(actions_pred - actions, 2).mean())
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


min_val_loss = float("inf")
# Training loop
for epoch in range(NUM_EPOCHS):
    vit_model.train()
    actor_critic.train()
    for batch_idx, (image_data, non_image_data, actions) in enumerate(training_data_loader):
        image_data, non_image_data, actions = image_data.to(device), non_image_data.to(device), actions.to(device)
        goal = non_image_data[:, -3:]
        goal = get_goal(GOAL_TYPE, goal)

        embeddings = vit_model(image_data, goal)
        combined_input = torch.cat((non_image_data, embeddings), dim=-1)
        actions_pred = actor_critic.actor(combined_input)

        optimizer.zero_grad()
        loss = F.mse_loss(actions_pred, actions)
        # loss = torch.sqrt(torch.pow(actions_pred - actions, 2).mean()) # change to torch.mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vit_model.parameters(), 10)  # TODO(kappi):see if this helps
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(training_data_loader)}], Loss: {loss.item()}"
            )
            wandb.log({"epoch": epoch + 1, "batch": batch_idx + 1, "loss": loss.item()})

    # Run evaluation
    val_loss = evaluate(vit_model, actor_critic, eval_data_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss}")
    wandb.log({"epoch": epoch + 1, "val_loss": val_loss})

    # Save the model if the validation loss is the lowest so far
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        utils.save(vit_model, actor_critic, save_dir, epoch)

print("Training completed.")
wandb.finish()
