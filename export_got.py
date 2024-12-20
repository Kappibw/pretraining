import os
import torch
from torch.utils.data import DataLoader
from got_embedder.goal_oriented_transformer import GoT
from data_loader import PKLDatasetStrip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the VIT model
vit_model = GoT().to(device)

##### LOADING MODEL #####
vit_model.load_state_dict(torch.load("checkpoints/2024-12-02_20-47-16/vit_model_151.pt"))

#### EXAMPLE DATA FOR TRACE ###
BATCH_SIZE = 8
training_data_dir = os.path.join(os.path.dirname(__file__), "data/training")
training_dataset = PKLDatasetStrip(training_data_dir)
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

for image_data, non_image_data, actions in training_data_loader:
    break

#### SAVING MODEL ######

#Convert to TorchScript
image_data = image_data[:, :2]
goal = non_image_data[:, -3:].to(device)
image_data = image_data.to(device)
vit_model.eval()
traced_model = torch.jit.trace(vit_model, (image_data, goal))

# Optimize for inference
optimized_model = torch.jit.optimize_for_inference(traced_model)

# Save the optimized model
optimized_model.save("2024-12-02_20-47-16_vit_151.pt")