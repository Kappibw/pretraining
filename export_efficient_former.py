import os
import torch
from torch.utils.data import DataLoader
from got_embedder.goal_oriented_transformer import GoT
from data_loader import PKLDatasetStripHistory, PKLDatasetSquare
from efficient_former.efficientformer_models import efficientformerv2_s1
from torch.export import export

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the VIT model
# vit_model = GoT().to(device)
vit_model = efficientformerv2_s1(pretrained=False, resolution=128, distillation = False).to(device)

##### LOADING MODEL #####
vit_model.load_state_dict(torch.load("checkpoints/2024-12-04_11-44-01/vit_model_105.pt"))

#### EXAMPLE DATA FOR TRACE ###
BATCH_SIZE = 8
training_data_dir = os.path.join(os.path.dirname(__file__), "data/training_easy")
training_dataset = PKLDatasetSquare(training_data_dir)
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

for image_data, non_image_data, actions in training_data_loader:
    break

#### SAVING MODEL ######
class ForwardOmnidirWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y):  # `forward` must match the signature you want to export
        return self.model.forward_omnidir(x, y)

wrapped_model = ForwardOmnidirWrapper(vit_model)

# Create a dynamic batch size
batch = torch.export.Dim("batch", min=1, max=2500)
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x": {0: batch}, "y": {0: batch}}

#Convert to TorchScript
goal = non_image_data[:, -3:].to(device)
image_data = image_data.to(device)
vit_model.eval()
traced_model = export(wrapped_model, args=(image_data, goal), dynamic_shapes=dynamic_shapes)

torch.export.save(traced_model, "2024-12-04_11-44-01_vit_105.pt")