# Omnidirectional Navigation Pretraining

This repo contains the supervised learning component of the omnidirectional navigation project.

There are 3 different ViT setups that can be trained on the same training data from IsaacLab, in an end-to-end training that optimizes generated actions from the Actor network in the rsl_rl ActorCritic:

- `efficient_former_pretraining.py`: Trains using efficient-former as the ViT.
- `got_pretraining.py`: Trains using the GoalOrientedTransformer Policy from [Oscar Huang](https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/catkin_ws/src/gtrl/scripts/SAC/GoalFormer.py)
- `omnidir_vit_pretraining.py`: Trains using the custom ViT written for the omnidir_nav project, based on GoT above.