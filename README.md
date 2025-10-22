# Evaluating Domain Gaps of Diffusion Models for Traffic Scenario Generation

This repository contains the code for my Master Thesis titled "Evaluating Domain Gaps of Diffusion Models for the Generation of Traffic Scenarios".

## Project Structure

The repository is organized as follows:

- `real_train.py`: The main script for training the diffusion model on real/synthetic data
- `networks_2.py`: Defines the neural network architectures for the diffusion models.
- `pca_trajectories.py`: Performss Principal Component Analysis (PCA) on trajectories and subsequently calls the PCA-Sinkhorn algo
- `metrics_map.py`: Implements realism metrics for evaluating the quality of the generated trajectories 
- `model_output_xmls.py`: Generates XML files from the model's output, to faciliate comparison with other datasets
- `utils.py`: A collection of utility functions for training the model 
- `map_pre_old.py`: Map and agent-data preprocessing class


![Model architecture and Training procedure](arc.png_url)
