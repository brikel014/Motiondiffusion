import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from networks_2 import Denoiser # Assuming this is correct
from utils import plot_trajectories, sample_noise, embed_features # Assuming these are correct
from map_pre_old import MapDataset # Assuming this is correct
import warnings

# Constants and Device
sigma_max = 20
sigma_min = 0.002
rho = 7
N_inference_steps = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_T_optimized(sigma_max, sigma_min, rho, N):
    """
    Generate noise schedule tensor using a vectorized approach.
    """
    t_vals = torch.arange(N, device=device)
    alpha = (sigma_min**(1/rho) - sigma_max**(1/rho)) / (N - 1) if N > 1 else 0
    return (sigma_max**(1/rho) + t_vals * alpha)**rho

def calculate_validation_loss_and_plot(
    model,
    val_xml_dir,
    val_batch_size=32,
    obs_len=10,
    pred_len=20,
    max_radius=100,
    num_polylines=100,
    num_points=10,
    max_agents=32,
    sigma_data=0.5
):
    """
    Calculate validation loss and generate a plot, with optimized operations.
    """
    print(f"Calculating validation loss and generating plot on {val_xml_dir}...")
    model.eval()
    fig_to_return = None
    
    # Dataset and DataLoader creation
    try:
        val_dataset = MapDataset(
            xml_dir=val_xml_dir, obs_len=obs_len, pred_len=pred_len, max_radius=max_radius,
            num_timesteps=obs_len + pred_len, num_polylines=num_polylines, num_points=num_points,
            save_plots=False, max_agents=max_agents
        )
        if len(val_dataset) == 0:
            warnings.warn(f"Validation dataset at {val_xml_dir} is empty.")
            model.train()
            return float('nan'), None
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=8)
    except Exception as e:
        warnings.warn(f"Error creating validation dataset/loader: {e}")
        model.train()
        return float('nan'), None

    total_loss = 0.0
    total_valid_elements = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            try:
                ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask, _, _, _, _, scene_means, scene_stds = batch

                # Send all tensors to device at once
                feature_tensor_dev = feature_tensor.to(device)
                feature_mask_dev = feature_mask.to(device)
                roadgraph_tensor_dev = roadgraph_tensor.to(device)
                roadgraph_mask_dev = roadgraph_mask.to(device)
                
                # Slicing and assigning to variables
                gt_future_part = feature_tensor_dev[:, :, obs_len:, :]
                gt_future_mask = feature_mask_dev[:, :, obs_len:]
                observed_past = feature_tensor_dev[:, :, :obs_len, :]

                T_inference = get_T_optimized(sigma_max, sigma_min, rho, N_inference_steps)
                B, A, T, F = feature_tensor_dev.shape
                future_shape = (B, A, pred_len, F)
                x = torch.randn(future_shape, device=device) * T_inference[0]
                
                full_sequence = torch.cat([observed_past, x], dim=2)
                full_mask = torch.cat([feature_mask_dev[:, :, :obs_len], gt_future_mask], dim=2)

                # Denoising loop
                for i in range(N_inference_steps - 1):
                    ti, ti_next = T_inference[i], T_inference[i + 1]
                    
                    # Vectorized calculations for c_vals
                    ti_squared_plus_sigma_data_squared = ti**2 + sigma_data**2
                    c_skip = sigma_data**2 / ti_squared_plus_sigma_data_squared
                    c_out = ti * sigma_data / torch.sqrt(ti_squared_plus_sigma_data_squared)
                    c_in = 1 / torch.sqrt(ti_squared_plus_sigma_data_squared)
                    c_noise_per_item = (0.25 * torch.log(ti)).expand(B)
                    
                    model_input_sequence = full_sequence.clone()
                    model_input_sequence[:, :, obs_len:, :] *= c_in

                    embedded = embed_features(model_input_sequence, c_noise_per_item, eval=True)
                    model_out = model(embedded, roadgraph_tensor_dev, full_mask, roadgraph_mask_dev)[:, :, obs_len:, :]
                    
                    D_theta = c_skip * x + c_out * model_out
                    di = (1 / ti) * (x - D_theta)
                    x_tilde = x + (ti_next - ti) * di
                    
                    if i < N_inference_steps - 2:
                        ti_next_squared_plus_sigma_data_squared = ti_next**2 + sigma_data**2
                        c_in_next = 1 / torch.sqrt(ti_next_squared_plus_sigma_data_squared)
                        c_noise_next_per_item = (0.25 * torch.log(ti_next)).expand(B)
                        
                        full_sequence_tilde = torch.cat([observed_past, x_tilde], dim=2)
                        model_input_sequence_tilde = full_sequence_tilde.clone()
                        model_input_sequence_tilde[:, :, obs_len:, :] *= c_in_next
                        
                        embedded_tilde = embed_features(model_input_sequence_tilde, c_noise_next_per_item, eval=True)
                        model_out_tilde = model(embedded_tilde, roadgraph_tensor_dev, full_mask, roadgraph_mask_dev)[:, :, obs_len:, :]
                        
                        D_theta_tilde = (sigma_data**2 / ti_next_squared_plus_sigma_data_squared) * x_tilde + \
                                        (ti_next * sigma_data / torch.sqrt(ti_next_squared_plus_sigma_data_squared)) * model_out_tilde
                        
                        d_prime_i = (1 / ti_next) * (x_tilde - D_theta_tilde)
                        x += (ti_next - ti) * 0.5 * (di + d_prime_i)
                    else:
                        x = x_tilde
                        
                # Loss Calculation
                final_sigma = T_inference[-1]
                final_predicted_x0 = (sigma_data**2 / (final_sigma**2 + sigma_data**2)) * x + \
                                      (final_sigma * sigma_data / torch.sqrt(final_sigma**2 + sigma_data**2)) * model_out
                
                valid_mask_loss = gt_future_mask.unsqueeze(-1)
                loss = (final_predicted_x0 - gt_future_part).pow(2) * valid_mask_loss
                total_loss += loss.sum().item()
                total_valid_elements += valid_mask_loss.sum().item()

                # Plotting (Only for the first batch)
                if batch_idx == 0:
                    pred_traj_unscaled = (final_predicted_x0[0].cpu() * scene_stds[0][None, None, :]).numpy()
                    gt_traj_unscaled = (gt_future_part[0].cpu() * scene_stds[0][None, None, :]).numpy()
                    initial_noise_unscaled = (torch.randn_like(gt_traj_unscaled) * T_inference[0] * scene_stds[0][None, None, :]).numpy()
                    map_polylines_unscaled = (roadgraph_tensor[0].cpu() * scene_stds[0][:2]).numpy()

                    fig_to_return = plot_trajectories(
                        map_polylines=map_polylines_unscaled,
                        polyline_masks=roadgraph_mask[0].numpy(),
                        pred_traj=pred_traj_unscaled,
                        gt_traj=gt_traj_unscaled,
                        noisy_traj=initial_noise_unscaled,
                        trajectory_mask=gt_future_mask[0].numpy(),
                        ego_id=ego_ids[0],
                        eval=True,
                        title_suffix="Validation"
                    )

                if (batch_idx + 1) % 20 == 0:
                    print(f"  Validation Batch {batch_idx+1} Processed...")

            except Exception as e:
                warnings.warn(f"Error processing validation batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / (total_valid_elements + 1e-6)
    model.train()
    print(f"Validation finished in {time.time() - start_time:.2f}s. Average Loss: {avg_loss:.4f}")
    return avg_loss, fig_to_return