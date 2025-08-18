import torch
from torch.utils.data import DataLoader
from networks_2 import Denoiser
# Make sure embed_features is imported correctly
from utils import plot_trajectories, sample_noise, embed_features
from map_pre_old import MapDataset
import matplotlib.pyplot as plt
import numpy as np
import time


sigma_max = 20 # Maximum noise level
sigma_min = 0.002 # Minimum noise level
rho = 7 # Controls the noise schedule
N_inference_steps = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_T(sigma_max, sigma_min, rho, N):
    T = []
    for i in range(N):
        divisor = N - 1 if N > 1 else 1
        t_val = (sigma_max**(1/rho) + (i/divisor) * ((sigma_min**(1/rho)) - (sigma_max**(1/rho))))**rho
        T.append(t_val)
    return torch.tensor(T, device=device)


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
    Calculate the average validation loss over the validation dataset for one epoch
    and generate a plot for the first validation batch.
    """
    print(f"Calculating validation loss and generating plot on {val_xml_dir}...")
    model.eval()
    fig_to_return = None

    # ... (Dataset and DataLoader creation - unchanged) ...
    try:
        val_dataset = MapDataset(
            xml_dir=val_xml_dir, obs_len=obs_len, pred_len=pred_len, max_radius=max_radius,
            num_timesteps=obs_len + pred_len, num_polylines=num_polylines, num_points=num_points,
            save_plots=False, max_agents=max_agents
        )
        if len(val_dataset) == 0:
            print(f"Warning: Validation dataset at {val_xml_dir} is empty. Returning NaN, None.")
            model.train(); return float('nan'), None
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=8)
    except Exception as e:
        print(f"Error creating validation dataset/loader: {e}. Returning NaN, None.")
        model.train(); return float('nan'), None


    total_loss = 0.0
    num_batches = 0
    total_valid_elements = 0

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            try:
                # --- Data Preparation ---
                ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask, _, _, _, _, scene_means, scene_stds = batch

                feature_tensor_dev = feature_tensor.to(device)
                feature_mask_dev = feature_mask.to(device)
                roadgraph_tensor_dev = roadgraph_tensor.to(device)
                roadgraph_mask_dev = roadgraph_mask.to(device)
                scene_stds_dev = scene_stds.to(device)

                gt_future_part = feature_tensor_dev[:, :, obs_len:, :].clone()
                gt_future_mask = feature_mask_dev[:, :, obs_len:].clone()
                observed_past = feature_tensor_dev[:, :, :obs_len, :].clone()

                # --- Inference Denoising Process ---
                T_inference = get_T(sigma_max, sigma_min, rho, N_inference_steps)
                B, A, _, F = feature_tensor_dev.shape
                future_shape = (B, A, pred_len, F)
                x = torch.randn(future_shape, device=device) * T_inference[0]

                # Denoising loop
                for i in range(N_inference_steps - 1):
                    ti = T_inference[i]
                    ti_next = T_inference[i + 1]

                    c_skip = sigma_data**2 / (ti**2 + sigma_data**2)
                    c_out = ti * sigma_data / torch.sqrt(sigma_data**2 + ti**2)
                    c_in = 1 / torch.sqrt(ti**2 + sigma_data**2)
                    # *** FIX: Prepare c_noise correctly for embed_features ***
                    # Assuming embed_features takes noise level per batch item [B] or [B, 1]
                    c_noise_per_item = (0.25 * torch.log(ti)).expand(B) # Shape [B]

                    full_sequence = torch.cat([observed_past, x], dim=2)
                    full_mask = torch.cat([feature_mask_dev[:, :, :obs_len], gt_future_mask], dim=2)

                    model_input_sequence = full_sequence.clone()
                    model_input_sequence[:, :, obs_len:, :] = c_in * full_sequence[:, :, obs_len:, :]

                    # *** FIX: Pass corrected c_noise ***
                    embedded = embed_features(model_input_sequence, c_noise_per_item, eval=True)
                    model_out = model(embedded, roadgraph_tensor_dev, full_mask, roadgraph_mask_dev)[:, :, obs_len:, :]

                    D_theta = c_skip * x + c_out * model_out

                    di = (1 / ti) * (x - D_theta)
                    x_tilde = x + (ti_next - ti) * di

                    if i < N_inference_steps - 2: # Correction step
                        c_skip_next = sigma_data**2 / (ti_next**2 + sigma_data**2)
                        c_out_next = ti_next * sigma_data / torch.sqrt(sigma_data**2 + ti_next**2)
                        c_in_next = 1 / torch.sqrt(ti_next**2 + sigma_data**2)
                        # *** FIX: Prepare c_noise_next correctly ***
                        c_noise_next_per_item = (0.25 * torch.log(ti_next)).expand(B) # Shape [B]

                        full_sequence_tilde = torch.cat([observed_past, x_tilde], dim=2)
                        model_input_sequence_tilde = full_sequence_tilde.clone()
                        model_input_sequence_tilde[:, :, obs_len:, :] = c_in_next * full_sequence_tilde[:, :, obs_len:, :]

                        # *** FIX: Pass corrected c_noise_next ***
                        embedded_tilde = embed_features(model_input_sequence_tilde, c_noise_next_per_item, eval=True)

                        model_out_tilde = model(embedded_tilde, roadgraph_tensor_dev, full_mask, roadgraph_mask_dev)[:, :, obs_len:, :]
                        D_theta_tilde = c_skip_next * x_tilde + c_out_next * model_out_tilde

                        d_prime_i = (1 / ti_next) * (x_tilde - D_theta_tilde)
                        x = x + (ti_next - ti) * 0.5 * (di + d_prime_i) # Update x
                    else:
                        x = x_tilde # Final update step

                # --- Calculate Loss for this batch ---
                final_sigma = T_inference[-1]
                c_skip_final = sigma_data**2 / (final_sigma**2 + sigma_data**2)
                c_out_final = final_sigma * sigma_data / torch.sqrt(sigma_data**2 + final_sigma**2)
                c_in_final = 1 / torch.sqrt(final_sigma**2 + sigma_data**2)
                # *** FIX: Prepare c_noise_final correctly ***
                c_noise_final_per_item = (0.25 * torch.log(final_sigma)).expand(B) # Shape [B]

                full_sequence_final = torch.cat([observed_past, x], dim=2)
                model_input_sequence_final = full_sequence_final.clone()
                model_input_sequence_final[:, :, obs_len:, :] = c_in_final * full_sequence_final[:, :, obs_len:, :]

                # *** FIX: Pass corrected c_noise_final ***
                embedded_final = embed_features(model_input_sequence_final, c_noise_final_per_item, eval=True)
                model_out_final = model(embedded_final, roadgraph_tensor_dev, full_mask, roadgraph_mask_dev)[:, :, obs_len:, :]

                final_predicted_x0 = c_skip_final * x + c_out_final * model_out_final

                valid_mask_loss = gt_future_mask.unsqueeze(-1).expand_as(final_predicted_x0)
                squared_diff = (final_predicted_x0 - gt_future_part) ** 2
                masked_squared_diff = squared_diff * valid_mask_loss.float()

                loss_per_sample = masked_squared_diff.sum(dim=[1, 2, 3])
                num_valid_elements_per_sample = valid_mask_loss.sum(dim=[1, 2, 3]).clamp(min=1e-6)

                total_loss += loss_per_sample.sum().item()
                total_valid_elements += num_valid_elements_per_sample.sum().item()
                num_batches += 1

                # --- Generate Plot (Only for the first batch) ---
                if batch_idx == 0 and fig_to_return is None:
                    # ... (Plotting logic remains the same - uses final_predicted_x0, etc.) ...
                    print("  Generating visualization for the first validation batch...")
                    pred_traj_future_scaled = final_predicted_x0[0]
                    gt_traj_future_scaled = gt_future_part[0]
                    initial_noise_future_scaled = (torch.randn_like(gt_traj_future_scaled) * T_inference[0])
                    trajectory_mask_future = gt_future_mask[0]

                    map_polylines_cpu = roadgraph_tensor[0]
                    polyline_masks_cpu = roadgraph_mask[0]
                    scene_stds_cpu = scene_stds[0]
                    ego_id_cpu = ego_ids[0]

                    scale_factor_cpu = scene_stds_cpu / sigma_data
                    poly_scale_factor_cpu = scale_factor_cpu[:2]

                    map_polylines_unscaled = map_polylines_cpu * poly_scale_factor_cpu[None, None, :]
                    pred_traj_unscaled = pred_traj_future_scaled.cpu() * scale_factor_cpu[None, None, :]
                    gt_traj_unscaled = gt_traj_future_scaled.cpu() * scale_factor_cpu[None, None, :]
                    noisy_traj_unscaled = initial_noise_future_scaled.cpu() * scale_factor_cpu[None, None, :]

                    fig_to_return = plot_trajectories(
                                                     map_polylines=map_polylines_unscaled.numpy(),
                                                     polyline_masks=polyline_masks_cpu.numpy(),
                                                     pred_traj=pred_traj_unscaled.numpy(),
                                                    gt_traj=gt_traj_unscaled.numpy(),
                                                    noisy_traj=noisy_traj_unscaled.numpy(),
                                                    trajectory_mask=trajectory_mask_future.cpu().numpy(),
                                                    ego_id=ego_id_cpu,
                                                    eval=True,
                                                    step=batch_idx,
                                                    title_suffix="Validation",
                                                    extra_info={
                                                                "batch_size": val_batch_size,
                                                                "num_files": len(val_dataset),
                                                                "avg_loss": f"{avg_loss:.4f}" if 'avg_loss' in locals() else "N/A"
                                                                }
                                                    )
                    print("  Visualization generated.")

                if (batch_idx + 1) % 20 == 0:
                    print(f"  Validation Batch {batch_idx+1}/{len(val_dataloader)} Processed...")

            except Exception as e:
                print(f"  Error processing validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue


    if total_valid_elements > 0:
        avg_loss = total_loss / total_valid_elements
    else:
        print("Warning: No valid elements found in validation set for loss calculation.")
        avg_loss = float('nan')

    end_time = time.time()
    print(f"Validation finished in {end_time - start_time:.2f}s. Average Loss: {avg_loss:.4f}")

    model.train()
    return avg_loss, fig_to_return
