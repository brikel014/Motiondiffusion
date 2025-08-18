import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.signal import savgol_filter 
from networks_2 import Denoiser 
from utils import plot_trajectories, sample_noise, embed_features 
from map_preprocessing import MapDataset 
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import math
import time
import traceback 


import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms


# --- Constants ---
SIGMA_MAX = 80
SIGMA_MIN = 0.002
RHO = 7
N_DENOISING_STEPS = 50
K_SAMPLES = 6 # Number of samples for min metrics
ADE_THRESHOLD = 5.0
OBS_LEN = 10
PRED_LEN = 20

# --- NEW Smoothing Constants ---
APPLY_SMOOTHING = True  # Master switch for applying Savitzky-Golay smoothing
SAVGOL_WINDOW_LEN = 15  # Window length for Savitzky-Golay filter (must be odd)
SAVGOL_POLYORDER = 3    # Polynomial order for Savitzky-Golay filter (must be < SAVGOL_WINDOW_LEN)

if APPLY_SMOOTHING:
    if SAVGOL_WINDOW_LEN % 2 == 0:
        SAVGOL_WINDOW_LEN -=1 
        if SAVGOL_WINDOW_LEN > 0:
            print(f"Warning: SAVGOL_WINDOW_LEN was even, adjusted to be odd: {SAVGOL_WINDOW_LEN}")
        else: # Became 0 or less, invalid
             print(f"Warning: SAVGOL_WINDOW_LEN became non-positive after adjustment ({SAVGOL_WINDOW_LEN}). Disabling smoothing.")
             APPLY_SMOOTHING = False
    
    if APPLY_SMOOTHING and SAVGOL_WINDOW_LEN <= 0: # Final check if window length is valid
        print(f"Warning: SAVGOL_WINDOW_LEN is {SAVGOL_WINDOW_LEN}, which is invalid. Disabling smoothing.")
        APPLY_SMOOTHING = False
    
    if APPLY_SMOOTHING and SAVGOL_POLYORDER >= SAVGOL_WINDOW_LEN:
        original_polyorder = SAVGOL_POLYORDER
        SAVGOL_POLYORDER = max(1, SAVGOL_WINDOW_LEN - 1) # Ensure polyorder is at least 1 and < window
        if SAVGOL_WINDOW_LEN <= 1 and SAVGOL_POLYORDER >= SAVGOL_WINDOW_LEN : # e.g. window=1, polyorder becomes 0, still invalid for some sgola defs
             print(f"Warning: SAVGOL_WINDOW_LEN ({SAVGOL_WINDOW_LEN}) too small to set a valid SAVGOL_POLYORDER. Disabling smoothing.")
             APPLY_SMOOTHING = False
        else:
            print(f"Warning: SAVGOL_POLYORDER ({original_polyorder}) was >= SAVGOL_WINDOW_LEN ({SAVGOL_WINDOW_LEN}). Adjusted to {SAVGOL_POLYORDER}.")

    if APPLY_SMOOTHING and SAVGOL_POLYORDER < 0 : # Should not happen with max(1, ...) but as a safeguard
        print(f"Warning: SAVGOL_POLYORDER is {SAVGOL_POLYORDER}, which is invalid. Disabling smoothing.")
        APPLY_SMOOTHING = False
    
    if APPLY_SMOOTHING:
        print(f"Trajectory smoothing ENABLED with SavGol: Window={SAVGOL_WINDOW_LEN}, Polyorder={SAVGOL_POLYORDER}")
    else:
        print("Trajectory smoothing DISABLED due to parameter issues or explicit setting.")


# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Constants ---
LANE_WIDTH = 3.5  # meters, typical lane width
CAR_LENGTH_VIZ = 4.0 # meters, for visualization
CAR_WIDTH_VIZ = 1.8  # meters, for visualization

# --- Helper function for lane boundaries ---
def _get_lane_boundaries(polyline, lane_half_width):
    if polyline.shape[0] < 2:
        return None, None

    left_boundary_points = []
    right_boundary_points = []
    
    segment_vectors = np.diff(polyline, axis=0)
    
    normals = np.array([-segment_vectors[:, 1], segment_vectors[:, 0]]).T
    
    norms_magnitude = np.linalg.norm(normals, axis=1)
    valid_norms_mask = norms_magnitude > 1e-6 
    
    normalized_normals = np.zeros_like(normals, dtype=float)
    if np.any(valid_norms_mask):
        normalized_normals[valid_norms_mask] = normals[valid_norms_mask] / norms_magnitude[valid_norms_mask, np.newaxis]

    for i in range(polyline.shape[0]):
        point = polyline[i]
        
        if polyline.shape[0] == 1: 
             current_normal = np.array([0.0, 0.0])
        elif i == 0: 
            current_normal = normalized_normals[0] if valid_norms_mask[0] else np.array([0.0, 0.0])
        elif i == polyline.shape[0] - 1: 
            current_normal = normalized_normals[-1] if valid_norms_mask[-1] else np.array([0.0, 0.0])
        else: 
            normal_prev = normalized_normals[i-1] if valid_norms_mask[i-1] else np.array([0.0, 0.0])
            normal_next = normalized_normals[i]   if valid_norms_mask[i]   else np.array([0.0, 0.0])
            
            avg_normal = normal_prev + normal_next
            avg_norm_mag = np.linalg.norm(avg_normal)
            
            if avg_norm_mag > 1e-6:
                current_normal = avg_normal / avg_norm_mag
            elif np.linalg.norm(normal_next) > 1e-6 :
                current_normal = normal_next
            elif np.linalg.norm(normal_prev) > 1e-6:
                 current_normal = normal_prev
            else: 
                current_normal = np.array([0.0, 0.0]) 

        left_boundary_points.append(point + current_normal * lane_half_width)
        right_boundary_points.append(point - current_normal * lane_half_width)
        
    return np.array(left_boundary_points), np.array(right_boundary_points)

# --- Noise Schedule ---
def get_T(sigma_max, sigma_min, rho, N):
    T = []
    for i in range(N):
        T.append((sigma_max**(1/rho) + (i/(N-1)) * ((sigma_min**(1/rho)) - (sigma_max**(1/rho))))**rho)
    return torch.tensor(T, device=device)

#T_schedule = get_T(SIGMA_MAX, SIGMA_MIN, RHO, N_DENOISING_STEPS)

# --- Denoising Function ---
def denoise_single_trajectory(model, observed_past, feature_mask_real, roadgraph_tensor, roadgraph_mask, T):
    batch_size, num_real_agents, obs_len, num_features = observed_past.shape
    pred_len = PRED_LEN
    x = torch.randn(batch_size, num_real_agents, pred_len, num_features, device=device) * T[0]
    for i in range(N_DENOISING_STEPS - 1):
        ti = T[i]
        ti_next = T[i + 1]
        sigma_data = 0.5
        c_skip = sigma_data**2 / (ti**2 + sigma_data**2)
        c_out = ti * sigma_data / torch.sqrt(sigma_data**2 + ti**2)
        c_in = 1 / torch.sqrt(ti**2 + sigma_data**2)
        c_noise = 0.25 * torch.log(ti).unsqueeze(0)
        if c_noise.dim() == 1 and batch_size > 1:
             c_noise = c_noise.view(-1, 1, 1, 1)
        full_sequence_real = torch.cat([observed_past, x], dim=2)
        result_input_real = full_sequence_real.clone()
        result_input_real[:, :, obs_len:, :] = c_in * full_sequence_real[:, :, obs_len:, :]
        try:
            embedded_real = embed_features(result_input_real, c_noise, feature_mask=feature_mask_real, eval=True)
        except TypeError:
            embedded_real = embed_features(result_input_real, c_noise, eval=True)
        model_out_real = model(embedded_real, roadgraph_tensor, feature_mask=feature_mask_real, roadgraph_mask=roadgraph_mask)
        if model_out_real.shape[1] != num_real_agents:
             raise ValueError(f"Model output agent dimension ({model_out_real.shape[1]}) doesn't match input ({num_real_agents})")
        model_out_future = model_out_real[:, :, obs_len:, :]
        D_theta = c_skip * x + c_out * model_out_future
        di = (1 / ti) * (x - D_theta)
        x_tilde = x + (ti_next - ti) * di
        if ti_next > SIGMA_MIN:
            c_skip_next = sigma_data**2 / (ti_next**2 + sigma_data**2)
            c_out_next = ti_next * sigma_data / torch.sqrt(sigma_data**2 + ti_next**2)
            c_in_next = 1 / torch.sqrt(ti_next**2 + sigma_data**2)
            c_noise_next = 0.25 * torch.log(ti_next).unsqueeze(0)
            if c_noise_next.dim() == 1 and batch_size > 1:
                c_noise_next = c_noise_next.view(-1, 1, 1, 1)
            full_sequence_tilde_real = torch.cat([observed_past, x_tilde], dim=2)
            result_tilde_real = full_sequence_tilde_real.clone()
            result_tilde_real[:, :, obs_len:, :] = c_in_next * full_sequence_tilde_real[:, :, obs_len:, :]
            try:
                embedded_tilde_real = embed_features(result_tilde_real, c_noise_next, feature_mask=feature_mask_real, eval=True)
            except TypeError:
                embedded_tilde_real = embed_features(result_tilde_real, c_noise_next, eval=True)
            model_out_tilde_real = model(embedded_tilde_real, roadgraph_tensor, feature_mask=feature_mask_real, roadgraph_mask=roadgraph_mask)
            if model_out_tilde_real.shape[1] != num_real_agents:
                 raise ValueError(f"Model (tilde) output agent dimension ({model_out_tilde_real.shape[1]}) doesn't match input ({num_real_agents})")
            model_out_tilde_future = model_out_tilde_real[:, :, obs_len:, :]
            D_theta_tilde = c_skip_next * x_tilde + c_out_next * model_out_tilde_future
            d_prime_i = (1 / ti_next) * (x_tilde - D_theta_tilde)
            x = x + (ti_next - ti) * 0.5 * (di + d_prime_i)
        else:
            x = x_tilde
    pred_future_local = x
    return pred_future_local

# --- Coordinate Transformation ---
def transform_to_global(traj_local_scaled, ego_state_global, scene_std):
    sigma_data = 0.5
    scale_factor_inv = scene_std.to(traj_local_scaled.device) / sigma_data
    if scale_factor_inv.dim() == 1 and traj_local_scaled.dim() == 3:
        scale_factor_inv = scale_factor_inv[None, None, :]
    elif scale_factor_inv.dim() == 1 and traj_local_scaled.dim() == 2:
        scale_factor_inv = scale_factor_inv[None, :]
    elif traj_local_scaled.dim() > 1 and scale_factor_inv.shape != traj_local_scaled.shape[-1:]: 
        if scale_factor_inv.shape[0] == traj_local_scaled.shape[-1]: 
            pass 
        else:
            raise ValueError(f"Cannot broadcast scene_std shape {scene_std.shape} ({scale_factor_inv.shape}) with trajectory feature shape {traj_local_scaled.shape[-1]}")

    traj_unscaled = traj_local_scaled * scale_factor_inv
    x_ego, y_ego, theta_ego = ego_state_global
    cos_theta = torch.cos(theta_ego)
    sin_theta = torch.sin(theta_ego)
    x_rel = traj_unscaled[..., 0]
    y_rel = traj_unscaled[..., 1]
    theta_rel = traj_unscaled[..., 2]
    dx = x_rel * cos_theta - y_rel * sin_theta
    dy = x_rel * sin_theta + y_rel * cos_theta
    x_global = x_ego + dx
    y_global = y_ego + dy
    theta_global = theta_rel + theta_ego
    theta_global = torch.atan2(torch.sin(theta_global), torch.cos(theta_global))
    global_traj = torch.stack([x_global, y_global, theta_global], dim=-1)
    if traj_unscaled.shape[-1] > 3:
        other_features_unscaled = traj_unscaled[..., 3:]
        global_traj = torch.cat([global_traj, other_features_unscaled], dim=-1)
    return global_traj

# --- NEW Trajectory Smoothing Function ---
def smooth_trajectory_savgol(trajectory_tensor, window_length, polyorder):
    """
    Smoothes a single agent's trajectory using the Savitzky-Golay filter.
    trajectory_tensor: torch.Tensor of shape [PredLen, NumFeatures].
                       Assumes x, y, theta are the first three features.
                       Optionally smooths vx, vy if they are features 3 and 4.
    window_length: Length of the filter window (must be odd).
    polyorder: Polynomial order (must be less than window_length).
    """
    if not isinstance(trajectory_tensor, torch.Tensor):
        raise TypeError("Input trajectory must be a PyTorch tensor.")
    if trajectory_tensor.ndim != 2:
        raise ValueError(f"Input trajectory must be 2D (PredLen, NumFeatures), got {trajectory_tensor.ndim}D.")

    pred_len, num_features = trajectory_tensor.shape

    if pred_len == 0: # No points to smooth
        return trajectory_tensor

    # Adjust window_length if trajectory is too short
    effective_window_length = min(window_length, pred_len)
    if effective_window_length % 2 == 0: # Ensure odd
        effective_window_length = max(1, effective_window_length - 1) # Max with 1 to prevent 0 or negative
    
    # Ensure polyorder is less than effective_window_length and non-negative
    effective_polyorder = min(polyorder, effective_window_length - 1)
    effective_polyorder = max(0, effective_polyorder) # Polyorder can be 0 (moving average)

    if effective_window_length <= effective_polyorder or effective_window_length <= 0:
        if effective_polyorder > 0 and effective_window_length <= effective_polyorder:

            return trajectory_tensor
        if effective_window_length < 1 : 
            return trajectory_tensor


    smoothed_trajectory_np = trajectory_tensor.cpu().numpy().copy()

    if num_features > 0:
        smoothed_trajectory_np[:, 0] = savgol_filter(smoothed_trajectory_np[:, 0], effective_window_length, effective_polyorder)
    if num_features > 1:
        smoothed_trajectory_np[:, 1] = savgol_filter(smoothed_trajectory_np[:, 1], effective_window_length, effective_polyorder)


    if num_features > 2:
        theta = smoothed_trajectory_np[:, 2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        smoothed_cos_theta = savgol_filter(cos_theta, effective_window_length, effective_polyorder)
        smoothed_sin_theta = savgol_filter(sin_theta, effective_window_length, effective_polyorder)
        
        smoothed_trajectory_np[:, 2] = np.arctan2(smoothed_sin_theta, smoothed_cos_theta)
    

    if num_features > 3: # vx
        smoothed_trajectory_np[:, 3] = savgol_filter(smoothed_trajectory_np[:, 3], effective_window_length, effective_polyorder)
    if num_features > 4: # vy
        smoothed_trajectory_np[:, 4] = savgol_filter(smoothed_trajectory_np[:, 4], effective_window_length, effective_polyorder)

    return torch.from_numpy(smoothed_trajectory_np).to(trajectory_tensor.device)

# --- Metrics Calculation ---
def calculate_ade(pred, gt, mask):
    mask_float = mask.float()
    diff = pred[..., :2] - gt[..., :2]
    dist = torch.norm(diff, dim=-1)
    dist = dist * mask_float
    sum_dist_per_agent = torch.sum(dist, dim=1)
    valid_timesteps_per_agent = torch.sum(mask_float, dim=1)
    ade_per_agent = torch.zeros_like(sum_dist_per_agent)
    valid_agent_mask = valid_timesteps_per_agent > 0
    ade_per_agent[valid_agent_mask] = sum_dist_per_agent[valid_agent_mask] / valid_timesteps_per_agent[valid_agent_mask]
    num_valid_agents = torch.sum(valid_agent_mask).item()
    if num_valid_agents == 0:
        return torch.tensor(0.0, device=pred.device), ade_per_agent, valid_agent_mask
    scene_ade = torch.sum(ade_per_agent[valid_agent_mask]) / num_valid_agents
    return scene_ade, ade_per_agent, valid_agent_mask

def calculate_fde(pred, gt, mask):
    valid_timesteps_per_agent = torch.sum(mask.float(), dim=1)
    valid_agent_mask = valid_timesteps_per_agent > 0
    num_valid_agents = torch.sum(valid_agent_mask).item()
    if num_valid_agents == 0:
        return torch.tensor(0.0, device=pred.device)
    final_indices = (valid_timesteps_per_agent[valid_agent_mask] - 1).long().clamp(min=0)
    if pred.shape[0] != mask.shape[0] or gt.shape[0] != mask.shape[0]:
         raise ValueError(f"Mismatch in agent dimension for FDE calculation. Pred: {pred.shape[0]}, GT: {gt.shape[0]}, Mask: {mask.shape[0]}")
    pred_valid_agents = pred[valid_agent_mask]
    gt_valid_agents = gt[valid_agent_mask]
    idx_gather = final_indices.view(-1, 1, 1).expand(-1, 1, gt.shape[-1])
    final_pred = torch.gather(pred_valid_agents, 1, idx_gather).squeeze(1)
    final_gt = torch.gather(gt_valid_agents, 1, idx_gather).squeeze(1)
    diff_final = final_pred[..., :2] - final_gt[..., :2]
    dist_final = torch.norm(diff_final, dim=-1)
    scene_fde = torch.mean(dist_final)
    return scene_fde

# --- XML Update Function ---
def update_xml_with_predictions(xml_path, predictions_by_agent, output_dir, segment_suffix=""):
    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found at {xml_path}. Skipping update.")
        return False
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Warning: Could not parse XML file {xml_path}. Skipping update.")
        return False
    updated = False
    for dob in root.findall('.//dynamicObstacle'):
        agent_id = dob.get('id')
        if agent_id in predictions_by_agent:
            traj_elem = dob.find('trajectory')
            if traj_elem is None: traj_elem = ET.SubElement(dob, 'trajectory')
            existing_states = {}
            for state in list(traj_elem):
                time_elem = state.find('time/exact')
                if time_elem is not None and time_elem.text is not None:
                    try:
                        timestep = int(time_elem.text)
                        existing_states[timestep] = state
                    except (ValueError, TypeError):
                        print(f"Warning: Non-integer or invalid exact time for agent {agent_id}, state ignored.")
            for timestep, (x, y, theta) in sorted(predictions_by_agent[agent_id].items()):
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)):
                    print(f"Warning: Invalid prediction values for agent {agent_id} at timestep {timestep}. Skipping.")
                    continue
                if timestep in existing_states:
                    state = existing_states[timestep]
                    state_updated = False
                    pos_elem = state.find('position');
                    if pos_elem is None: pos_elem = ET.SubElement(state, 'position')
                    point_elem = pos_elem.find('point');
                    if point_elem is None: point_elem = ET.SubElement(pos_elem, 'point')
                    x_elem = point_elem.find('x')
                    if x_elem is None: ET.SubElement(point_elem, 'x').text = f"{x:.4f}"; state_updated = True
                    elif abs(float(x_elem.text) - x) > 1e-4 : x_elem.text = f"{x:.4f}"; state_updated = True
                    y_elem = point_elem.find('y')
                    if y_elem is None: ET.SubElement(point_elem, 'y').text = f"{y:.4f}"; state_updated = True
                    elif abs(float(y_elem.text) - y) > 1e-4: y_elem.text = f"{y:.4f}"; state_updated = True
                    orient_elem = state.find('orientation')
                    if orient_elem is None: orient_elem = ET.SubElement(state, 'orientation')
                    theta_elem = orient_elem.find('exact')
                    if theta_elem is None: ET.SubElement(orient_elem, 'exact').text = f"{theta:.4f}"; state_updated = True
                    elif abs(float(theta_elem.text) - theta) > 1e-4: theta_elem.text = f"{theta:.4f}"; state_updated = True
                    if state_updated: updated = True
                else:
                    state = ET.SubElement(traj_elem, 'state')
                    time_cont = ET.SubElement(state, 'time'); ET.SubElement(time_cont, 'exact').text = str(timestep)
                    pos_elem = ET.SubElement(state, 'position'); point_elem = ET.SubElement(pos_elem, 'point')
                    ET.SubElement(point_elem, 'x').text = f"{x:.4f}"; ET.SubElement(point_elem, 'y').text = f"{y:.4f}"
                    orient_elem = ET.SubElement(state, 'orientation'); ET.SubElement(orient_elem, 'exact').text = f"{theta:.4f}"
                    updated = True
    if updated:
        output_filename = os.path.basename(xml_path)
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            if hasattr(ET, 'indent'): ET.indent(tree, space="  ", level=0)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            return True
        except IOError as e:
            print(f"Error writing XML file {output_path}: {e}")
            return False
    else:
        return False

# --- NEW HELPER FUNCTION FOR PLOTTING ROADGRAPH ---
def transform_polylines_to_global(polylines_local_scaled, roadgraph_mask, ego_state_global, scene_std):
    if polylines_local_scaled.numel() == 0:
        return []

    sigma_data = 0.5
    relevant_scene_std_xy = scene_std[:2].to(polylines_local_scaled.device) 
    scale_factor_inv_xy = relevant_scene_std_xy / sigma_data

    if polylines_local_scaled.shape[-1] < 2:
        print("Warning: Polyline features < 2, cannot extract x,y for transformation.")
        return []
    polylines_unscaled_xy = polylines_local_scaled[..., :2] * scale_factor_inv_xy[None, None, :]

    x_ego, y_ego, theta_ego = ego_state_global
    cos_theta = torch.cos(theta_ego)
    sin_theta = torch.sin(theta_ego)

    global_polylines_list = []
    valid_polylines_indices = torch.where(roadgraph_mask)[0]

    for p_idx in valid_polylines_indices:
        poly_xy_unscaled = polylines_unscaled_xy[p_idx] 
        valid_points_mask = torch.any(poly_xy_unscaled != 0, dim=1)
        poly_xy_unscaled_valid = poly_xy_unscaled[valid_points_mask]

        if poly_xy_unscaled_valid.shape[0] < 2: 
            continue

        x_rel = poly_xy_unscaled_valid[:, 0]
        y_rel = poly_xy_unscaled_valid[:, 1]

        dx = x_rel * cos_theta - y_rel * sin_theta
        dy = x_rel * sin_theta + y_rel * cos_theta
        x_global = x_ego + dx
        y_global = y_ego + dy

        global_poly_points = torch.stack([x_global, y_global], dim=-1).cpu().numpy()
        global_polylines_list.append(global_poly_points)
    return global_polylines_list

# --- NEW PLOTTING FUNCTION ---
def plot_scenario_and_save_png(
    observed_past_global,
    all_predicted_futures_global,
    best_pred_future_global,
    roadgraph_polylines_global,
    agent_ids_to_plot,
    ego_id,
    output_filepath,
    title=""
):
    """
    Plots the scenario including agent history, predicted trajectories, and roadgraph.
    Saves the plot to a PNG file.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot roadgraph boundaries
    lane_half_width = LANE_WIDTH / 2.0
    for centerline_polyline in roadgraph_polylines_global:
        if centerline_polyline.shape[0] >= 2:
            left_boundary, right_boundary = _get_lane_boundaries(centerline_polyline, lane_half_width)
            if left_boundary is not None and left_boundary.shape[0] >= 2:
                ax.plot(left_boundary[:, 0], left_boundary[:, 1], color='darkgray', linewidth=0.5, alpha=0.7, zorder=1)
            if right_boundary is not None and right_boundary.shape[0] >= 2:
                ax.plot(right_boundary[:, 0], right_boundary[:, 1], color='darkgray', linewidth=0.5, alpha=0.7, zorder=1)

    # --- MODIFIED COLORS: Unified colors for agents and predictions ---
    color_agent_t0 = 'darkblue'                 # Unified color for all agents at T=0
    color_predicted_sample_band = 'deepskyblue' # Color for the band of predicted samples
    color_best_prediction = 'purple'            # Unified color for the best prediction for all agents

    # Collect all coordinates to set plot bounds
    all_x, all_y = [], []
    def _collect_coords_from_dict(data_dict, target_agent_ids):
        if data_dict:
            for aid_key in target_agent_ids:
                if aid_key in data_dict and data_dict[aid_key].shape[0] > 0:
                    all_x.extend(data_dict[aid_key][:, 0])
                    all_y.extend(data_dict[aid_key][:, 1])

    def _collect_coords_from_list_of_dicts(list_of_dicts, target_agent_ids):
        if list_of_dicts:
            for traj_dict_sample in list_of_dicts:
                _collect_coords_from_dict(traj_dict_sample, target_agent_ids)

    _collect_coords_from_dict(observed_past_global, agent_ids_to_plot)
    _collect_coords_from_list_of_dicts(all_predicted_futures_global, agent_ids_to_plot)
    _collect_coords_from_dict(best_pred_future_global, agent_ids_to_plot)

    temp_rg_x, temp_rg_y = [], []
    for centerline_polyline in roadgraph_polylines_global:
         if centerline_polyline.shape[0] >=2:
            left_boundary, right_boundary = _get_lane_boundaries(centerline_polyline, lane_half_width)
            if left_boundary is not None and left_boundary.shape[0] > 0:
                temp_rg_x.extend(left_boundary[:,0]); temp_rg_y.extend(left_boundary[:,1])
            if right_boundary is not None and right_boundary.shape[0] > 0:
                temp_rg_x.extend(right_boundary[:,0]); temp_rg_y.extend(right_boundary[:,1])
    if not temp_rg_x and roadgraph_polylines_global:
        for poly in roadgraph_polylines_global:
            if poly.shape[0] > 0:
                temp_rg_x.extend(poly[:,0]); temp_rg_y.extend(poly[:,1])
    all_x.extend(temp_rg_x)
    all_y.extend(temp_rg_y)

    # Handle cases with no data to plot
    if not all_x or not all_y:
        is_data_present = False
        if observed_past_global:
            for aid_key in agent_ids_to_plot:
                if aid_key in observed_past_global and observed_past_global[aid_key].shape[0] > 0:
                    is_data_present = True; break
        if not is_data_present and roadgraph_polylines_global:
             if any(poly.shape[0] > 0 for poly in roadgraph_polylines_global):
                 is_data_present = True

        if not is_data_present:
            print(f"Warning: No plottable data for {output_filepath}. Skipping plot generation.")
            plt.close(fig)
            return

    if not all_x: all_x = [0.0]
    if not all_y: all_y = [0.0]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_range = max(max_x - min_x, 10.0)
    y_range = max(max_y - min_y, 10.0)
    padding = max(x_range, y_range) * 0.10

    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect('equal', adjustable='box')

    plotted_labels = set()

    for aid in agent_ids_to_plot:
        is_ego = (str(aid) == str(ego_id))
        z_order_ego_bonus = 5 if is_ego else 0

        lw_pred_best = 0.9 if is_ego else 0.7
        lw_pred_sample = lw_pred_best * 0.6
        pred_alpha = 0.95
        sample_alpha = 0.25

        current_pos_processed_for_agent = False
        current_x_for_pred, current_y_for_pred = None, None

        # Plot agent's current position (T=0)
        if aid in observed_past_global and observed_past_global[aid].shape[0] > 0:
            obs_traj = observed_past_global[aid]
            num_obs_points = obs_traj.shape[0]
            num_obs_features = obs_traj.shape[1]

            current_x, current_y = obs_traj[-1, 0], obs_traj[-1, 1]
            heading_rad = 0.0

            if num_obs_features > 2 and not np.isnan(obs_traj[-1, 2]):
                heading_rad = obs_traj[-1, 2]
            elif num_obs_points > 1:
                dx = obs_traj[-1, 0] - obs_traj[-2, 0]
                dy = obs_traj[-1, 1] - obs_traj[-2, 1]
                if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                    heading_rad = np.arctan2(dy, dx)
            
            # Use unified color for all agent car bodies
            car_rect = patches.Rectangle(
                (-CAR_LENGTH_VIZ / 2, -CAR_WIDTH_VIZ / 2),
                CAR_LENGTH_VIZ, CAR_WIDTH_VIZ,
                facecolor=color_agent_t0, # MODIFIED
                edgecolor='black', linewidth=0.3,
                zorder=4 + z_order_ego_bonus
            )
            transform = mtransforms.Affine2D().rotate(heading_rad).translate(current_x, current_y) + ax.transData
            car_rect.set_transform(transform)
            ax.add_patch(car_rect)

            # Use unified label for all agents at T=0
            label_car = "Agent at T=0" # MODIFIED
            if label_car not in plotted_labels:
                ax.plot([], [], color=color_agent_t0, marker='s', linestyle='None', markersize=8, label=label_car) # MODIFIED
                plotted_labels.add(label_car)

            current_pos_processed_for_agent = True
            current_x_for_pred, current_y_for_pred = current_x, current_y

        # Plot all predicted futures (samples)
        if all_predicted_futures_global and len(all_predicted_futures_global) > 0:
            label_pred_sample = "Predicted samples (K)"

            for k_sample_idx, k_sample_preds_dict in enumerate(all_predicted_futures_global):
                if aid in k_sample_preds_dict and k_sample_preds_dict[aid].shape[0] > 0:
                    pred_traj_k = k_sample_preds_dict[aid]

                    line_k, = ax.plot(pred_traj_k[:, 0], pred_traj_k[:, 1], color=color_predicted_sample_band,
                                 linewidth=lw_pred_sample, alpha=sample_alpha,
                                 zorder=2 + z_order_ego_bonus)

                    if label_pred_sample not in plotted_labels:
                         line_k.set_label(label_pred_sample)
                         plotted_labels.add(label_pred_sample)

                    if current_pos_processed_for_agent:
                         ax.plot([current_x_for_pred, pred_traj_k[0,0]], [current_y_for_pred, pred_traj_k[0,1]],
                                 color=color_predicted_sample_band, linewidth=lw_pred_sample, alpha=sample_alpha,
                                 zorder=2 + z_order_ego_bonus)

        # Plot the single best predicted future
        if aid in best_pred_future_global and best_pred_future_global[aid].shape[0] > 0:
            best_pred_traj = best_pred_future_global[aid]
            
            # Use unified label and color for the best prediction
            label_best_pred = "Best prediction (ADE)" # MODIFIED

            line_best, = ax.plot(best_pred_traj[:, 0], best_pred_traj[:, 1], color=color_best_prediction, # MODIFIED
                               linewidth=lw_pred_best, alpha=pred_alpha,
                               zorder=5 + z_order_ego_bonus)

            if label_best_pred not in plotted_labels:
                line_best.set_label(label_best_pred)
                plotted_labels.add(label_best_pred)

            if current_pos_processed_for_agent:
                 ax.plot([current_x_for_pred, best_pred_traj[0,0]], [current_y_for_pred, best_pred_traj[0,1]],
                         color=color_best_prediction, linewidth=lw_pred_best, alpha=pred_alpha, # MODIFIED
                         zorder=5 + z_order_ego_bonus)

    # Final plot setup
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.set_title(title, fontsize=12)
    if plotted_labels:
        ax.legend(loc='best', fontsize='small', framealpha=0.9)
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.6)

    try:
        plt.savefig(output_filepath, bbox_inches='tight', dpi=800)
    except Exception as e:
        print(f"Error saving plot {output_filepath}: {e}")
    plt.close(fig)

# --- Main Processing Function ---
def run_inference_and_save(model_path, xml_input_dir, output_dir, metrics_filename="final_metrics.json"):
    print("Loading model...")
    model = Denoiser().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        traceback.print_exc(); return
    model.eval()
    print("Model loaded.")

    print("Loading dataset...")
    try:
        dataset = MapDataset(xml_dir=xml_input_dir, obs_len=OBS_LEN, pred_len=PRED_LEN, num_timesteps=OBS_LEN + PRED_LEN,
                             max_radius=100, num_polylines=500, num_points=10, max_agents=32,
                             save_plots=False, single_ego=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"Dataset loaded with {len(dataset)} segments.")
    except Exception as e:
        print(f"Error creating dataset/dataloader: {e}")
        traceback.print_exc(); return

    plot_output_dir = os.path.join(output_dir, 'scenario_plots_png')
    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"Scenario plots will be saved to: {plot_output_dir}")

    all_global_predictions_for_xml = defaultdict(lambda: defaultdict(dict))
    valid_metrics_accumulator = []
    total_segments_processed = 0
    segments_above_threshold = 0

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            total_segments_processed += 1
            scenario_id, segment_start_time = 'unknown_scenario', 'unknown_segment' 
            batch_info_str = f"Batch {batch_idx}/{len(dataloader)-1}"

            try:
                processed_batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
                batch = tuple(processed_batch)

                ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask, \
                observed, observed_masks, ground_truth, ground_truth_masks, \
                scene_means, scene_stds, scenario_id_list, segment_start_tensor, \
                ego_state_global, agent_ids_list = batch

                feature_tensor_squeezed = feature_tensor.squeeze(0)
                feature_mask_squeezed = feature_mask.squeeze(0)
                roadgraph_tensor_squeezed = roadgraph_tensor.squeeze(0) 
                roadgraph_mask_squeezed = roadgraph_mask.squeeze(0)     
                ground_truth_squeezed = ground_truth.squeeze(0)
                ground_truth_masks_squeezed = ground_truth_masks.squeeze(0)
                scene_stds_single = scene_stds.squeeze(0)
                ego_state_global_single = ego_state_global.squeeze(0)

                scenario_id = scenario_id_list[0]
                segment_start_time = segment_start_tensor.item()
                current_ego_id = ego_ids[0]
                current_agent_ids = [aid[0] if isinstance(aid, (list, tuple)) and len(aid)>0 else aid for aid in agent_ids_list]
                batch_info_str = f"Batch {batch_idx}/{len(dataloader)-1} | Scen: {os.path.basename(scenario_id)} | SegStart: {segment_start_time}"

                initial_agent_mask = feature_mask_squeezed[:, 0].bool()
                num_real_agents = initial_agent_mask.sum().item()

                if num_real_agents == 0:
                    print(f"{batch_info_str}: Skipping - No real agents found at t=0.")
                    continue

                observed_past_real_local_scaled = feature_tensor_squeezed[initial_agent_mask, :OBS_LEN, :]
                gt_future_real_local_scaled = ground_truth_squeezed[initial_agent_mask]
                gt_future_mask_real = ground_truth_masks_squeezed[initial_agent_mask] 
                real_agent_ids = [aid for aid, keep in zip(current_agent_ids, initial_agent_mask.cpu().tolist()) if keep]
                feature_mask_real_total = feature_mask_squeezed[initial_agent_mask] 

                observed_past_real_b = observed_past_real_local_scaled.unsqueeze(0)
                feature_mask_real_total_b = feature_mask_real_total.unsqueeze(0)

                all_pred_futures_local_scaled_samples = []
                for k in range(K_SAMPLES):
                    pred_future_local_k_b = denoise_single_trajectory(
                        model, observed_past_real_b, feature_mask_real_total_b,
                        roadgraph_tensor, roadgraph_mask, T_schedule 
                    )
                    all_pred_futures_local_scaled_samples.append(pred_future_local_k_b.squeeze(0))

                gt_future_global_tensor = transform_to_global(gt_future_real_local_scaled, ego_state_global_single, scene_stds_single)
                observed_past_global_tensor = transform_to_global(observed_past_real_local_scaled, ego_state_global_single, scene_stds_single)
                
                roadgraph_polylines_global_list = transform_polylines_to_global(
                    roadgraph_tensor_squeezed, roadgraph_mask_squeezed,
                    ego_state_global_single, scene_stds_single
                )

                scene_ades_k_global, scene_fdes_k_global = [], []
                all_pred_futures_global_samples_tensor = [] # This will store smoothed trajectories
                
                for k_idx in range(K_SAMPLES):
                    pred_k_local_scaled = all_pred_futures_local_scaled_samples[k_idx]
                    pred_k_global_original = transform_to_global(pred_k_local_scaled, ego_state_global_single, scene_stds_single)
                    
                    pred_k_global_to_use = pred_k_global_original # Default to original

                    if APPLY_SMOOTHING and pred_k_global_original.shape[0] > 0: # If there are agents
                        pred_k_global_smoothed_list = []
                        for agent_idx in range(pred_k_global_original.shape[0]):
                            single_agent_traj_global = pred_k_global_original[agent_idx] # [PredLen, NumFeatures]
                            smoothed_agent_traj = smooth_trajectory_savgol(
                                single_agent_traj_global, 
                                window_length=SAVGOL_WINDOW_LEN, 
                                polyorder=SAVGOL_POLYORDER
                            )
                            pred_k_global_smoothed_list.append(smoothed_agent_traj)
                        
                        if pred_k_global_smoothed_list:
                             pred_k_global_to_use = torch.stack(pred_k_global_smoothed_list)
                        # else: pred_k_global_to_use remains pred_k_global_original (e.g. if list is empty)
                    
                    all_pred_futures_global_samples_tensor.append(pred_k_global_to_use) 
                    
                    scene_ade_k, _, valid_agent_mask_k = calculate_ade(pred_k_global_to_use, gt_future_global_tensor, gt_future_mask_real)
                    scene_ades_k_global.append(scene_ade_k)
                    scene_fdes_k_global.append(calculate_fde(pred_k_global_to_use, gt_future_global_tensor, gt_future_mask_real) if valid_agent_mask_k.any() else torch.tensor(0.0, device=device))

                scene_ades_k_global_tensor = torch.stack(scene_ades_k_global)
                min_ade_index = torch.argmin(scene_ades_k_global_tensor)
                min_scene_ade_global = scene_ades_k_global_tensor[min_ade_index].item()
                min_scene_fde_global = torch.stack(scene_fdes_k_global)[min_ade_index].item()
                best_pred_future_global_tensor = all_pred_futures_global_samples_tensor[min_ade_index.item()]

                scene_ade_single_global = scene_ades_k_global[0].item() # Metrics for the first (smoothed) sample
                scene_fde_single_global = scene_fdes_k_global[0].item() # Metrics for the first (smoothed) sample

                print(f"{batch_info_str} | ADE_glob: {scene_ade_single_global:.4f} | FDE_glob: {scene_fde_single_global:.4f} | minADE_glob: {min_scene_ade_global:.4f} | minFDE_glob: {min_scene_fde_global:.4f}")

                if scene_ade_single_global > ADE_THRESHOLD:
                    segments_above_threshold += 1
                else:
                     valid_metrics_accumulator.append({"sceneADE": scene_ade_single_global, "sceneFDE": scene_fde_single_global,
                                                       "minSceneADE": min_scene_ade_global, "minSceneFDE": min_scene_fde_global})

                valid_obs_mask_per_agent = feature_mask_real_total[:, :OBS_LEN].any(dim=1) 
                valid_future_mask_per_agent = gt_future_mask_real.any(dim=1) 

                observed_past_plot_dict = {
                    real_agent_ids[i]: observed_past_global_tensor[i].cpu().numpy()
                    for i in range(num_real_agents) if valid_obs_mask_per_agent[i]
                }
                all_predicted_futures_plot_list_of_dicts = []
                for pred_sample_tensor in all_pred_futures_global_samples_tensor: # These are now smoothed
                    all_predicted_futures_plot_list_of_dicts.append({
                        real_agent_ids[i]: pred_sample_tensor[i].cpu().numpy()
                        for i in range(num_real_agents) 
                    })
                best_pred_plot_dict = { # This is also from smoothed samples
                    real_agent_ids[i]: best_pred_future_global_tensor[i].cpu().numpy()
                    for i in range(num_real_agents)
                }
                
                agent_ids_for_plot = [
                    real_agent_ids[i] for i in range(num_real_agents)
                    if valid_obs_mask_per_agent[i] or valid_future_mask_per_agent[i]
                ]

                if not agent_ids_for_plot:
                     print(f"  {batch_info_str}: No agents with valid trajectory data for plotting this segment.")
                else:
                    base_scenario_id_for_plot = os.path.splitext(os.path.basename(scenario_id))[0]
                    plot_filename = f"{base_scenario_id_for_plot}_segment_{segment_start_time}_pred.png"
                    plot_filepath = os.path.join(plot_output_dir, plot_filename)
                    
                    plot_title = f"Scenario: {base_scenario_id_for_plot} | Segment Start: {segment_start_time}"
                    if APPLY_SMOOTHING:
                        plot_title += " (Smoothed)"

                    #plot_scenario_and_save_png(
                        #observed_past_plot_dict,
                        #all_predicted_futures_plot_list_of_dicts,
                        #best_pred_plot_dict,
                        #roadgraph_polylines_global_list,
                        #agent_ids_for_plot, 
                        #str(current_ego_id), 
                        #plot_filepath,
                        #title=plot_title # Pass constructed title
                        #)

                pred_k0_global = all_pred_futures_global_samples_tensor[0] # First sample (smoothed)
                for i, agent_id_str in enumerate(real_agent_ids):
                    agent_traj_global = pred_k0_global[i].cpu().numpy()
                    for t_pred_idx in range(agent_traj_global.shape[0]):
                        global_timestep = segment_start_time + OBS_LEN + t_pred_idx
                        x, y, theta = agent_traj_global[t_pred_idx, 0], agent_traj_global[t_pred_idx, 1], agent_traj_global[t_pred_idx, 2]
                        if math.isfinite(x) and math.isfinite(y) and math.isfinite(theta):
                           all_global_predictions_for_xml[scenario_id][agent_id_str][global_timestep] = (float(x), float(y), float(theta))

            except Exception as e:
                print(f"!! Error processing {batch_info_str}: {e}")
                traceback.print_exc()
                continue

    end_time = time.time()
    print(f"\n--- Inference, Metrics, and Plotting Completed ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")

    final_metrics = {"avg_sceneADE_global": 0.0, "avg_sceneFDE_global": 0.0,
                     "avg_minSceneADE_global": 0.0, "avg_minSceneFDE_global": 0.0,
                     "num_valid_segments_for_metrics": len(valid_metrics_accumulator),
                     "total_segments_processed": total_segments_processed,
                     "ade_threshold_global": ADE_THRESHOLD, "k_samples": K_SAMPLES,
                     "smoothing_applied": APPLY_SMOOTHING, 
                     "savgol_window": SAVGOL_WINDOW_LEN if APPLY_SMOOTHING else "N/A",
                     "savgol_polyorder": SAVGOL_POLYORDER if APPLY_SMOOTHING else "N/A"}
    if valid_metrics_accumulator:
        num_valid = len(valid_metrics_accumulator)
        final_metrics["avg_sceneADE_global"] = sum(m["sceneADE"] for m in valid_metrics_accumulator) / num_valid
        final_metrics["avg_sceneFDE_global"] = sum(m["sceneFDE"] for m in valid_metrics_accumulator) / num_valid
        final_metrics["avg_minSceneADE_global"] = sum(m["minSceneADE"] for m in valid_metrics_accumulator) / num_valid
        final_metrics["avg_minSceneFDE_global"] = sum(m["minSceneFDE"] for m in valid_metrics_accumulator) / num_valid

    print("\n--- Final Averaged Metrics (over segments with GLOBAL sceneADE <= threshold) ---")
    for key, val in final_metrics.items():
        if isinstance(val, float): print(f"{key.replace('_', ' ').title()}: {val:.4f}")
        else: print(f"{key.replace('_', ' ').title()}: {val}")

    metrics_output_path = os.path.join(output_dir, metrics_filename)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    try:
        with open(metrics_output_path, 'w') as f: json.dump(final_metrics, f, indent=4)
        print(f"Final metrics saved to {metrics_output_path}")
    except Exception as e: print(f"Error saving final metrics: {e}")

    print("\n--- Updating XML Files ---")
    xml_output_dir_path = os.path.join(output_dir, 'predicted_xmls') 
    os.makedirs(xml_output_dir_path, exist_ok=True)
    updated_xml_count, failed_xml_count = 0, 0
    predictions_by_filename = defaultdict(lambda: defaultdict(dict))
    scenario_id_to_filename = {sid: os.path.basename(sid) for sid in all_global_predictions_for_xml.keys()}
    for scenario_id_path, predictions_for_scenario in all_global_predictions_for_xml.items():
        base_filename = scenario_id_to_filename[scenario_id_path]
        for agent_id, timestep_data in predictions_for_scenario.items():
            predictions_by_filename[base_filename][agent_id].update(timestep_data)

    num_files_to_process = len(predictions_by_filename)
    print(f"Found predictions for {num_files_to_process} unique XML files.")
    for file_idx, (xml_filename, combined_predictions) in enumerate(predictions_by_filename.items()):
        original_xml_path = os.path.join(xml_input_dir, xml_filename)
        if not os.path.exists(original_xml_path):
            print(f"  Warning: Original XML not found at {original_xml_path}, skipping.")
            failed_xml_count +=1; continue
        if update_xml_with_predictions(original_xml_path, combined_predictions, xml_output_dir_path):
             updated_xml_count += 1
        else:
             failed_xml_count += 1

    print(f"\n--- XML Update Summary ---")
    print(f"Successfully updated/saved XMLs: {updated_xml_count}")
    print(f"Failed, skipped, or unchanged XMLs: {failed_xml_count}")
    print(f"Updated XMLs saved in: {xml_output_dir_path}")


# --- Main Execution ---
if __name__ == "__main__":
    MODEL_FILE = "./saved_models_3/model_epoch_100_+440_REAL_CARLA_poly_10_pts_fixscale_newscenes.pt"
    XML_INPUT_FOLDER = './real_mixture/cleaneddata/test'
    OUTPUT_FOLDER = 'carla_new->real_higher_dpi' # New output folder name

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    run_inference_and_save(
        model_path=MODEL_FILE,
        xml_input_dir=XML_INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        metrics_filename="final_metrics_global_with_plots_smoothed.json" # New metrics filename
    )
    print("\nScript finished.")