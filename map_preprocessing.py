import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader



def transform_segment_to_ego_perspective(obstacles, start, num_timesteps, obs_len):
    """
    Transform dynamic obstacle trajectories within a segment into ego-centric perspectives.
    
    Args:
        obstacles (list): List of {'id': str, 'traj': np.array} with traj as [time, x, y, theta].
        start (int): Starting timestep of the segment.
        num_timesteps (int): Length of the segment (e.g., 30).
        obs_len (int): Number of observation timesteps (e.g., 10).
    
    Returns:
        dict: Transformed trajectories for each ego agent in the segment.
    """
    end = start + num_timesteps - 1
    transform_time = start + obs_len - 1

    ego_candidates_ids = []
    if not obstacles:
        return {}
        
    for agent in obstacles:
        if 'traj' in agent and isinstance(agent['traj'], np.ndarray) and \
           agent['traj'].ndim == 2 and agent['traj'].shape[0] > 0 and agent['traj'].shape[1] > 0:
            # Ensure agent ID is string and time column is present for 'in' check
            if agent['traj'].shape[1] > 0 and transform_time in agent['traj'][:, 0]:
                ego_candidates_ids.append(str(agent['id']))
    
    if not ego_candidates_ids:
        return {}

    results = {}
    for ego_id_str in ego_candidates_ids: # ego_id_str is already string
        ego_agent_data = next((agent for agent in obstacles if str(agent['id']) == ego_id_str), None)
        if ego_agent_data is None or not isinstance(ego_agent_data.get('traj'), np.ndarray) or ego_agent_data['traj'].size == 0:
            continue
        
        ego_traj = ego_agent_data['traj']
        ego_state_indices = np.where(ego_traj[:, 0] == transform_time)[0]
        if ego_state_indices.size == 0:
            continue
        ego_state_idx = ego_state_indices[0]
        
        if ego_traj.shape[1] < 4: # Need x,y,theta
            continue
        ego_state = ego_traj[ego_state_idx, 1:4]
        x_ego, y_ego, theta_ego = ego_state

        cos_theta = np.cos(-theta_ego)
        sin_theta = np.sin(-theta_ego)

        # This inner dict will contain agent_id_str -> transformed_traj_array
        # for all agents from the perspective of the current ego_id_str
        transformed_agents_for_this_ego = {}
        for agent_data_inner in obstacles:
            agent_id_inner_str = str(agent_data_inner['id'])
            agent_traj_data_inner = agent_data_inner.get('traj')

            if not isinstance(agent_traj_data_inner, np.ndarray) or agent_traj_data_inner.ndim != 2 or \
               agent_traj_data_inner.shape[0] == 0 or agent_traj_data_inner.shape[1] < 4:
                # Pad with empty if necessary, or just skip if agent has no valid trajectory
                transformed_agents_for_this_ego[agent_id_inner_str] = np.zeros((num_timesteps, 4)) # All mask false
                continue

            mask = (agent_traj_data_inner[:, 0] >= start) & (agent_traj_data_inner[:, 0] <= end)
            segment_traj_inner = agent_traj_data_inner[mask]

            local_traj_inner = np.zeros((num_timesteps, 3))
            local_mask_inner = np.zeros(num_timesteps, dtype=bool)
            
            if segment_traj_inner.shape[0] > 0 and segment_traj_inner.shape[1] >= 4:
                for t, x, y, theta in segment_traj_inner[:, :4]: 
                    local_t = int(t - start)
                    if 0 <= local_t < num_timesteps:
                        local_traj_inner[local_t] = [x, y, theta]
                        local_mask_inner[local_t] = True
            
            transformed_traj_points_inner = np.zeros((num_timesteps, 4))
            transformed_traj_points_inner[:, 3] = local_mask_inner
            for local_t in range(num_timesteps):
                if local_mask_inner[local_t]:
                    x_abs, y_abs, theta_abs = local_traj_inner[local_t]
                    dx = x_abs - x_ego
                    dy = y_abs - y_ego
                    x_rel = dx * cos_theta - dy * sin_theta
                    y_rel = dx * sin_theta + dy * cos_theta
                    theta_rel = theta_abs - theta_ego
                    theta_rel = (theta_rel + np.pi) % (2 * np.pi) - np.pi
                    transformed_traj_points_inner[local_t, :3] = [x_rel, y_rel, theta_rel]
            transformed_agents_for_this_ego[agent_id_inner_str] = transformed_traj_points_inner
        results[ego_id_str] = transformed_agents_for_this_ego # Key: ego_id, Value: dict of {other_agent_id: its_traj_from_ego_pov}
    return results


def transform_points(points, ego_state):
    x_ego, y_ego, theta_ego = ego_state
    cos_theta = np.cos(-theta_ego)
    sin_theta = np.sin(-theta_ego)
    transformed = []
    if not points: return [] # Handle empty list of points
    for point in points:
        if not (isinstance(point, (tuple,list)) and len(point)==2): continue # skip malformed points
        x, y = point
        dx = x - x_ego
        dy = y - y_ego
        x_rel = dx * cos_theta - dy * sin_theta
        y_rel = dx * sin_theta + dy * cos_theta
        transformed.append((x_rel, y_rel))
    return transformed

def resample_polyline(polyline, num_points=10):
    if not polyline or len(polyline) == 0: 
        return [(0,0)] * num_points
    if len(polyline) < 2:
        return [polyline[0]] * num_points

    distances = [0.0] 
    for i in range(1, len(polyline)):
        p1 = polyline[i-1]
        p2 = polyline[i]
        if not (isinstance(p1, (tuple, list)) and len(p1) == 2 and isinstance(p1[0], (int, float)) and isinstance(p1[1], (int, float)) and
                isinstance(p2, (tuple, list)) and len(p2) == 2 and isinstance(p2[0], (int, float)) and isinstance(p2[1], (int, float))):
            return [(0,0)] * num_points 

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(distances[-1] + dist)
    
    total_length = distances[-1]
    # Handle num_points < 2 or total_length effectively zero
    if num_points < 2 or total_length < 1e-6:
        return [polyline[0]] * num_points

    step = total_length / (num_points - 1)
    resampled = []
    current_segment_idx = 0
    for i in range(num_points):
        target_dist = i * step
        found_segment = False
        for j in range(current_segment_idx, len(distances) - 1):
            # Ensure segment has non-zero length for safe division for t_interp
            if distances[j] <= target_dist <= distances[j + 1] + 1e-6 : 
                segment_len = distances[j+1] - distances[j]
                if segment_len < 1e-6: t_interp = 0.0 
                else: t_interp = (target_dist - distances[j]) / segment_len
                
                p_start, p_end = polyline[j], polyline[j+1]
                x = (1 - t_interp) * p_start[0] + t_interp * p_end[0]
                y = (1 - t_interp) * p_start[1] + t_interp * p_end[1]
                resampled.append((x, y))
                current_segment_idx = j 
                found_segment = True
                break
        if not found_segment: resampled.append(polyline[-1])
    while len(resampled) < num_points: resampled.append(polyline[-1]) 
    return resampled[:num_points]


def plot_scene(ego_id, polylines, transformed_agents_dict, save_path): # Renamed for clarity
    fig, ax = plt.subplots(figsize=(10, 10))
    for poly in polylines: # polylines is np.array [P, N_pts, 2]
        if poly.shape[0] > 0 and not np.all(poly == 0):
            ax.plot(poly[:, 0], poly[:, 1], 'k-', linewidth=1)

    # transformed_agents_dict is {agent_id_str: agent_traj_array}
    for agent_id_plot, traj_plot in transformed_agents_dict.items():
        if traj_plot.ndim == 2 and traj_plot.shape[0] > 0 and traj_plot.shape[1] == 4:
            valid_mask = traj_plot[:, 3] == 1
            if np.any(valid_mask):
                x, y = traj_plot[valid_mask, 0], traj_plot[valid_mask, 1]
                is_ego = (str(agent_id_plot) == str(ego_id))
                if is_ego:
                    ax.plot(x, y, 'r-', label='Ego')
                    ax.plot(x[0], y[0], 'ro', markersize=10)
                else:
                    ax.plot(x, y, 'b-', label='Other' if 'Other' not in ax.get_legend_handles_labels()[1] else "")
                    ax.plot(x[0], y[0], 'bo', markersize=5)
    ax.set_aspect('equal')
    ax.set_title(f"Ego {ego_id} Trajectory")
    if ax.has_data(): ax.legend()
    ax.grid(True)
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Failed to save plot {save_path}: {e}")
    plt.close(fig)

### Main Dataset Class
class MapDataset(Dataset):
    def __init__(self, xml_dir, obs_len=20, pred_len=40, max_radius=100, num_timesteps=60, num_polylines=30, num_points=10, save_plots=False, max_agents=5, single_ego=True): # Added single_ego
        self.xml_dir = xml_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_radius = max_radius
        self.num_timesteps = num_timesteps
        self.num_polylines = num_polylines
        self.num_points = num_points
        self.save_plots = save_plots
        self.max_agents = max_agents
        self.single_ego = single_ego # Store it

        if self.obs_len + self.pred_len != self.num_timesteps:
            raise ValueError("obs_len + pred_len must equal num_timesteps")

        self.data = []
        xml_files = sorted([f for f in os.listdir(self.xml_dir) if f.endswith('.xml')])
        if not xml_files: print(f"Warning: No XML files found in directory: {self.xml_dir}")

        for filename_idx, filename in enumerate(xml_files):
            file_path = os.path.join(self.xml_dir, filename)
            try:
                file_data = self._process_data(file_path)
                if file_data: self.data.extend(file_data)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                import traceback
                traceback.print_exc()

    def _process_data(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        lanelets = []
        for lanelet_elem in root.findall('lanelet'):
            lane_id = str(lanelet_elem.get('id'))
            lane_data = {'id': lane_id}
            for bound_name in ['leftBound', 'rightBound']:
                bound_elem = lanelet_elem.find(bound_name)
                if bound_elem is not None:
                    points = []
                    for point_elem in bound_elem.findall('point'):
                        x_elem, y_elem = point_elem.find('x'), point_elem.find('y')
                        if x_elem is not None and x_elem.text is not None and y_elem is not None and y_elem.text is not None:
                            try: points.append((float(x_elem.text), float(y_elem.text)))
                            except ValueError: continue
                    if points: lane_data[bound_name.replace('Bound', '')] = points
            if 'left' in lane_data and 'right' in lane_data:
                left_pts, right_pts = lane_data['left'], lane_data['right']
                if len(left_pts) == len(right_pts) and len(left_pts) > 0:
                    center_pts = [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in zip(left_pts, right_pts)]
                    lane_data['center'] = center_pts
                    lanelets.append(lane_data)

        obstacles = []
        for dob_elem in root.findall('dynamicObstacle'):
            obs_id = str(dob_elem.get('id'))
            current_traj_points = []
            init_state_elem = dob_elem.find('initialState')
            if init_state_elem:
                pos_e, orient_e, time_e = init_state_elem.find('position/point'), init_state_elem.find('orientation/exact'), init_state_elem.find('time/exact')
                if pos_e is not None and orient_e is not None:
                    try:
                        x_s, y_s, th_s = pos_e.find('x').text, pos_e.find('y').text, orient_e.text
                        t_s = time_e.text if time_e is not None and time_e.text is not None else "0"
                        if all(s is not None for s in [x_s, y_s, th_s, t_s]):
                            current_traj_points.append([float(t_s), float(x_s), float(y_s), float(th_s)])
                    except (ValueError, AttributeError): pass
            traj_elem = dob_elem.find('trajectory')
            if traj_elem is not None:
                for state_elem in traj_elem.findall('state'):
                    pos_e, orient_e, time_e = state_elem.find('position/point'), state_elem.find('orientation/exact'), state_elem.find('time/exact')
                    if pos_e is not None and orient_e is not None and time_e is not None:
                        try:
                            x_s, y_s, th_s, t_s = pos_e.find('x').text, pos_e.find('y').text, orient_e.text, time_e.text
                            if all(s is not None for s in [x_s, y_s, th_s, t_s]):
                                current_traj_points.append([float(t_s), float(x_s), float(y_s), float(th_s)])
                        except (ValueError, AttributeError): pass
            if current_traj_points:
                current_traj_points.sort(key=lambda p: p[0])
                unique_traj_pts = []
                seen_times = set()
                for pt in current_traj_points:
                    if pt[0] not in seen_times: unique_traj_pts.append(pt); seen_times.add(pt[0])
                if unique_traj_pts: obstacles.append({'id': obs_id, 'traj': np.array(unique_traj_pts, dtype=np.float64)})

        if not obstacles: T = 0
        else:
            max_t_overall = -1.0; valid_traj_found = False
            for agent in obstacles:
                agent_traj_data = agent.get('traj')
                if isinstance(agent_traj_data, np.ndarray) and agent_traj_data.ndim == 2 and \
                   agent_traj_data.shape[0] > 0 and agent_traj_data.shape[1] > 0:
                    try:
                        current_max_t = np.max(agent_traj_data[:, 0])
                        if current_max_t > max_t_overall: max_t_overall = current_max_t
                        valid_traj_found = True
                    except IndexError: pass
            T = int(np.ceil(max_t_overall)) + 1 if valid_traj_found else 0
        
        stride = self.num_timesteps
        N = T // self.num_timesteps
        data_list = []
        # print(f"DEBUG: File {os.path.basename(file_path)}, T={T}, N={N}")

        # Corrected loop for segments
        for i in range(N): # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< USE THIS
            start = i * stride
            # The `transformed_results` dictionary has:
            # keys: string ego_ids that are valid candidates for this segment
            # values: dictionaries of {string agent_id: transformed_traj_numpy_array_for_that_agent_from_ego_pov}
            transformed_results_per_ego_candidate = transform_segment_to_ego_perspective(obstacles, start, self.num_timesteps, self.obs_len)

            if not transformed_results_per_ego_candidate:
                # print(f"DEBUG: File {os.path.basename(file_path)}, seg {i}, start {start}: No ego candidates returned.")
                continue

            # --- Single Ego Selection Logic (Simplified from WV's implicit behavior) ---
            egos_to_process_in_segment = []
            if self.single_ego:
                # Pick the first valid ego candidate alphabetically
                sorted_candidate_ego_ids = sorted(list(transformed_results_per_ego_candidate.keys()))
                for cand_ego_id in sorted_candidate_ego_ids:
                    # Check if this candidate ego itself has valid data in its own transformed view
                    # The value for cand_ego_id in transformed_results_per_ego_candidate is a dict of *other* agents' trajs
                    # We need the ego's own trajectory from that dict.
                    all_agent_trajs_from_this_ego_pov = transformed_results_per_ego_candidate[cand_ego_id]
                    if cand_ego_id in all_agent_trajs_from_this_ego_pov and \
                       np.any(all_agent_trajs_from_this_ego_pov[cand_ego_id][:, 3] == 1):
                        egos_to_process_in_segment.append(cand_ego_id)
                        break # Found one valid ego, stop.
            else: # Process all valid ego candidates
                for cand_ego_id, all_agent_trajs_from_this_ego_pov in transformed_results_per_ego_candidate.items():
                    if cand_ego_id in all_agent_trajs_from_this_ego_pov and \
                       np.any(all_agent_trajs_from_this_ego_pov[cand_ego_id][:, 3] == 1):
                        egos_to_process_in_segment.append(cand_ego_id)
            # --- End Single Ego Selection ---

            if not egos_to_process_in_segment:
                # print(f"DEBUG: File {os.path.basename(file_path)}, seg {i}, start {start}: No valid egos to process after single_ego/validity check.")
                continue
            
            # Loop through the selected ego(s) for this segment
            for ego_id_str in egos_to_process_in_segment: # ego_id_str is string
                # `transformed_agents_from_ego_pov` is the dict: {agent_id: agent_traj_array} for this specific ego
                transformed_agents_from_ego_pov = transformed_results_per_ego_candidate[ego_id_str]

                # This check was already implicitly done to populate egos_to_process_in_segment
                # if not np.any(transformed_agents_from_ego_pov[ego_id_str][:, 3] == 1):
                #     continue

                # Get original global state of this ego_id_str at transform_time
                current_transform_time = start + self.obs_len - 1
                ego_original_traj_data = next((obs['traj'] for obs in obstacles if str(obs['id']) == ego_id_str), None)
                if ego_original_traj_data is None: continue
                
                ego_state_indices_global = np.where(ego_original_traj_data[:, 0] == current_transform_time)[0]
                if ego_state_indices_global.size == 0: continue
                ego_state_global_np = ego_original_traj_data[ego_state_indices_global[0], 1:4]


                transformed_polylines = []
                distances = []
                if lanelets:
                    for lanelet in lanelets:
                        if 'center' not in lanelet or not lanelet['center']: continue
                        center_abs = lanelet['center']
                        if not center_abs: continue
                        trans_center_rel = transform_points(center_abs, ego_state_global_np)
                        if not trans_center_rel or not trans_center_rel[0]: continue
                        x0_r, y0_r = trans_center_rel[0]
                        dist = np.sqrt(x0_r**2 + y0_r**2)
                        transformed_polylines.append(trans_center_rel)
                        distances.append((dist, len(transformed_polylines) - 1))
                
                distances.sort()
                selected_indices = [idx for _, idx in distances[:self.num_polylines]]
                selected_polylines = [transformed_polylines[i] for i in selected_indices]
                polylines_resampled = [resample_polyline(poly, self.num_points) for poly in selected_polylines]
                while len(polylines_resampled) < self.num_polylines:
                    polylines_resampled.append([(0, 0)] * self.num_points)
                polylines_array = np.array(polylines_resampled, dtype=np.float32)
                polylines_tensor = torch.tensor(polylines_array, dtype=torch.float32)
                num_real_polylines = len(selected_polylines)
                polyline_mask = torch.ones(self.num_polylines, dtype=torch.bool)
                if num_real_polylines < self.num_polylines:
                    polyline_mask[num_real_polylines:] = False

                # Agent handling (using transformed_agents_from_ego_pov)
                # `agent_order` will store string IDs
                agent_order = [ego_id_str] # Ego first
                # Get all agent IDs present in this ego's perspective, sort them
                all_other_agent_ids_in_view = sorted([str(aid) for aid in transformed_agents_from_ego_pov.keys() if str(aid) != ego_id_str])

                for other_aid_str in all_other_agent_ids_in_view:
                    other_agent_traj_data = transformed_agents_from_ego_pov.get(other_aid_str)
                    if other_agent_traj_data is not None and np.any(other_agent_traj_data[:, 3] == 1):
                        agent_order.append(other_aid_str)
                    if len(agent_order) >= self.max_agents: break
                
                num_real_agents = len(agent_order)
                if num_real_agents == 0: continue # Should not happen if ego is valid

                # Stack trajectories based on agent_order
                feature_tensor_list = [transformed_agents_from_ego_pov[aid_s] for aid_s in agent_order]
                feature_tensor_stacked = np.stack(feature_tensor_list, axis=0)

                if num_real_agents < self.max_agents:
                    padding = np.zeros((self.max_agents - num_real_agents, self.num_timesteps, 4), dtype=np.float32)
                    feature_tensor_final = np.concatenate([feature_tensor_stacked, padding], axis=0)
                else:
                    feature_tensor_final = feature_tensor_stacked

                feature_data = torch.tensor(feature_tensor_final[:, :, :3], dtype=torch.float32)
                feature_mask = torch.tensor(feature_tensor_final[:, :, 3].astype(bool), dtype=torch.bool)
                observed_all = feature_tensor_final[:, :self.obs_len, :]
                observed_data = torch.tensor(observed_all[:, :, :3], dtype=torch.float32)
                observed_mask = torch.tensor(observed_all[:, :, 3].astype(bool), dtype=torch.bool)
                ground_truth_all = feature_tensor_final[:, self.obs_len:, :]
                ground_truth_data = torch.tensor(ground_truth_all[:, :, :3], dtype=torch.float32)
                ground_truth_mask = torch.tensor(ground_truth_all[:, :, 3].astype(bool), dtype=torch.bool)

                valid_features = feature_data[feature_mask]
                if valid_features.numel() > 0 and valid_features.shape[0] > 1 :
                    scene_mean = torch.mean(valid_features, dim=0)
                    scene_std_raw = torch.std(valid_features, dim=0)
                    scene_std_raw = torch.where(scene_std_raw > 1e-6, scene_std_raw, torch.ones_like(scene_std_raw) * 1e-6)
                else:
                    scene_mean = torch.zeros(3, dtype=torch.float32)
                    scene_std_raw = torch.ones(3, dtype=torch.float32)

                MIN_STD_POS, MIN_STD_THETA = 1.0, 0.05
                scene_std_clamped = torch.stack([
                    torch.clamp(scene_std_raw[0], min=MIN_STD_POS),
                    torch.clamp(scene_std_raw[1], min=MIN_STD_POS),
                    torch.clamp(scene_std_raw[2], min=MIN_STD_THETA)
                ])
                scale_factor = 0.5 / scene_std_clamped

                feature_data *= scale_factor[None, None, :]
                observed_data *= scale_factor[None, None, :]
                ground_truth_data *= scale_factor[None, None, :]
                scale_xy = scale_factor[:2]
                if polylines_tensor.numel() > 0: polylines_tensor *= scale_xy[None, None, :]

                if self.save_plots:
                    file_base = os.path.splitext(os.path.basename(file_path))[0]
                    plot_save_path = os.path.join(os.getcwd(), f"{file_base}_ego_{ego_id_str}_seg_{i}_debug.png")
                    # For plotting, use the unscaled local data for agents and polylines
                    unscaled_polylines_for_plot = polylines_array.copy()
                    if unscaled_polylines_for_plot.size > 0 and scale_xy.numel() >0:
                        unscaled_polylines_for_plot[...,0] /= scale_xy[0].item()
                        unscaled_polylines_for_plot[...,1] /= scale_xy[1].item()
                    plot_scene(ego_id_str, unscaled_polylines_for_plot, transformed_agents_from_ego_pov, plot_save_path)


                data_item = {
                    "ego_id": ego_id_str, # String
                    "feature_tensor": feature_data, "feature_mask": feature_mask,
                    "polylines": polylines_tensor, "polyline_mask": polyline_mask,
                    "observed": observed_data, "observed_mask": observed_mask,
                    "ground_truth": ground_truth_data, "ground_truth_mask": ground_truth_mask,
                    "scene_mean": scene_mean, "scene_std": scene_std_clamped, # Store the clamped std used for scaling
                    "scenario_id": file_path, "segment_start": start,
                    "ego_state_global": torch.tensor(ego_state_global_np, dtype=torch.float32),
                    "agent_ids": agent_order # List of strings
                }
                data_list.append(data_item)
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["ego_id"], item["feature_tensor"], item["feature_mask"],
            item["polylines"], item["polyline_mask"],
            item["observed"], item["observed_mask"],
            item["ground_truth"], item["ground_truth_mask"],
            item["scene_mean"], item["scene_std"],
            item["scenario_id"], item["segment_start"],
            item["ego_state_global"], item["agent_ids"]
        )

### Example Usage
if __name__ == "__main__":
    dummy_xml_dir = 'dummy_created_scenarios_merged' # New name to avoid conflict
    os.makedirs(dummy_xml_dir, exist_ok=True)
    dummy_xml_path = os.path.join(dummy_xml_dir, 'dummy_scenario_merged_01.xml')

    root = ET.Element("snapshot")
    lanelet_elem = ET.SubElement(root, "lanelet", id="lane1")
    lb, rb = ET.SubElement(lanelet_elem, "leftBound"), ET.SubElement(lanelet_elem, "rightBound")
    for k_pt in range(5):
        ET.SubElement(ET.SubElement(lb, "point"), "x").text = str(k_pt * 10.0)
        ET.SubElement(ET.SubElement(lb, "point"), "y").text = str(5.0)
        ET.SubElement(ET.SubElement(rb, "point"), "x").text = str(k_pt * 10.0)
        ET.SubElement(ET.SubElement(rb, "point"), "y").text = str(-5.0)
    
    # Agent 1 (ego candidate)
    dob1 = ET.SubElement(root, "dynamicObstacle", id="ego_agent_1")
    init1 = ET.SubElement(dob1, "initialState")
    ET.SubElement(ET.SubElement(ET.SubElement(init1, "position"), "point"), "x").text = "0"
    ET.SubElement(ET.SubElement(ET.SubElement(init1, "position"), "point"), "y").text = "0"
    ET.SubElement(ET.SubElement(init1, "orientation"), "exact").text = "0"
    ET.SubElement(ET.SubElement(init1, "time"), "exact").text = "0"
    traj1 = ET.SubElement(dob1, "trajectory")
    for t in range(1, 50): # Long enough for N=1 with num_timesteps=30
        s = ET.SubElement(traj1, "state")
        ET.SubElement(ET.SubElement(ET.SubElement(s, "position"), "point"), "x").text = str(t * 1.0)
        ET.SubElement(ET.SubElement(ET.SubElement(s, "position"), "point"), "y").text = "0"
        ET.SubElement(ET.SubElement(s, "orientation"), "exact").text = "0"
        ET.SubElement(ET.SubElement(s, "time"), "exact").text = str(t)

    # Agent 2 (another ego candidate, different trajectory)
    dob2 = ET.SubElement(root, "dynamicObstacle", id="ego_agent_2")
    init2 = ET.SubElement(dob2, "initialState")
    ET.SubElement(ET.SubElement(ET.SubElement(init2, "position"), "point"), "x").text = "2"
    ET.SubElement(ET.SubElement(ET.SubElement(init2, "position"), "point"), "y").text = "1"
    ET.SubElement(ET.SubElement(init2, "orientation"), "exact").text = "0.2"
    ET.SubElement(ET.SubElement(init2, "time"), "exact").text = "0"
    traj2 = ET.SubElement(dob2, "trajectory")
    for t in range(1, 50):
        s = ET.SubElement(traj2, "state")
        ET.SubElement(ET.SubElement(ET.SubElement(s, "position"), "point"), "x").text = str(2 + t * 0.8)
        ET.SubElement(ET.SubElement(ET.SubElement(s, "position"), "point"), "y").text = str(1 + t * 0.1)
        ET.SubElement(ET.SubElement(s, "orientation"), "exact").text = "0.2"
        ET.SubElement(ET.SubElement(s, "time"), "exact").text = str(t)

    # Agent 3 (just an other agent)
    dob3 = ET.SubElement(root, "dynamicObstacle", id="other_agent_3")
    init3 = ET.SubElement(dob3, "initialState")
    ET.SubElement(ET.SubElement(ET.SubElement(init3, "position"), "point"), "x").text = "-5"
    ET.SubElement(ET.SubElement(ET.SubElement(init3, "position"), "point"), "y").text = "-2"
    ET.SubElement(ET.SubElement(init3, "orientation"), "exact").text = "1.0"
    ET.SubElement(ET.SubElement(init3, "time"), "exact").text = "0"
    traj3 = ET.SubElement(dob3, "trajectory")
    for t in range(1, 50):
        s = ET.SubElement(traj3, "state")
        ET.SubElement(ET.SubElement(ET.SubElement(s, "position"), "point"), "x").text = str(-5 + t * 0.5)
        ET.SubElement(ET.SubElement(ET.SubElement(s, "position"), "point"), "y").text = str(-2 - t * 0.2)
        ET.SubElement(ET.SubElement(s, "orientation"), "exact").text = "1.0"
        ET.SubElement(ET.SubElement(s, "time"), "exact").text = str(t)

    tree = ET.ElementTree(root)
    if hasattr(ET, 'indent'): ET.indent(tree, space="  ")
    tree.write(dummy_xml_path, encoding='utf-8', xml_declaration=True)
    print(f"Created/Recreated dummy XML: {dummy_xml_path} with 3 agents.")
    xml_directory_to_use = dummy_xml_dir
    # else: # Use your actual dir if dummy exists and you don't want to recreate
    #     xml_directory_to_use = 'created_scenarios' 
    #     print(f"Using existing directory: {xml_directory_to_use}")


    print(f"\n--- Initializing MapDataset with single_ego=False ---")
    dataset_multi_ego = MapDataset(xml_dir=xml_directory_to_use, obs_len=10, pred_len=20, num_timesteps=30, 
                                   num_polylines=10, num_points=10, save_plots=True, max_agents=3, single_ego=False)
    print(f"Dataset (multi-ego) loaded with {len(dataset_multi_ego)} segments.")
    if len(dataset_multi_ego) > 0:
        print("Multi-ego segments (ego_id, segment_start):")
        for item_idx in range(min(len(dataset_multi_ego), 5)): # Print first 5
            item = dataset_multi_ego.data[item_idx]
            print(f"  Item {item_idx}: Ego ID: {item['ego_id']}, Start: {item['segment_start']}, Agents: {item['agent_ids']}")
    
    print(f"\n--- Initializing MapDataset with single_ego=True ---")
    dataset_single_ego = MapDataset(xml_dir=xml_directory_to_use, obs_len=10, pred_len=20, num_timesteps=30,
                                   num_polylines=10, num_points=10, save_plots=True, max_agents=3, single_ego=True)
    print(f"Dataset (single-ego) loaded with {len(dataset_single_ego)} segments.")
    if len(dataset_single_ego) > 0:
        print("Single-ego segments (ego_id, segment_start):")
        for item_idx in range(min(len(dataset_single_ego), 5)): # Print first 5
            item = dataset_single_ego.data[item_idx]
            print(f"  Item {item_idx}: Ego ID: {item['ego_id']}, Start: {item['segment_start']}, Agents: {item['agent_ids']}")
            
    print("\nScript finished.")
