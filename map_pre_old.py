import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader

### Helper Functions

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
    transform_time = start + obs_len - 1  # e.g., 9 if start=0, obs_len=10

    # Identify agents with a state at transform_time
    ego_candidates = [agent['id'] for agent in obstacles if transform_time in agent['traj'][:, 0]]
    if not ego_candidates:
        return {}

    results = {}
    for ego_id in ego_candidates:
        # Get ego state at transform_time
        ego_traj = next(agent['traj'] for agent in obstacles if agent['id'] == ego_id)
        ego_state_idx = np.where(ego_traj[:, 0] == transform_time)[0][0]
        ego_state = ego_traj[ego_state_idx, 1:4]  # [x, y, theta]
        x_ego, y_ego, theta_ego = ego_state

        # Precompute rotation
        cos_theta = np.cos(-theta_ego)
        sin_theta = np.sin(-theta_ego)

        transformed_agents = {}
        for agent in obstacles:
            agent_id = agent['id']
            agent_traj = agent['traj']
            mask = (agent_traj[:, 0] >= start) & (agent_traj[:, 0] <= end)
            segment_traj = agent_traj[mask]

            # Pad trajectory to num_timesteps with local indices [0, num_timesteps-1]
            local_traj = np.zeros((num_timesteps, 3))
            local_mask = np.zeros(num_timesteps, dtype=bool)
            for t, x, y, theta in segment_traj:
                local_t = int(t - start)
                if 0 <= local_t < num_timesteps:
                    local_traj[local_t] = [x, y, theta]
                    local_mask[local_t] = True

            # Transform trajectory
            transformed_traj = np.zeros((num_timesteps, 4))
            transformed_traj[:, 3] = local_mask
            for local_t in range(num_timesteps):
                if local_mask[local_t]:
                    x_abs, y_abs, theta_abs = local_traj[local_t]
                    dx = x_abs - x_ego
                    dy = y_abs - y_ego
                    x_rel = dx * cos_theta - dy * sin_theta
                    y_rel = dx * sin_theta + dy * cos_theta
                    theta_rel = theta_abs - theta_ego
                    theta_rel = (theta_rel + np.pi) % (2 * np.pi) - np.pi
                    transformed_traj[local_t, :3] = [x_rel, y_rel, theta_rel]
            transformed_agents[agent_id] = transformed_traj

        results[ego_id] = transformed_agents
    return results

def transform_points(points, ego_state):
    """Transform points to ego-centric coordinates."""
    x_ego, y_ego, theta_ego = ego_state
    cos_theta = np.cos(-theta_ego)
    sin_theta = np.sin(-theta_ego)
    transformed = []
    for x, y in points:
        dx = x - x_ego
        dy = y - y_ego
        x_rel = dx * cos_theta - dy * sin_theta
        y_rel = dx * sin_theta + dy * cos_theta
        transformed.append((x_rel, y_rel))
    return transformed

def resample_polyline(polyline, num_points=10):
    """Resample a polyline to a fixed number of points."""
    if len(polyline) < 2:
        return [polyline[0]] * num_points if polyline else [(0, 0)] * num_points
    distances = [0]
    for i in range(1, len(polyline)):
        dx = polyline[i][0] - polyline[i-1][0]
        dy = polyline[i][1] - polyline[i-1][1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(distances[-1] + dist)
    total_length = distances[-1]
    if total_length == 0:
        return [polyline[0]] * num_points
    step = total_length / (num_points - 1)
    resampled = []
    for i in range(num_points):
        target_dist = i * step
        if target_dist >= total_length:
            resampled.append(polyline[-1])
            continue
        for j in range(len(distances) - 1):
            if distances[j] <= target_dist < distances[j + 1]:
                t = (target_dist - distances[j]) / (distances[j + 1] - distances[j])
                x = (1 - t) * polyline[j][0] + t * polyline[j + 1][0]
                y = (1 - t) * polyline[j][1] + t * polyline[j + 1][1]
                resampled.append((x, y))
                break
    return resampled

def plot_scene(ego_id, polylines, transformed_agents, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    for poly in polylines:
        if not np.all(poly == 0):
            ax.plot(poly[:, 0], poly[:, 1], 'k-', linewidth=1)
    for agent_id, traj in transformed_agents.items():
        pass
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.xlabel('X (meters)', fontsize=22.5)
    plt.ylabel('Y (meters)', fontsize=22.5)
    try:
        plt.savefig(save_path)
        print(f"Successfully saved plot to {save_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    plt.close()
### Main Dataset Class

class MapDataset(Dataset):
    def __init__(self, xml_dir, obs_len=20, pred_len=40, max_radius=100, num_timesteps=60, num_polylines=30, num_points=10, save_plots=False, max_agents=5):
        self.xml_dir = xml_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_radius = max_radius
        self.num_timesteps = num_timesteps
        self.num_polylines = num_polylines
        self.num_points = num_points
        self.save_plots = save_plots
        self.max_agents = max_agents

        if self.obs_len + self.pred_len != self.num_timesteps:
            raise ValueError("obs_len + pred_len must equal num_timesteps")

        self.data = []
        for filename in os.listdir(self.xml_dir):
            if filename.endswith('.xml'):
                file_path = os.path.join(self.xml_dir, filename)
                file_data = self._process_data(file_path)
                print(f"{filename}//{len(os.listdir(self.xml_dir))}")
                self.data.extend(file_data)

    def _process_data(self, file_path):
        """Process a single XML file into multiple segments."""
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract lanelet centerlines (static across segments)
        lanelets = []
        for lanelet in root.findall('lanelet'):
            lane_id = lanelet.get('id')
            lane_data = {'id': lane_id}
            for bound in ['leftBound', 'rightBound']:
                bound_elem = lanelet.find(bound)
                if bound_elem is not None:
                    points = []
                    for point in bound_elem.findall('point'):
                        x_elem = point.find('x')
                        y_elem = point.find('y')
                        if x_elem is not None and y_elem is not None:
                            try:
                                x, y = float(x_elem.text), float(y_elem.text)
                                points.append((x, y))
                            except ValueError:
                                continue
                    if points:
                        lane_data[bound.replace('Bound', '')] = points
            if 'left' in lane_data and 'right' in lane_data:
                left, right = lane_data['left'], lane_data['right']
                if len(left) == len(right):
                    center = [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in zip(left, right)]
                    lane_data['center'] = center
                else:
                    print("skipped one")
                    continue
                lanelets.append(lane_data)

        # Parse all trajectories to determine total timesteps T
        obstacles = []
        for dob in root.findall('dynamicObstacle'):
            #print(dob)
            obs_id = dob.get('id')
            traj = []
            init_state = dob.find('initialState')
            if init_state:
                pos_elem = init_state.find('position/point')
                orient_elem = init_state.find('orientation/exact')
                time_elem = init_state.find('time/exact')
                if pos_elem is not None and orient_elem is not None:
                    try:
                        x = float(pos_elem.find('x').text or 0.0)
                        y = float(pos_elem.find('y').text or 0.0)
                        theta = float(orient_elem.text or 0.0)
                        time = float(time_elem.text) if time_elem is not None else 0
                        traj.append([time, x, y, theta])
                    except ValueError:
                        print(f"Could not parse initialState for obstacle {obs_id}")
            traj_elem = dob.find('trajectory')
            if traj_elem is not None:
                for state in traj_elem.findall('state'):
                    pos_elem = state.find('position/point')
                    orient_elem = state.find('orientation/exact')
                    time_elem = state.find('time/exact')
                    if pos_elem is not None and orient_elem is not None and time_elem is not None:
                        try:
                            x = float(pos_elem.find('x').text or 0.0)
                            y = float(pos_elem.find('y').text or 0.0)
                            theta = float(orient_elem.text or 0.0)
                            time = float(time_elem.text)
                            traj.append([time, x, y, theta])
                        except ValueError:
                            continue
            if traj:
                obstacles.append({'id': obs_id, 'traj': np.array(traj)})

        # Determine total timesteps T
        T = int(max(max(agent['traj'][:, 0]) for agent in obstacles)) + 1 if obstacles else 0
        #if T < self.num_timesteps:
            #print(f"Warning: Scenario {file_path} has only {T} timesteps, less than {self.num_timesteps}")
            #return []  # Skip if too short, or handle with padding if desired

        # Compute number of non-overlapping segments
        stride = self.num_timesteps  # Non-overlapping segments as per query
        N = T // self.num_timesteps  # Floor division for full segments only
        data_list = []

        # Process each segment
        for i in range(1, N):
            start = i * stride
            transformed_results = transform_segment_to_ego_perspective(obstacles, start, self.num_timesteps, self.obs_len)

            for ego_id, transformed_agents in transformed_results.items():
                if not np.any(transformed_agents[ego_id][:, 3] == 1):
                    continue

                # Get ego state at transform_time for polyline transformation
                transform_time = start + self.obs_len - 1
                ego_traj = next(agent['traj'] for agent in obstacles if agent['id'] == ego_id)
                ego_state_idx = np.where(ego_traj[:, 0] == transform_time)[0][0]
                ego_state = ego_traj[ego_state_idx, 1:4]

                # Transform and select polylines
                transformed_polylines = []
                distances = []
                for lanelet in lanelets:
                    center = lanelet['center']
                    trans_center = transform_points(center, ego_state)
                    x0, y0 = trans_center[0]
                    dist = np.sqrt(x0**2 + y0**2)
                    transformed_polylines.append(trans_center)
                    distances.append((dist, len(transformed_polylines) - 1))

                distances.sort()
                selected_indices = [idx for _, idx in distances[:self.num_polylines]]
                selected_polylines = [transformed_polylines[i] for i in selected_indices]
                polylines_resampled = [resample_polyline(poly, self.num_points) for poly in selected_polylines]
                while len(polylines_resampled) < self.num_polylines:
                    polylines_resampled.append([(0, 0)] * self.num_points)
                polylines_array = np.array(polylines_resampled)
                polylines_tensor = torch.tensor(polylines_array, dtype=torch.float32)

                num_real_polylines = len(selected_polylines)
                polyline_mask = torch.ones(self.num_polylines, dtype=torch.bool)
                if num_real_polylines < self.num_polylines:
                    polyline_mask[num_real_polylines:] = False

                # Agent handling
                agent_ids = sorted(transformed_agents.keys())
                agent_order = [ego_id]
                for aid in agent_ids:
                    if aid != ego_id:
                        traj = transformed_agents[aid]
                        if np.any((traj[:, 3] == 1)):
                            agent_order.append(aid)
                        if len(agent_order) == self.max_agents:
                            break

                num_real_agents = len(agent_order)
                if num_real_agents == 0:
                    continue

                feature_tensor = np.stack([transformed_agents[aid] for aid in agent_order], axis=0)
                if num_real_agents < self.max_agents:
                    padding = np.zeros((self.max_agents - num_real_agents, self.num_timesteps, 4), dtype=np.float32)
                    feature_tensor = np.concatenate([feature_tensor, padding], axis=0)

                # Split into data and mask
                feature_data = feature_tensor[:, :, :3]
                feature_mask = feature_tensor[:, :, 3].astype(bool)
                feature_data = torch.tensor(feature_data, dtype=torch.float32)
                feature_mask = torch.tensor(feature_mask, dtype=torch.bool)

                observed = feature_tensor[:, :self.obs_len, :]
                observed_data = observed[:, :, :3]
                observed_mask = observed[:, :, 3].astype(bool)
                observed_data = torch.tensor(observed_data, dtype=torch.float32)
                observed_mask = torch.tensor(observed_mask, dtype=torch.bool)

                ground_truth = feature_tensor[:, self.obs_len:, :]
                ground_truth_data = ground_truth[:, :, :3]
                ground_truth_mask = ground_truth[:, :, 3].astype(bool)
                ground_truth_data = torch.tensor(ground_truth_data, dtype=torch.float32)
                ground_truth_mask = torch.tensor(ground_truth_mask, dtype=torch.bool)

                # Compute per-segment statistics
                valid_features = feature_data[feature_mask]
                if valid_features.shape[0] > 1:
                    scene_mean = torch.mean(valid_features, dim=0)
                    scene_std = torch.std(valid_features, dim=0)
                    scene_std = torch.where(scene_std > 0, scene_std, torch.ones_like(scene_std) * 1e-6)
                else:
                    scene_mean = torch.zeros(3, dtype=torch.float32)
                    scene_std = torch.ones(3, dtype=torch.float32)

                MIN_STD_POS_FOR_SCALING = 1.0  # Clamp for x and y std (positions)
                MIN_STD_THETA_FOR_SCALING = 0.05 # Clamp for theta std (headings)

                # Assuming scene_std is a tensor like: [std_x, std_y, std_theta]
                # scene_std was previously `safe_scaling_scene_std` which floored at 1e-6
                
                std_x_clamped = torch.clamp(scene_std[0], min=MIN_STD_POS_FOR_SCALING)
                std_y_clamped = torch.clamp(scene_std[1], min=MIN_STD_POS_FOR_SCALING)
                std_theta_clamped = torch.clamp(scene_std[2], min=MIN_STD_THETA_FOR_SCALING)

                # Reconstruct the clamped standard deviation tensor
                scene_std = torch.stack([
                    std_x_clamped,
                    std_y_clamped,
                    std_theta_clamped
                ])
                # --- END: Integration of Clamping ---

                # Now calculate scale_factor using the robustly clamped standard deviation
                scale_factor = 0.5 / scene_std # Use the new clamped std

                # The rest of your scaling logic remains the same,
                # it just uses the new, more stable scale_factor.
                feature_data *= scale_factor[None, None, :]
                observed_data *= scale_factor[None, None, :] # Make sure observed_data is defined before this
                ground_truth_data *= scale_factor[None, None, :] # Make sure ground_truth_data is defined
                
                scale_xy = scale_factor[:2] # This will use the potentially clamped x, y scale factors
                polylines_tensor *= scale_xy[None, None, :]

                # Save debug plot with segment index
                if self.save_plots:
                    file_base = os.path.splitext(os.path.basename(file_path))[0]
                    save_path = os.path.join(os.getcwd(), f"{file_base}_ego_{ego_id}_seg_{i}_debug.png")
                    plot_scene(ego_id, polylines_array, transformed_agents, save_path)

                data_item = {
                    "ego_id": ego_id,
                    "feature_tensor": feature_data,
                    "feature_mask": feature_mask,
                    "polylines": polylines_tensor,
                    "polyline_mask": polyline_mask,
                    "observed": observed_data,
                    "observed_mask": observed_mask,
                    "ground_truth": ground_truth_data,
                    "ground_truth_mask": ground_truth_mask,
                    "scene_mean": scene_mean,
                    "scene_std": scene_std
                }
                data_list.append(data_item)

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["ego_id"],
            item["feature_tensor"],
            item["feature_mask"],
            item["polylines"],
            item["polyline_mask"],
            item["observed"],
            item["observed_mask"],
            item["ground_truth"],
            item["ground_truth_mask"],
            item["scene_mean"],
            item["scene_std"]
        )

### Example Usage
if __name__ == "__main__":
    dataset = MapDataset(
        xml_dir='/Users/brikelkeputa/Downloads/Master-Thesis-main/carla_new->real_higher_dpi/predicted_xmls',
        obs_len=10,
        pred_len=20,
        max_radius=100,
        num_timesteps=30,
        num_polylines=500,
        num_points=10,
        save_plots=True,
        max_agents=32
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    for batch in dataloader:
        ego_ids, feature_tensors, feature_masks, polylines, polyline_masks, observed, observed_masks, ground_truth, ground_truth_masks, means, stds = batch
        #print(means)
        break