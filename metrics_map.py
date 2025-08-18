import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import os, random, shutil
from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time as timing 
import traceback 
import itertools 
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely.ops import nearest_points
import ot
SHAPELY_AVAILABLE = True
OT_AVAILABLE = True

# --- Configuration ---
# <<< --- SET YOUR XML DIRECTORIES HERE --- >>>
XML_DIRECTORY_1 = 'carla_new->real_with_plots_overfit_smoothed/predicted_xmls'  # Replace with the path to your first dataset folder
original_xml_dir = './real_mixture/cleaneddata/train'

# New directory with only 150 randomly sampled XMLs
sampled_xml_dir = './real_mixture/cleaneddata/train_sampled_150'
# Use this as your new XML_DIRECTORY_2
XML_DIRECTORY_2 = sampled_xml_dir

#XML_DIRECTORY_2 = './real_mixture/cleaneddata/test'  # Replace with the path to your second dataset folder
# Add more directories here if needed
#XML_DIRECTORY_3 = './pittsburgh_split/cleaneddata/test'

# <<< --- SET YOUR OUTPUT DIRECTORY HERE --- >>>
# Updated output dir name reflecting TTC label change
OUTPUT_DIRECTORY = 'output_gap_synth_vs_real_random_sample' # Indicate >90 label

# <<< --- PLOTTING OPTIONS --- >>>
SAVE_DPI = 300 # DPI for saved figures
HIST_COLORS = ['deepskyblue','palegreen', 'saddlebrown'] # Colors for datasets in histograms
SHOW_HISTOGRAM_LEGEND = True # Set to False to explicitly remove legends
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (7, 7) # Width, Height in inches
# --- Metric Calculation Parameters ---
NUM_BINS = 50 # Target number of bins for metrics with fixed ranges
TTC_SPLIT_POINT = 90.0 # Upper bound for regular TTC bins (e.g., 0-90)
# TTC_FINAL_BIN_START_LABEL is no longer needed with the ">90" label
NUM_BINS_TTC_FINITE = 45 # Number of bins to use specifically for the 0-TTC_SPLIT_POINT range
MIN_DT_SINGLE_STEP = 1e-9
MIN_DT_TWO_STEP = 1e-9
TIME_MATCH_TOLERANCE = 0.05 # Max time difference (seconds) to consider states simultaneous
COLLISION_DISTANCE_THRESHOLD = 0.05 # Distance threshold (meters) for collision detection
MIN_RELATIVE_SPEED_FOR_TTC = 0.1 # Minimum relative speed (m/s) along connection line to calculate finite TTC

# --- Resampling Parameters ---
NUM_PERMUTATIONS = 1000 # Number of iterations for resampling

# --- Fixed Histogram Ranges (Based on provided analysis) ---
FIXED_HIST_RANGES = {
    "linear_speed":           (0.0, 3.5),
    "linear_acceleration":    (-0.2, 0.2),
    "angular_speed":          (-0.1, 0.1),
    "angular_acceleration":   (-0.05, 0.05),
    "dist_road_edge":         (0.0, 2.0),
    "dist_nearest_obstacle":  (0.0, 70.0),
    "ttc":                    (0.0, TTC_SPLIT_POINT), # Range for bins 0 to NUM_BINS_TTC_FINITE-1
    "collision":              (-0.5, 1.5), # Specific bins below, range for xlim
    "offroad":                (-0.5, 1.5), # Specific bins below, range for xlim
}

# --- Metric Names for Plotting ---
METRIC_NAME_MAP = {
    "linear_speed": "Linear Speed (m/s)",
    "linear_acceleration": "Linear Acceleration (m/s^2)",
    "angular_speed": "Angular Speed (rad/s)",
    "angular_acceleration": "Angular Acceleration (rad/s^2)",
    "dist_road_edge": "Distance to Nearest Road Edge (m)",
    "dist_nearest_obstacle": "Distance to Nearest Obstacle (m)",
    "ttc": "Time to Collision (s)",
    "collision": "Collision Indication", # Remove units for binary label
    "offroad": "Off-Road Indication",   # Remove units for binary label
}

# --- WOSAC-Style Helper Functions (NumPy Implementation) ---
def _wrap_angle_numpy(angle: np.ndarray) -> np.ndarray:
    """Wraps angles in the range [-pi, pi]. NumPy version."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def central_diff_numpy(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Computes central difference (f[i+1]-f[i-1])/(t[i+1]-t[i-1]) with NaN padding.
    Handles NaNs in input data.
    """
    if data.shape[0] < 3: return np.full_like(data, np.nan, dtype=float)
    diffs = np.full_like(data[1:-1], np.nan, dtype=float)
    dt_2step = times[2:] - times[:-2]
    valid_dt_mask = dt_2step > MIN_DT_TWO_STEP
    potential_indices = np.where(valid_dt_mask)[0]
    if len(potential_indices) == 0: return np.full_like(data, np.nan, dtype=float)
    original_indices_center = potential_indices + 1
    valid_data_mask = (original_indices_center + 1 < data.shape[0]) & \
                      (original_indices_center - 1 >= 0) & \
                      ~np.isnan(data[original_indices_center + 1]) & \
                      ~np.isnan(data[original_indices_center - 1])
    valid_calculation_indices_center = original_indices_center[valid_data_mask]
    if len(valid_calculation_indices_center) == 0: return np.full_like(data, np.nan, dtype=float)
    valid_calculation_indices_diffs = valid_calculation_indices_center - 1
    data_plus_1 = data[valid_calculation_indices_center + 1]
    data_minus_1 = data[valid_calculation_indices_center - 1]
    dt_valid = dt_2step[valid_dt_mask][valid_data_mask]
    diffs[valid_calculation_indices_diffs] = (data_plus_1 - data_minus_1) / dt_valid
    padded_diffs = np.full(data.shape[0], np.nan, dtype=float)
    padded_diffs[1:-1] = diffs
    return padded_diffs

def central_logical_and_numpy(valid_mask: np.ndarray) -> np.ndarray:
    """
    Computes central logical_and (valid[i-1] and valid[i+1]) with False padding.
    """
    if valid_mask.shape[0] < 3: return np.full_like(valid_mask, False, dtype=bool)
    central_and = np.logical_and(valid_mask[:-2], valid_mask[2:])
    return np.pad(central_and, (1, 1), mode='constant', constant_values=False)

# --- Geometric Helper Function ---
if SHAPELY_AVAILABLE:
    def create_vehicle_polygon(x: float, y: float, heading: float, length: float, width: float) -> Optional[Polygon]:
        if any(v is None or math.isnan(v) for v in [x, y, heading, length, width]): return None
        try:
            half_length = length / 2.0; half_width = width / 2.0
            cos_h = math.cos(heading); sin_h = math.sin(heading)
            fl_x = x + half_length * cos_h - half_width * sin_h; fl_y = y + half_length * sin_h + half_width * cos_h
            fr_x = x + half_length * cos_h + half_width * sin_h; fr_y = y + half_length * sin_h - half_width * cos_h
            rl_x = x - half_length * cos_h - half_width * sin_h; rl_y = y - half_length * sin_h + half_width * cos_h
            rr_x = x - half_length * cos_h + half_width * sin_h; rr_y = y - half_length * sin_h - half_width * cos_h
            return Polygon([(fl_x, fl_y), (fr_x, fr_y), (rr_x, rr_y), (rl_x, rl_y)])
        except (TypeError, ValueError): return None
else:
     def create_vehicle_polygon(x: float, y: float, heading: float, length: float, width: float) -> None: return None

# --- Helper function to calculate 1D Wasserstein distance ---
def calculate_1d_wasserstein(data1_no_nan: np.ndarray, data2_no_nan: np.ndarray,
                             metric_key: str, cost_matrix: Optional[np.ndarray]) -> float:
    """Calculates 1D Wasserstein distance between two data arrays using histograms."""
    dist = np.nan
    if cost_matrix is None: # Cost matrix might not be creatable for some reason
        return dist

    n_bins_wasserstein = NUM_BINS
    finite_bin_edges_calc = None
    if metric_key == "ttc":
        n_bins_wasserstein = NUM_BINS_TTC_FINITE + 1
        finite_bin_edges_calc = np.linspace(0, TTC_SPLIT_POINT, NUM_BINS_TTC_FINITE + 1)
    elif metric_key in ["collision", "offroad"]:
        n_bins_wasserstein = 2

    # Build histograms for Wasserstein
    hists_for_w = []
    sums_for_w = []
    valid_hists = True
    for data_arr_nn in [data1_no_nan, data2_no_nan]:
        hist_w = None
        current_sum = 0
        if len(data_arr_nn) > 0:
            try:
                if metric_key == "ttc":
                    hist_finite, _ = np.histogram(data_arr_nn[data_arr_nn < TTC_SPLIT_POINT], bins=finite_bin_edges_calc)
                    count_ge_split = np.sum(data_arr_nn >= TTC_SPLIT_POINT)
                    hist_w = np.append(hist_finite, count_ge_split)
                elif metric_key in ["collision", "offroad"]:
                    bins_config_calc = [-0.5, 0.5, 1.5] # Ensure this matches plot
                    hist_w, _ = np.histogram(data_arr_nn, bins=bins_config_calc)
                else:
                    current_range_calc = FIXED_HIST_RANGES.get(metric_key)
                    if current_range_calc is not None:
                        hist_w, _ = np.histogram(data_arr_nn, bins=NUM_BINS, range=current_range_calc)
                    else:
                        hist_w, _ = np.histogram(data_arr_nn, bins=NUM_BINS) # Auto-range if not fixed

                if hist_w is not None:
                    current_sum = hist_w.sum()
            except Exception as e:
                # print(f"  Warning: Error calculating histogram for W-dist during permutation ({metric_key}): {e}")
                valid_hists = False; break
        hists_for_w.append(hist_w); sums_for_w.append(current_sum)

    if not valid_hists: return np.nan

    sum1, sum2 = sums_for_w
    if sum1 > 0 and sum2 > 0:
        try:
            p1 = hists_for_w[0] / sum1
            p2 = hists_for_w[1] / sum2
            wasserstein2_squared = ot.emd2(p1, p2, cost_matrix)
            if wasserstein2_squared < 0: wasserstein2_squared = 0
            dist = np.sqrt(wasserstein2_squared)
        except Exception as e:
            # print(f"  Warning: Could not calculate W-dist during permutation ({metric_key}): {e}")
            dist = np.nan
    elif sum1 == 0 and sum2 == 0:
        dist = 0.0
    else:
        dist = np.inf
    return dist

# --- Manual Holm-Bonferroni implementation ---
def holm_bonferroni_correction(p_values_list: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Performs Holm-Bonferroni correction.
    Returns:
        - rejected: A list of booleans indicating if null hypothesis is rejected.
        - pvals_corrected: A list of adjusted p-values.
    """
    if not p_values_list:
        return [], []

    m = len(p_values_list)
    # Create pairs of (original_index, p_value) to keep track after sorting
    indexed_p_values = sorted([(i, p) for i, p in enumerate(p_values_list)], key=lambda x: x[1])

    # Initialize outputs in the original order
    rejected = [False] * m
    pvals_corrected_temp = [0.0] * m # Will store adjusted p-values in sorted order first
    
    for k in range(m):
        p_val_sorted = indexed_p_values[k][1]
        # Calculate the Holm adjusted p-value for this step
        p_adj_k = min(1.0, (m - k) * p_val_sorted) # This is p_raw_sorted * (m - rank + 1)
                                                  # or more precisely, p_raw_sorted * (number of remaining hypotheses at this step)
                                                  # The (m-k) term correctly reflects the decreasing divisor in the comparison.
        pvals_corrected_temp[k] = p_adj_k

    # Ensure monotonicity: The adjusted p-value cannot be smaller than the previous one in the sorted list
    # This is the "max over previous adjustments" part
    for k in range(1, m):
        pvals_corrected_temp[k] = max(pvals_corrected_temp[k], pvals_corrected_temp[k-1])
    
    # Map adjusted p-values back to their original positions
    pvals_corrected_final = [0.0] * m
    for k in range(m):
        original_index = indexed_p_values[k][0]
        pvals_corrected_final[original_index] = pvals_corrected_temp[k]
            
    # Final rejection decision based on comparing adjusted p-values to alpha
    for i in range(m):
        if pvals_corrected_final[i] <= alpha:
            rejected[i] = True
            
    return rejected, pvals_corrected_final


# --- Plotting Function for Grouped Histograms and Pairwise Wasserstein Calculation ---
def plot_grouped_histograms_and_calculate_pairwise_wasserstein(
    metrics_data_list: List[Dict[str, List[float]]], # List of metric dicts for each dataset
    labels: List[str], # List of labels for each dataset
    output_dir: str
) -> Dict[str, Dict[Tuple[str, str], Tuple[float, float]]]: # Return (distance, p-value)
    """
    Generates and saves grouped histograms and calculates pairwise 1D Wasserstein
    distances with p-values from permutation tests.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating Grouped Histograms & Calculating Pairwise Wasserstein Distances in: {output_path}")

    if not metrics_data_list: print("  Error: No metrics data provided."); return {}
    if len(metrics_data_list) != len(labels): print("  Error: Number of metrics data dictionaries must match number of labels."); return {}

    num_datasets = len(metrics_data_list)
    if num_datasets < 2: print("  Warning: Need at least two datasets to calculate Wasserstein distances and p-values.")

    colors = HIST_COLORS[:num_datasets]

    all_metric_keys = set().union(*(d.keys() for d in metrics_data_list))
    all_metric_keys = sorted(list(all_metric_keys))

    all_wasserstein_results = defaultdict(dict) # Stores {(metric_key): {(label_i, label_j): (distance, p_value)}}

    for key in all_metric_keys:
        user_friendly_name = METRIC_NAME_MAP.get(key, key)

        # --- Prepare data for plotting (same as before) ---
        data_arrays_no_nan = []
        total_non_nan_counts = []
        data_for_hist_list = []
        weights_list = []
        hist_labels = []
        has_any_data = False

        for i in range(num_datasets):
            data_list_raw = metrics_data_list[i].get(key, [])
            # Ensure data is float, handle potential non-numeric entries gracefully
            data_list_numeric = []
            for item in data_list_raw:
                try:
                    data_list_numeric.append(float(item))
                except (ValueError, TypeError):
                    data_list_numeric.append(np.nan) # Convert non-floatable to NaN

            data_array = np.array(data_list_numeric, dtype=float)
            nan_mask = ~np.isnan(data_array)
            data_array_no_nan = data_array[nan_mask]

            total_non_nan_count = len(data_array_no_nan)
            data_arrays_no_nan.append(data_array_no_nan)
            total_non_nan_counts.append(total_non_nan_count)
            hist_labels.append(f"{labels[i]} (N={total_non_nan_count})")
            if total_non_nan_count > 0:
                has_any_data = True
                weights = np.ones_like(data_array_no_nan) / total_non_nan_count * 100
                data_for_hist = data_array_no_nan
            else:
                weights = None
                data_for_hist = np.array([])
            weights_list.append(weights)
            data_for_hist_list.append(data_for_hist)

        if not has_any_data:
            print(f"  No valid (non-NaN) data for '{user_friendly_name}' in any dataset. Skipping metric.")
            continue

        # --- Plotting setup (same as before) ---
        plt.figure() # Uses rcParams for figsize now
        plot_range = FIXED_HIST_RANGES.get(key, None)
        bins_config_plot = NUM_BINS
        n_bins_wasserstein = NUM_BINS # Default for Wasserstein calculation bins
        ylabel = 'Percentage (%)'
        plot_xlim = plot_range
        xtick_labels = None
        xtick_locs = None
        current_plot_range_for_hist = plot_range
        finite_bin_edges_calc = None # For TTC W-dist

        if key == "ttc":
            n_bins_wasserstein = NUM_BINS_TTC_FINITE + 1
            finite_bin_edges_calc = np.linspace(0, TTC_SPLIT_POINT, NUM_BINS_TTC_FINITE + 1)
            bin_width_plot = TTC_SPLIT_POINT / NUM_BINS_TTC_FINITE if NUM_BINS_TTC_FINITE > 0 else 1.0
            plot_bin_upper_edge = TTC_SPLIT_POINT + bin_width_plot
            bins_config_plot = np.append(finite_bin_edges_calc, plot_bin_upper_edge)
            current_plot_range_for_hist = (0, plot_bin_upper_edge)
            plot_xlim = (0, plot_bin_upper_edge)
            last_bin_midpoint = (TTC_SPLIT_POINT + plot_bin_upper_edge) / 2
            for i in range(num_datasets):
                if total_non_nan_counts[i] > 0:
                    data_array_nn = data_arrays_no_nan[i]
                    plot_data = np.where(data_array_nn >= TTC_SPLIT_POINT, last_bin_midpoint, data_array_nn)
                    data_for_hist_list[i] = plot_data
            num_ticks = 6
            xtick_locs = np.linspace(0, TTC_SPLIT_POINT, num_ticks)
            xtick_labels = [f"{int(loc)}" for loc in xtick_locs[:-1]]
            xtick_labels.append(f">{int(TTC_SPLIT_POINT)}")

        elif key in ["collision", "offroad"]:
             bins_config_plot = [-0.5, 0.5, 1.5]
             current_plot_range_for_hist = FIXED_HIST_RANGES[key]
             plot_xlim = current_plot_range_for_hist
             n_bins_wasserstein = 2
             xtick_locs = [0, 1]; xtick_labels = ["0", "1"]


        # --- Cost Matrix for Wasserstein ---
        cost_matrix_w = None
        if OT_AVAILABLE:
            bin_indices = np.arange(n_bins_wasserstein)
            try:
                cost_matrix_w = ot.dist(bin_indices.reshape((n_bins_wasserstein, 1)),
                                        bin_indices.reshape((n_bins_wasserstein, 1)),
                                        metric='sqeuclidean')
            except Exception as e:
                 print(f"  Warning: Could not create cost matrix for '{user_friendly_name}': {e}. Skipping W-dist for this metric.")
                 cost_matrix_w = None


        # --- Calculate Observed Wasserstein Distances and P-values via Permutation ---
        if OT_AVAILABLE and num_datasets >= 2 and cost_matrix_w is not None:
            for i, j in itertools.combinations(range(num_datasets), 2):
                data1_nn = data_arrays_no_nan[i]
                data2_nn = data_arrays_no_nan[j]

                if len(data1_nn) == 0 and len(data2_nn) == 0:
                    observed_dist = 0.0
                    p_value = 1.0 # No data to distinguish
                elif len(data1_nn) == 0 or len(data2_nn) == 0:
                    observed_dist = np.inf
                    p_value = 0.0 # Infinitely different if one is empty and other not (conventionally)
                                  # or np.nan if preferred to indicate non-comparable for p-value
                else:
                    observed_dist = calculate_1d_wasserstein(data1_nn, data2_nn, key, cost_matrix_w)

                    # Permutation Test
                    perm_distances = []
                    if not np.isnan(observed_dist) and observed_dist != np.inf: # Only permute if observed is valid
                        combined_data = np.concatenate((data1_nn, data2_nn))
                        n1 = len(data1_nn)
                        for _ in range(NUM_PERMUTATIONS):
                            np.random.shuffle(combined_data)
                            perm_data1 = combined_data[:n1]
                            perm_data2 = combined_data[n1:]
                            perm_dist = calculate_1d_wasserstein(perm_data1, perm_data2, key, cost_matrix_w)
                            if not np.isnan(perm_dist):
                                perm_distances.append(perm_dist)

                        if perm_distances:
                            perm_distances_arr = np.array(perm_distances)
                            p_value = (np.sum(perm_distances_arr >= observed_dist) + 1) / (len(perm_distances_arr) + 1)
                        else: # No valid permuted distances calculated (e.g., if all were inf or NaN)
                            p_value = np.nan
                    else: # Observed distance was NaN or Inf
                        p_value = np.nan

                all_wasserstein_results[key][(labels[i], labels[j])] = (observed_dist, p_value)

        # --- Plotting ---
        plot_data_filtered = [d for k, d in enumerate(data_for_hist_list) if total_non_nan_counts[k] > 0]
        plot_weights_filtered = [w for k, w in enumerate(weights_list) if total_non_nan_counts[k] > 0]
        plot_labels_filtered = [l for k, l in enumerate(hist_labels) if total_non_nan_counts[k] > 0]
        plot_colors_filtered = [c for k, c in enumerate(colors) if total_non_nan_counts[k] > 0]

        if not plot_data_filtered:
            print(f"  Skipping plot for '{user_friendly_name}' as no filtered data exists.")
            plt.close(); continue

        plt.hist(plot_data_filtered, bins=bins_config_plot, weights=plot_weights_filtered, range=current_plot_range_for_hist,
                 color=plot_colors_filtered, label=plot_labels_filtered, edgecolor='darkgrey')

        title_suffix = "Fixed Range"
        if key == "ttc": title_suffix = f"(Fixed Range Incl. >{int(TTC_SPLIT_POINT)} Bin)"
        elif plot_range is None and key not in ["collision", "offroad"]: title_suffix = "Auto Range (No Fixed Range Defined)"
        elif key in ["collision", "offroad"]: title_suffix = "Binary (0=No, 1=Yes)"
        plt.xlabel(f'{user_friendly_name}'); plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.ylim(bottom=0)
        if SHOW_HISTOGRAM_LEGEND and plot_labels_filtered: plt.legend()
        if plot_xlim: plt.xlim(plot_xlim)
        if xtick_locs is not None and xtick_labels is not None: plt.xticks(ticks=xtick_locs, labels=xtick_labels)
        base_name = user_friendly_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('^','_').replace('|','').replace('.','').replace('=','')
        filename = f'{base_name}_grouped_hist.png'
        output_file = output_path / filename
        try: plt.savefig(output_file, bbox_inches='tight', dpi=SAVE_DPI)
        except Exception as e: print(f"  Error saving histogram {output_file}: {e}")
        plt.close()

    print("\nGrouped histogram generation and Pairwise Wasserstein distance (with p-values) calculation complete.")
    return all_wasserstein_results

# --- Main Processing Function ---
def calculate_raw_metrics_for_all_vehicles(xml_dir: str) -> Dict[str, List[float]]:
    xml_path = Path(xml_dir)
    if not xml_path.is_dir(): print(f"Error: XML directory not found: {xml_dir}"); return {}
    all_metrics_data = defaultdict(list); required_keys = list(METRIC_NAME_MAP.keys())
    for key in required_keys: all_metrics_data[key]
    print(f"\nProcessing XML files in: {xml_path} for metrics (ALL vehicles)...")
    file_count = 0; total_trajectories_processed = 0; total_trajectories_skipped_short = 0; skipped_context_metrics_count = 0
    xml_files = list(xml_path.glob('*.xml')); total_files = len(xml_files)
    if total_files == 0: print(f"No XML files found: {xml_dir}"); return {}
    for idx, filepath in enumerate(xml_files):
        file_count += 1
        print(f"  Processing file {file_count}/{total_files}: {filepath.name} [{xml_dir}] ...", end='\r', flush=True)
        lanelets_data = {}; obstacles_data: Dict[str, Dict[str, Any]] = {}
        try:
            tree = ET.parse(filepath); root = tree.getroot()
            if SHAPELY_AVAILABLE:
                for lanelet_elem in root.findall('.//lanelet'):
                    lanelet_id = lanelet_elem.get('id')
                    if not lanelet_id: continue
                    try:
                        left_pts = [(float(p.find('x').text), float(p.find('y').text)) for p in lanelet_elem.findall('./leftBound/point')]
                        right_pts = [(float(p.find('x').text), float(p.find('y').text)) for p in lanelet_elem.findall('./rightBound/point')]
                        if len(left_pts)>1 and len(right_pts)>1:
                            left_ls=LineString(left_pts); right_ls=LineString(right_pts); poly_pts=left_pts+right_pts[::-1]
                            try:
                                lanelet_poly=Polygon(poly_pts)
                                if not lanelet_poly.is_valid:
                                    fixed_poly=lanelet_poly.buffer(0)
                                    if fixed_poly.is_valid and isinstance(fixed_poly, Polygon): lanelet_poly = fixed_poly
                                    else: continue
                                lanelets_data[lanelet_id] = {'left': left_ls, 'right': right_ls, 'polygon': lanelet_poly}
                            except Exception: continue
                    except (AttributeError, ValueError, TypeError): continue
            for obs_elem in root.findall('.//dynamicObstacle'):
                obs_id = obs_elem.get('id');
                if not obs_id: continue
                try:
                    shape_elem = obs_elem.find('./shape/rectangle');
                    if shape_elem is None: continue
                    length_elem=shape_elem.find('length'); width_elem=shape_elem.find('width')
                    if length_elem is None or width_elem is None: continue
                    length=float(length_elem.text); width=float(width_elem.text)
                    if math.isnan(length) or math.isnan(width): continue
                    trajectory_states={}; states=obs_elem.findall('.//state')
                    initial_state_elem=obs_elem.find('./initialState')
                    if initial_state_elem is not None: states.insert(0,initial_state_elem)
                    state_list_for_sorting=[]
                    for state in states:
                        time_exact_elem=state.find('./time/exact'); time_exact=time_exact_elem.text if time_exact_elem is not None else None
                        pos_x_elem=state.find('./position/point/x'); pos_x=pos_x_elem.text if pos_x_elem is not None else None
                        pos_y_elem=state.find('./position/point/y'); pos_y=pos_y_elem.text if pos_y_elem is not None else None
                        orient_exact_elem=state.find('./orientation/exact'); orient_exact=orient_exact_elem.text if orient_exact_elem is not None else None
                        vel_exact_elem=state.find('./velocity/exact'); vel_exact=vel_exact_elem.text if vel_exact_elem is not None else None
                        if None in [time_exact, pos_x, pos_y, orient_exact]: continue
                        try:
                            t=float(time_exact)
                            state_data={'time':t,'x':float(pos_x),'y':float(pos_y),'heading':float(orient_exact),
                                          'velocity':float(vel_exact) if vel_exact is not None else np.nan}
                            if any(math.isnan(val) for key_val,val in state_data.items() if key_val!='velocity'): continue # corrected variable name
                            state_list_for_sorting.append(state_data)
                        except (ValueError,TypeError): continue
                    state_list_for_sorting.sort(key=lambda s:s['time'])
                    unique_states={s['time']:s for s in state_list_for_sorting}; trajectory_states=unique_states
                    if trajectory_states: obstacles_data[obs_id]={'shape':{'length':length,'width':width},'trajectory':trajectory_states,
                                                  'sorted_times':np.array(sorted(trajectory_states.keys()))}
                except (AttributeError,ValueError,TypeError,ET.ParseError): continue
            if not obstacles_data: continue
            for current_vehicle_id,current_vehicle_data in obstacles_data.items():
                ego_trajectory=current_vehicle_data['trajectory']; ego_times=current_vehicle_data['sorted_times']; ego_shape=current_vehicle_data['shape']
                if len(ego_trajectory)<3: total_trajectories_skipped_short+=1; continue
                times_list,pos_x_list,pos_y_list,headings_list,velocity_list=[],[],[],[],[]
                for t in ego_times:
                    state=ego_trajectory.get(t)
                    if state: times_list.append(state['time']);pos_x_list.append(state['x']);pos_y_list.append(state['y'])
                    headings_list.append(state['heading']);velocity_list.append(state['velocity'])
                times=np.array(times_list,dtype=float);pos_x=np.array(pos_x_list,dtype=float);pos_y=np.array(pos_y_list,dtype=float)
                headings=np.array(headings_list,dtype=float);velocities=np.array(velocity_list,dtype=float)
                initial_valid_mask=~np.isnan(times)&~np.isnan(pos_x)&~np.isnan(pos_y)&~np.isnan(headings)
                if np.sum(initial_valid_mask)<3: total_trajectories_skipped_short+=1; continue
                pos_z=np.zeros_like(pos_x)
                speed_validity=central_logical_and_numpy(initial_valid_mask);acceleration_validity=central_logical_and_numpy(speed_validity)
                vx_cd=central_diff_numpy(pos_x,times);vy_cd=central_diff_numpy(pos_y,times);vz_cd=central_diff_numpy(pos_z,times)
                linear_speed=np.sqrt(vx_cd**2+vy_cd**2+vz_cd**2);linear_speed[~speed_validity]=np.nan
                linear_acceleration=central_diff_numpy(linear_speed,times);linear_acceleration[~acceleration_validity]=np.nan
                angular_speed=np.full_like(headings,np.nan,dtype=float)
                if headings.shape[0]>=3:
                    dt_2step_head=times[2:]-times[:-2];valid_dt_mask_head=dt_2step_head>MIN_DT_TWO_STEP
                    potential_indices_head=np.where(valid_dt_mask_head)[0]
                    if len(potential_indices_head)>0:
                        original_indices_center_head=potential_indices_head+1
                        valid_input_mask_head=(original_indices_center_head-1>=0)& \
                                                (original_indices_center_head+1<headings.shape[0])& \
                                                initial_valid_mask[original_indices_center_head-1]& \
                                                initial_valid_mask[original_indices_center_head+1]
                        valid_calc_indices_head_center=original_indices_center_head[valid_input_mask_head]
                        if len(valid_calc_indices_head_center)>0:
                            valid_calc_indices_head_diffs=potential_indices_head[valid_input_mask_head]
                            diff=headings[valid_calc_indices_head_center+1]-headings[valid_calc_indices_head_center-1]
                            wrapped_diff=_wrap_angle_numpy(diff)
                            angular_speed_center_valid=wrapped_diff/dt_2step_head[valid_calc_indices_head_diffs]
                            angular_speed[valid_calc_indices_head_center]=angular_speed_center_valid
                angular_speed[~speed_validity]=np.nan
                angular_acceleration=central_diff_numpy(angular_speed,times);angular_acceleration[~acceleration_validity]=np.nan
                all_metrics_data["linear_speed"].extend(linear_speed.tolist());all_metrics_data["linear_acceleration"].extend(linear_acceleration.tolist())
                all_metrics_data["angular_speed"].extend(angular_speed.tolist());all_metrics_data["angular_acceleration"].extend(angular_acceleration.tolist())
                num_states=len(times)
                traj_dist_edge=[np.nan]*num_states;traj_dist_obstacle=[np.nan]*num_states
                traj_ttc=[np.nan]*num_states;traj_collision=[np.nan]*num_states;traj_offroad=[np.nan]*num_states
                if not SHAPELY_AVAILABLE: skipped_context_metrics_count+=np.sum(initial_valid_mask)
                else:
                    ego_polygons=[create_vehicle_polygon(pos_x[i],pos_y[i],headings[i],ego_shape['length'],ego_shape['width']) if initial_valid_mask[i] else None for i in range(num_states)]
                    for i,t_ego in enumerate(times):
                        if not initial_valid_mask[i]: continue
                        ego_poly=ego_polygons[i];
                        if ego_poly is None: continue
                        ego_x,ego_y,ego_heading=pos_x[i],pos_y[i],headings[i];ego_point=Point(ego_x,ego_y)
                        ego_vel_magnitude_xml=velocities[i];ego_vx_cd_i,ego_vy_cd_i=vx_cd[i],vy_cd[i]
                        ego_vx_for_ttc,ego_vy_for_ttc=np.nan,np.nan
                        if not np.isnan(ego_vel_magnitude_xml):
                            ego_vx_for_ttc=ego_vel_magnitude_xml*math.cos(ego_heading);ego_vy_for_ttc=ego_vel_magnitude_xml*math.sin(ego_heading)
                        elif not np.isnan(ego_vx_cd_i) and not np.isnan(ego_vy_cd_i):
                            ego_vx_for_ttc=ego_vx_cd_i;ego_vy_for_ttc=ego_vy_cd_i
                        current_lanelet_geom=None;is_offroad=1.0;dist_edge=np.nan
                        for ll_id,ll_data in lanelets_data.items():
                            try:
                                if ll_data['polygon'].contains(ego_point): current_lanelet_geom=ll_data;break
                            except Exception: continue
                        if current_lanelet_geom:
                            lanelet_poly=current_lanelet_geom['polygon'];left_bound=current_lanelet_geom['left'];right_bound=current_lanelet_geom['right']
                            try: dist_left=ego_point.distance(left_bound);dist_right=ego_point.distance(right_bound);dist_edge=min(dist_left,dist_right)
                            except Exception: dist_edge=np.nan
                            is_offroad_check=1.0
                            try:
                                if lanelet_poly.contains(ego_poly): is_offroad_check=0.0
                                elif lanelet_poly.contains(ego_point) and lanelet_poly.intersects(ego_poly):
                                    intersection=lanelet_poly.intersection(ego_poly)
                                    if not intersection.is_empty and intersection.area>1e-6 and ego_poly.area>1e-6:
                                        if intersection.area/ego_poly.area>0.75: is_offroad_check=0.0
                            except Exception: pass
                            is_offroad=is_offroad_check
                        traj_offroad[i]=is_offroad;traj_dist_edge[i]=dist_edge
                        min_dist_obstacle=float('inf');nearest_obstacle_info=None;collision_occurred_step=0.0
                        for other_obs_id,other_obs_data in obstacles_data.items():
                            if other_obs_id==current_vehicle_id: continue
                            other_obs_times=other_obs_data['sorted_times'];other_obs_trajectory=other_obs_data['trajectory']
                            time_diffs=np.abs(other_obs_times-t_ego);closest_time_idx=np.argmin(time_diffs)
                            if time_diffs[closest_time_idx]<=TIME_MATCH_TOLERANCE:
                                other_obs_time=other_obs_times[closest_time_idx];other_obs_state=other_obs_trajectory.get(other_obs_time);other_obs_shape=other_obs_data['shape']
                                if not other_obs_state: continue
                                other_obs_poly=create_vehicle_polygon(other_obs_state['x'],other_obs_state['y'],other_obs_state['heading'],other_obs_shape['length'],other_obs_shape['width'])
                                if other_obs_poly is None: continue
                                try: dist=ego_poly.distance(other_obs_poly)
                                except Exception: continue
                                if dist<=COLLISION_DISTANCE_THRESHOLD:
                                    collision_occurred_step=1.0;min_dist_obstacle=dist;nearest_obstacle_info=(other_obs_state,other_obs_poly,dist);break
                                if dist<min_dist_obstacle:
                                    min_dist_obstacle=dist;nearest_obstacle_info=(other_obs_state,other_obs_poly,dist)
                        traj_dist_obstacle[i]=min_dist_obstacle if min_dist_obstacle!=float('inf') else np.nan;traj_collision[i]=collision_occurred_step
                        ttc_value=np.inf
                        if collision_occurred_step==0.0 and nearest_obstacle_info is not None:
                            nearest_obs_state,nearest_obs_poly,current_min_dist=nearest_obstacle_info
                            obs_vel_magnitude=nearest_obs_state.get('velocity',np.nan);obs_heading=nearest_obs_state.get('heading',np.nan)
                            velocities_valid_for_ttc=not any(np.isnan(v) for v in [ego_vx_for_ttc,ego_vy_for_ttc,obs_vel_magnitude,obs_heading])
                            if velocities_valid_for_ttc:
                                try:
                                    obs_vx=obs_vel_magnitude*math.cos(obs_heading);obs_vy=obs_vel_magnitude*math.sin(obs_heading)
                                    rel_vx=obs_vx-ego_vx_for_ttc;rel_vy=obs_vy-ego_vy_for_ttc
                                    rel_vel_vec=np.array([rel_vx,rel_vy])
                                    ego_center=ego_poly.centroid;obs_center=nearest_obs_poly.centroid
                                    rel_pos_vec=np.array([obs_center.x-ego_center.x,obs_center.y-ego_center.y])
                                    dist_centers=np.linalg.norm(rel_pos_vec)
                                    if dist_centers>1e-6:
                                        rel_speed_along_connection=-np.dot(rel_vel_vec,rel_pos_vec)/dist_centers
                                        if rel_speed_along_connection>MIN_RELATIVE_SPEED_FOR_TTC:
                                            if not np.isnan(current_min_dist) and current_min_dist>=0:
                                                 ttc_calculated=max(0,current_min_dist/rel_speed_along_connection)
                                                 ttc_value=ttc_calculated
                                except (ValueError,ZeroDivisionError,AttributeError,TypeError): ttc_value=np.nan
                            else: ttc_value=np.nan
                        elif collision_occurred_step==1.0: ttc_value=0.0
                        traj_ttc[i]=ttc_value
                all_metrics_data["dist_road_edge"].extend(traj_dist_edge);all_metrics_data["dist_nearest_obstacle"].extend(traj_dist_obstacle)
                all_metrics_data["ttc"].extend(traj_ttc);all_metrics_data["collision"].extend(traj_collision);all_metrics_data["offroad"].extend(traj_offroad)
                total_trajectories_processed+=1
        except ET.ParseError as e: print(f"\n    Error: Failed to parse XML file {filepath.name}: {e}")
        except Exception as e: print(f"\n    Error: An unexpected error occurred processing {filepath.name}: {e}\n{traceback.format_exc()}")
    print(" " * 100, end='\r')
    print(f"\n--- Processing Summary for {xml_dir} ---")
    print(f"Processed {file_count}/{total_files} files.")
    print(f"Successfully calculated metrics for {total_trajectories_processed} trajectories.")
    print(f"Skipped {total_trajectories_skipped_short} trajectories (fewer than 3 valid states).")
    if not SHAPELY_AVAILABLE: print(f"Skipped ALL context metrics calculations (Shapely not installed).")
    elif skipped_context_metrics_count>0: print(f"Skipped context metrics calculation for {skipped_context_metrics_count} states.")
    print("-" * 30)
    final_metrics_data={}
    for key in required_keys: final_metrics_data[key]=all_metrics_data.get(key,[])
    return final_metrics_data

# --- Run the calculation and plotting ---
if __name__ == "__main__":
    start_time = timing.time()
    xml_directories = [XML_DIRECTORY_1, XML_DIRECTORY_2] # Assuming 3 directories again for this example
    all_metrics_results = []
    dataset_labels = ["Trained on CARLA evaluated on Real", "Real"] # Adjusted labels
    overall_calc_start_time = start_time

    if len(xml_directories) != len(dataset_labels):
        print(f"Error: Mismatch between XML directories ({len(xml_directories)}) and labels ({len(dataset_labels)})."); exit()

    for i, xml_dir in enumerate(xml_directories):
        print(f"\n--- Starting Metric Calculation for Dataset {i+1} ({dataset_labels[i]}) ---")
        dataset_start_time_loop = timing.time() # Use a different variable for loop timing
        metrics = calculate_raw_metrics_for_all_vehicles(xml_dir)
        all_metrics_results.append(metrics)
        print(f"\nDataset {i+1} Calculation took {timing.time() - dataset_start_time_loop:.2f} seconds.")
    
    total_calc_end_time = timing.time() # Mark end of calculation
    print(f"\nTotal Metric Calculation took {total_calc_end_time - overall_calc_start_time:.2f} seconds.")


    print("\n--- Starting Grouped Histogram Generation & Pairwise Wasserstein Calculation (with p-values) ---")
    plotting_start_time = timing.time() # Mark start of plotting/W-dist
    pairwise_wasserstein_distances_with_pvals = {} # This will store {(metric): {(pair_labels): (dist, raw_pval)}}
    if all_metrics_results and OUTPUT_DIRECTORY:
        has_any_valid_data = any(any(val for val in d.get(key, []) if not (isinstance(val, float) and math.isnan(val))) for d in all_metrics_results for key in d)
        if has_any_valid_data:
            pairwise_wasserstein_distances_with_pvals = plot_grouped_histograms_and_calculate_pairwise_wasserstein(
                all_metrics_results, dataset_labels, OUTPUT_DIRECTORY
            )
        else: print("\nNo valid metric data. Skipping W-dist.")
    elif not all_metrics_results: print("\nMetrics calculation failed. Skipping W-dist.")
    else: print("\nOUTPUT_DIRECTORY not set. Skipping W-dist.")
    
    plotting_and_wdist_end_time = timing.time()
    print(f"\nHistogram Generation & W-Dist Calc took {plotting_and_wdist_end_time - plotting_start_time:.2f} seconds.")


    if OT_AVAILABLE and pairwise_wasserstein_distances_with_pvals and len(all_metrics_results) >= 2:
        print(f"\n--- Pairwise 1D Wasserstein Distance Summary (with Raw and Adjusted p-values) ---")
        
        all_raw_p_values_list_for_correction = []
        p_value_identifiers_for_correction = []
        max_key_len = 0
        max_label_len = 0
        
        sorted_metric_keys_for_collection = sorted(pairwise_wasserstein_distances_with_pvals.keys(), 
                                                 key=lambda k: METRIC_NAME_MAP.get(k, k))

        for key_metric in sorted_metric_keys_for_collection:
            metric_name_display = METRIC_NAME_MAP.get(key_metric, key_metric)
            max_key_len = max(max_key_len, len(metric_name_display))
            pair_dict_metric = pairwise_wasserstein_distances_with_pvals[key_metric]
            
            sorted_pairs_for_collection = sorted(pair_dict_metric.items(), 
                                               key=lambda item: (item[0][0], item[0][1]))

            for pair_labels_tuple, (dist, p_val_raw) in sorted_pairs_for_collection:
                comparison_str_display = f"{pair_labels_tuple[0]} vs {pair_labels_tuple[1]}"
                max_label_len = max(max_label_len, len(comparison_str_display))
                
                if p_val_raw is not None and not np.isnan(p_val_raw):
                    all_raw_p_values_list_for_correction.append(p_val_raw)
                    p_value_identifiers_for_correction.append(
                        {'metric_key': key_metric, 'pair_labels': pair_labels_tuple} # Removed raw_p and dist, not needed here
                    )
        
        adjusted_p_values_holm_map = {}
        if all_raw_p_values_list_for_correction:
            _, corrected_p_values_flat = holm_bonferroni_correction(all_raw_p_values_list_for_correction)
            
            for i, identifier_info in enumerate(p_value_identifiers_for_correction):
                metric_k = identifier_info['metric_key']
                pair_l = identifier_info['pair_labels']
                if metric_k not in adjusted_p_values_holm_map:
                    adjusted_p_values_holm_map[metric_k] = {}
                adjusted_p_values_holm_map[metric_k][pair_l] = corrected_p_values_flat[i]
        
        dist_header = "W-Dist"
        raw_pval_header = "Raw p"
        adj_pval_header = "Adj p (Holm)"
        header = f"{'Metric':<{max_key_len}} | {'Comparison':<{max_label_len}} | {dist_header:<10} | {raw_pval_header:<10} | {adj_pval_header:<12}"
        print(header)
        print("-" * len(header))

        for key_print in sorted_metric_keys_for_collection:
            metric_name_print = METRIC_NAME_MAP.get(key_print, key_print)
            pair_dict_print = pairwise_wasserstein_distances_with_pvals[key_print]
            
            sorted_pairs_print = sorted(pair_dict_print.items(), 
                                        key=lambda item: (item[0][0], item[0][1]))

            for pair_labels_print, (dist_print, raw_p_val_print) in sorted_pairs_print:
                comparison_str_print = f"{pair_labels_print[0]} vs {pair_labels_print[1]}"
                dist_str_print = f"{dist_print:.6f}" if dist_print is not None and not np.isnan(dist_print) else "N/A"
                raw_p_val_str_print = f"{raw_p_val_print:.4f}" if raw_p_val_print is not None and not np.isnan(raw_p_val_print) else "N/A"
                
                adj_p_val_retrieved = adjusted_p_values_holm_map.get(key_print, {}).get(pair_labels_print, np.nan)
                adj_p_val_str_print = f"{adj_p_val_retrieved:.4f}" if adj_p_val_retrieved is not None and not np.isnan(adj_p_val_retrieved) else "N/A"
                
                print(f"{metric_name_print:<{max_key_len}} | {comparison_str_print:<{max_label_len}} | {dist_str_print:<10} | {raw_p_val_str_print:<10} | {adj_p_val_str_print:<12}")
        print("-" * len(header))
        if all_raw_p_values_list_for_correction:
            print(f"Note: Adjusted p-values calculated using Holm-Bonferroni correction for {len(all_raw_p_values_list_for_correction)} tests.")
        else:
            print("Note: No valid p-values were available for multiple comparison correction.")


    # (Overall summary statistics printing code remains unchanged)
    print("\n--- Overall Summary Statistics (excluding NaNs) ---")
    summary_stats_start_time = timing.time() # Mark start of this section
    for i, metrics_data in enumerate(all_metrics_results):
        print(f"\n--- Dataset {i+1}: {dataset_labels[i]} ---")
        if not metrics_data or not any(m for m in metrics_data.values() if m): print("No metrics data."); continue
        sorted_keys_summary = sorted(metrics_data.keys(), key=lambda k_s: METRIC_NAME_MAP.get(k_s, k_s))
        for key_s in sorted_keys_summary:
            if key_s not in METRIC_NAME_MAP: continue
            name = METRIC_NAME_MAP.get(key_s, key_s); data_list = metrics_data.get(key_s, [])
            if not data_list: print(f"Metric: {name:<30} - No data."); continue
            data_array = np.array(data_list, dtype=float)
            data_array_no_nan = data_array[~np.isnan(data_array)]; num_total_valid = data_array_no_nan.size
            data_array_finite_or_bounded = data_array_no_nan
            # Print header for metric and valid count once
            print(f"Metric: {name:<30}"); print(f"  Valid Count (non-NaN): {num_total_valid}")

            if key_s == "ttc":
                 num_infinite_or_large = np.sum(data_array_no_nan >= TTC_SPLIT_POINT)
                 data_array_finite_or_bounded = data_array_no_nan[data_array_no_nan < TTC_SPLIT_POINT]
                 print(f"  Count >= {TTC_SPLIT_POINT:.1f} (incl. Inf): {num_infinite_or_large}")
            elif key_s not in ["collision", "offroad"]:
                 data_array_finite_or_bounded = data_array_no_nan[np.isfinite(data_array_no_nan)]
            # else collision/offroad, data_array_finite_or_bounded is already data_array_no_nan

            num_stats_basis = len(data_array_finite_or_bounded)
            if num_stats_basis > 0:
                stats_basis_label = "Finite"
                if key_s == "ttc": stats_basis_label = f"< {TTC_SPLIT_POINT:.1f}"
                elif key_s in ["collision", "offroad"]: stats_basis_label = "Valid (0 or 1)"
                print(f"  Count ({stats_basis_label}): {num_stats_basis}")
                if key_s not in ["collision", "offroad"]:
                    print(f"  Mean ({stats_basis_label}):  {np.mean(data_array_finite_or_bounded):>10.4f}")
                    print(f"  Std Dev ({stats_basis_label}): {np.std(data_array_finite_or_bounded):>10.4f}")
                print(f"  Min ({stats_basis_label}):   {np.min(data_array_finite_or_bounded):>10.4f}")
                if key_s not in ["collision", "offroad", "ttc"]:
                    try:
                        p1=np.percentile(data_array_finite_or_bounded,1); p50=np.percentile(data_array_finite_or_bounded,50); p99=np.percentile(data_array_finite_or_bounded,99)
                        print(f"  1st Pctl ({stats_basis_label}): {p1:>9.4f}"); print(f"  Median ({stats_basis_label}): {p50:>10.4f}"); print(f"  99th Pctl ({stats_basis_label}):{p99:>9.4f}")
                    except IndexError: print(f"  Percentiles ({stats_basis_label}): Could not calculate")
                elif key_s in ["collision", "offroad"]:
                    count_0=np.sum(data_array_finite_or_bounded==0); count_1=np.sum(data_array_finite_or_bounded==1)
                    print(f"  Count 0: {count_0}"); print(f"  Count 1: {count_1}")
                max_val = np.max(data_array_finite_or_bounded)
                if max_val != np.min(data_array_finite_or_bounded): print(f"  Max ({stats_basis_label}):   {max_val:>10.4f}")
            elif num_total_valid > 0 : print(f"  Count for stats basis ({stats_basis_label}): 0")
            print("-" * 20)
        print("-" * 40)
    
    summary_stats_end_time = timing.time()
    print(f"\nOverall Summary Stats Gen took {summary_stats_end_time - plotting_and_wdist_end_time:.2f} seconds.")


    total_end_time = timing.time()
    print(f"\n--- Total Script Execution Time: {total_end_time - start_time:.2f} seconds ---")