import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import math
import json
import ot
import ot.plot as opl
import warnings
from tqdm import tqdm
import time
import torch
import os

# --- Filter out specific warnings ---
warnings.filterwarnings("ignore", message="findfont: Font family.*not found.")
import logging
# Suppress Matplotlib font manager warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# --- Configuration ---
BASE_RESULTS_DIRECTORY = 'ot_analysis_experiment_v6_fixed_subspace' # Renamed for new logic
MAX_SAMPLES_FOR_OT_PLOT = 1000 # Max samples from source/target for OT plan viz

# --- Data Paths ---
XML_DIR_SYNTHETIC_ORIGINAL = "./test"
XML_DIR_REAL_TEST = "./real_mixture/cleaneddata/train_sampled_150"
XML_DIR_SYNTHETIC_MODEL_OUTPUT = "./carla_new->real_with_plots_overfit_smoothed/predicted_xmls"

# --- Sampling & Trajectory Config ---
SUBSET_SIZE_XML_FILES = 50 # Set to None to use all files
RANDOM_SEED_SYNTHETIC_ORIGINAL_SUBSET = 123
RANDOM_SEED_REAL_SUBSET = 42
RANDOM_SEED_MODEL_OUTPUT_SUBSET = 999
FIXED_TRAJECTORY_LENGTH = 30
COORDINATE_THRESHOLD = 1000.0

# --- PCA & OT Config ---
N_COMPONENTS_FOR_PCA = 5
SINKHORN_REGULARIZATION = 0.1
SINKHORN_MAX_ITER = 100000
SINKHORN_STOP_THRESHOLD = 5e-3
SINKHORN_METHOD = 'sinkhorn_log'

# --- Plotting Config ---
FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 15
EIGENVECTOR_VIS_SCALE = 1.0
N_EIGENVECTORS_TO_PLOT = N_COMPONENTS_FOR_PCA

# --- Dynamic Output Directories ---
results_suffix = f"xml_subset{SUBSET_SIZE_XML_FILES}" if SUBSET_SIZE_XML_FILES is not None else "all_files"
RESULTS_DIRECTORY = Path(BASE_RESULTS_DIRECTORY) / results_suffix
OUTPUT_DIR_VISUALIZATIONS = RESULTS_DIRECTORY / 'visualizations'
OUTPUT_DIR_RESULTS = RESULTS_DIRECTORY / 'results'
Path(OUTPUT_DIR_VISUALIZATIONS).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_RESULTS).mkdir(parents=True, exist_ok=True)


# --- Helper: Data Extraction ---
def extract_segments_by_filename(xml_dir_path: Path, segment_length=FIXED_TRAJECTORY_LENGTH):
    if not xml_dir_path.is_dir():
        print(f"Error: Directory not found: {xml_dir_path}")
        return {}
    segments_by_file = {}
    print(f"Processing XML files in: {xml_dir_path}")
    file_list = sorted(list(xml_dir_path.glob('*.xml')))
    if not file_list:
        print(f"Warning: No XML files found in {xml_dir_path}")
        return {}
    for filepath in tqdm(file_list, desc=f"Extracting from {xml_dir_path.name}"):
        segments_from_this_file = []
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            obstacles = root.findall('.//dynamicObstacle')
            for obstacle in obstacles:
                states = []
                init = obstacle.find('./initialState')
                if init is not None:
                    time_val = float(init.find('./time/exact').text); x = float(init.find('./position/point/x').text); y = float(init.find('./position/point/y').text); theta = float(init.find('./orientation/exact').text)
                    states.append({'time': time_val, 'x': x, 'y': y, 'theta': theta})
                traj = obstacle.find('./trajectory')
                if traj is not None:
                    for state in traj.findall('./state'):
                        time_val = float(state.find('./time/exact').text); x = float(state.find('./position/point/x').text); y = float(state.find('./position/point/y').text); theta = float(state.find('./orientation/exact').text)
                        states.append({'time': time_val, 'x': x, 'y': y, 'theta': theta})
                if len(states) < segment_length: continue
                states.sort(key=lambda s: s['time'])
                num_segments_in_obstacle = (len(states) - segment_length) // 1 + 1
                for i in range(num_segments_in_obstacle):
                    segment = states[i:i + segment_length]
                    x0, y0, theta0 = segment[0]['x'], segment[0]['y'], segment[0]['theta']
                    alpha = np.pi / 2 - theta0; cos_a, sin_a = np.cos(alpha), np.sin(alpha)
                    transformed = [[cos_a * (s['x'] - x0) - sin_a * (s['y'] - y0), sin_a * (s['x'] - x0) + cos_a * (s['y'] - y0)] for s in segment]
                    segments_from_this_file.append(np.array(transformed))
            if segments_from_this_file:
                segments_by_file[filepath.name] = segments_from_this_file
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    total_segments = sum(len(v) for v in segments_by_file.values())
    if not segments_by_file: print(f"\nWarning: No valid trajectory segments extracted from {xml_dir_path}.")
    else: print(f"\nSuccessfully extracted {total_segments} segments from {len(segments_by_file)} files in {xml_dir_path}.")
    return segments_by_file

# --- Filtering ---
def filter_segments_dict(segments_dict, threshold, dataset_name=""):
    if not segments_dict:
        print(f"\n[{dataset_name}] Filtering: Input segments dictionary is empty. Nothing to filter.")
        return {}
    filtered_dict = {}
    total_removed_count = 0
    total_before_filter = sum(len(v) for v in segments_dict.values())
    for fname, seg_list in segments_dict.items():
        kept_segments = []
        for seg in seg_list:
            if np.any(np.abs(seg) > threshold):
                total_removed_count += 1
            else:
                kept_segments.append(seg)
        if kept_segments:
            filtered_dict[fname] = kept_segments
    total_after_filter = sum(len(v) for v in filtered_dict.values())
    print(f"\n[{dataset_name}] Filtering: Removed {total_removed_count} segments out of {total_before_filter} with coordinates > {threshold}")
    print(f"[{dataset_name}] Remaining segments for Analysis: {total_after_filter} from {len(filtered_dict)} files.")
    return filtered_dict

# --- Flatten, Standardize ---
def flatten_trajectories_from_list(segments_list):
    if not segments_list: return np.array([]).reshape(0, FIXED_TRAJECTORY_LENGTH * 2 if FIXED_TRAJECTORY_LENGTH else 0)
    return np.array([s.flatten() for s in segments_list])

def standardize_data(X, mean=None, std=None, epsilon=1e-8):
    if X.shape[0] == 0: return X, np.array([]), np.array([])
    if mean is None: mean = np.mean(X, axis=0)
    if std is None: std = np.std(X, axis=0)
    std_safe = np.where(std < epsilon, 1.0, std)
    return (X - mean) / std_safe, mean, std_safe

# --- Visualization: Scree Plot ---
def plot_scree(explained_variance_ratio_np, output_dir, plot_title_suffix, filename_suffix, N_COMPONENTS_FOR_PCA, FONT_FAMILY, FONT_SIZE):
    # (Function remains the same, no changes needed)
    output_path = Path(output_dir)
    if not isinstance(explained_variance_ratio_np, (list, np.ndarray)) or len(explained_variance_ratio_np) == 0:
        print(f"Warning: Cannot create scree plot for '{plot_title_suffix}', invalid or empty EVR.")
        return
    try: plt.rcParams['font.family'] = FONT_FAMILY; plt.rcParams['font.size'] = FONT_SIZE
    except Exception as e: print(f"Warning: Could not set font {FONT_FAMILY} for scree plot: {e}. Matplotlib will use fallback.")
    num_components_to_plot = min(len(explained_variance_ratio_np), N_COMPONENTS_FOR_PCA)
    if num_components_to_plot == 0: print(f"Warning: No components to plot for '{plot_title_suffix}'."); return
    evr_subset = np.array(explained_variance_ratio_np)[:num_components_to_plot]
    components_to_display = np.arange(1, num_components_to_plot + 1)
    cumulative_variance_subset = np.cumsum(evr_subset)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_ylabel("PCA Components", fontsize=FONT_SIZE); ax1.set_xlabel("Proportion of Variance Explained", fontsize=FONT_SIZE)
    ax1.barh(components_to_display, evr_subset, color='teal', alpha=0.6, label='Individual EVR')
    ax1.plot(cumulative_variance_subset, components_to_display, color='black', marker='o', linestyle='--', label='Cumulative EVR')
    ax1.tick_params(axis='x', labelsize=FONT_SIZE); ax1.set_yticks(components_to_display); ax1.set_yticklabels([f"PC {i}" for i in components_to_display], fontsize=FONT_SIZE)
    max_val_on_x = max(np.max(evr_subset), np.max(cumulative_variance_subset)) if len(evr_subset) > 0 and len(cumulative_variance_subset) > 0 else (np.max(evr_subset) if len(evr_subset) > 0 else (np.max(cumulative_variance_subset) if len(cumulative_variance_subset) > 0 else 0))
    ax1.set_xlim(left=0, right=max_val_on_x * 1.15 if max_val_on_x > 0 else 0.5); ax1.invert_yaxis()
    ax2 = ax1.twinx(); ax2.set_ylim(ax1.get_ylim()); ax2.set_yticks(components_to_display); ax2.set_yticklabels([f"{val:.3f}" for val in cumulative_variance_subset], fontsize=FONT_SIZE, color='black')
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=FONT_SIZE, color='black', labelpad=10)
    ax1.grid(True, linestyle=':', which='major', axis='x', alpha=0.6)
    try: fig.tight_layout()
    except Exception: fig.tight_layout(rect=[0.12, 0.1, 0.8, 0.9])
    output_path.mkdir(parents=True, exist_ok=True)
    sanitized_suffix = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in filename_suffix.lower())
    filename = output_path / f'pca_scree_plot_{sanitized_suffix}.png'
    try: plt.savefig(filename, dpi=300); print(f"Saved scree plot: {filename}")
    except Exception as e: print(f"Error saving scree plot {filename}: {e}")
    plt.close(fig)

# --- Visualization: OT Matrix ---
def plot_ot_matrix(G, xs_data, xt_data, title_prefix, plot_suffix_name, output_dir, max_plot_size=200):
    # (Function remains the same, no changes needed)
    if G is None: print(f"Skipping OT matrix plot for {title_prefix}: G is None."); return
    Gs_np = G.cpu().numpy() if isinstance(G, torch.Tensor) else np.array(G)
    ns_orig, nt_orig = Gs_np.shape
    if ns_orig == 0 or nt_orig == 0: print(f"Skipping OT matrix plot for {title_prefix}: Matrix dimensions are zero ({ns_orig}x{nt_orig})."); return
    K_TOP_VALUES_TO_HIGHLIGHT = 400; K_TOP_VALUES_TO_HIGHLIGHT = min(K_TOP_VALUES_TO_HIGHLIGHT, ns_orig * nt_orig)
    Gs_highlighted = np.full_like(Gs_np, np.nan); flat_G = Gs_np.flatten()
    if K_TOP_VALUES_TO_HIGHLIGHT > 0 and flat_G.size > 0:
        threshold_value = np.partition(flat_G, -K_TOP_VALUES_TO_HIGHLIGHT)[-K_TOP_VALUES_TO_HIGHLIGHT] if K_TOP_VALUES_TO_HIGHLIGHT < flat_G.size else (np.min(flat_G) if flat_G.size > 0 else 0)
        mask_top_values = Gs_np >= threshold_value; Gs_highlighted[mask_top_values] = Gs_np[mask_top_values]
    else: Gs_highlighted = Gs_np.copy()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    final_Gs_to_plot, is_subsampled_for_display = Gs_highlighted, False
    if Gs_highlighted.shape[0] > max_plot_size or Gs_highlighted.shape[1] > max_plot_size:
        is_subsampled_for_display = True
        # (Subsampling logic remains the same)
    ax.set_xticks([]); ax.set_yticks([])
    current_cmap = plt.cm.get_cmap('viridis').copy(); current_cmap.set_bad(color='black')
    finite_values = final_Gs_to_plot[np.isfinite(final_Gs_to_plot)]
    vmin_plot, vmax_plot = (np.min(finite_values), np.max(finite_values)) if finite_values.size > 0 else (0, 0.0001)
    if vmin_plot >= vmax_plot: vmax_plot = vmin_plot + 1e-9
    im = ax.imshow(final_Gs_to_plot, interpolation='nearest', cmap=current_cmap, aspect='auto', vmin=vmin_plot, vmax=vmax_plot)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Transported Mass', rotation=270, labelpad=20);
    plot_title = f'{title_prefix}\n(Original Dims: {ns_orig}x{nt_orig})'
    if is_subsampled_for_display: plot_title += f', Displayed as {final_Gs_to_plot.shape[0]}x{final_Gs_to_plot.shape[1]}'
    ax.set_title(plot_title, fontsize=FONT_SIZE)
    ax.set_xlabel(f'Target Samples (RT, N={xt_data.shape[0]})', fontsize=FONT_SIZE-2)
    ax.set_ylabel(f'Source Samples ({plot_suffix_name.split("_vs_")[0].upper()}, N={xs_data.shape[0]})', fontsize=FONT_SIZE-2)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    filename = output_dir / f"ot_matrix_highlighted_noticks_{plot_suffix_name}.png"
    plt.savefig(filename, dpi=300); print(f"Saved OT matrix plot: {filename}"); plt.close(fig)

# --- Helper: OT Calculation & JSON Serialization ---
def calculate_ot_distance(Z_source_torch, Z_target_torch, reg, method, num_iter_max, stop_thr, device_type_str, pair_name=""):
    # (Function remains the same, no changes needed)
    print(f"\nCalculating Sinkhorn for {pair_name} (Source: {Z_source_torch.shape[0]}, Target: {Z_target_torch.shape[0]})")
    dist_np, sqrt_dist_np, log_dict, plan = np.nan, np.nan, None, None
    if Z_source_torch.shape[0] == 0 or Z_target_torch.shape[0] == 0:
        print(f"Error: Empty source or target for OT in {pair_name}.")
        return dist_np, sqrt_dist_np, log_dict, plan
    a = torch.ones(Z_source_torch.shape[0], device=Z_source_torch.device, dtype=Z_source_torch.dtype) / Z_source_torch.shape[0]
    b = torch.ones(Z_target_torch.shape[0], device=Z_target_torch.device, dtype=Z_target_torch.dtype) / Z_target_torch.shape[0]
    try:
        M_cost = ot.dist(Z_source_torch, Z_target_torch, metric='sqeuclidean')
    except Exception as e_m:
        print(f"Error creating M_cost for {pair_name}: {e_m}"); return dist_np, sqrt_dist_np, log_dict, plan
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", UserWarning)
            plan, log_dict = ot.sinkhorn(a, b, M_cost, reg, method=method, numItermax=num_iter_max, stopThr=stop_thr, log=True, verbose=False, warn=True)
        if plan is not None:
            dist_torch = torch.sum(plan * M_cost)
            dist_np = dist_torch.item()
            sqrt_dist_np = math.sqrt(dist_np) if dist_np >= 0 else 0.0
    except Exception as e_s:
        print(f"Error during Sinkhorn for {pair_name}: {e_s}")
    return dist_np, sqrt_dist_np, log_dict, plan

def _serialize_item(item):
    if isinstance(item, torch.Tensor): return item.item() if item.numel() == 1 else item.cpu().tolist()
    elif isinstance(item, np.ndarray): return item.item() if item.size == 1 else item.tolist()
    elif isinstance(item, list): return [_serialize_item(i) for i in item]
    elif isinstance(item, dict): return {str(sk): _serialize_item(sv) for sk, sv in item.items()}
    elif isinstance(item, (int, float, str, bool)) or item is None: return item
    elif isinstance(item, (np.integer, np.floating, np.bool_)): return item.item()
    else: return str(item)

def serialize_log(log_dict_to_serialize):
    if not isinstance(log_dict_to_serialize, dict): return None
    return _serialize_item(log_dict_to_serialize)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # --- 0. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch using: {device}")
    torch.set_default_dtype(torch.float32)
    try:
        plt.rcParams['font.family'] = FONT_FAMILY
        plt.rcParams['font.size'] = FONT_SIZE
    except Exception as e:
        print(f"Warning: Could not set font {FONT_FAMILY} globally: {e}.")

    # --- 1. Load & Filter ALL Data from the Three Sources ---
    print("\n--- Loading & Filtering Trajectory Data ---")
    segments_sto_all_filt = filter_segments_dict(extract_segments_by_filename(Path(XML_DIR_SYNTHETIC_ORIGINAL)), COORDINATE_THRESHOLD, "STO")
    segments_rt_all_filt = filter_segments_dict(extract_segments_by_filename(Path(XML_DIR_REAL_TEST)), COORDINATE_THRESHOLD, "RT")
    segments_stmo_all_filt = filter_segments_dict(extract_segments_by_filename(Path(XML_DIR_SYNTHETIC_MODEL_OUTPUT)), COORDINATE_THRESHOLD, "STMO")

    # --- 2. Select Independent Random Subsets from Each Source ---
    print(f"\n--- Selecting Independent Subsets (Size: {SUBSET_SIZE_XML_FILES} files each) ---")

    def select_subset_from_files(segments_dict, num_files_to_sample, seed, name):
        all_filenames = sorted(list(segments_dict.keys()))
        selected_segments = []
        selected_filenames = []
        if num_files_to_sample is not None and len(all_filenames) > 0:
            num_to_sample = min(num_files_to_sample, len(all_filenames))
            np.random.seed(seed)
            selected_filenames = np.random.choice(all_filenames, size=num_to_sample, replace=False)
        elif len(all_filenames) > 0:
            selected_filenames = all_filenames # Use all if subset size is None
        for fname in selected_filenames:
            if fname in segments_dict:
                selected_segments.extend(segments_dict[fname])
        flat_np = flatten_trajectories_from_list(selected_segments)
        print(f"[{name}] Selected {len(selected_filenames)} files, yielding {flat_np.shape[0]} trajectory segments.")
        return flat_np, sorted(list(map(str, selected_filenames)))

    X_sto_subset_flat_np, selected_sto_filenames = select_subset_from_files(segments_sto_all_filt, SUBSET_SIZE_XML_FILES, RANDOM_SEED_SYNTHETIC_ORIGINAL_SUBSET, "STO")
    X_rt_subset_flat_np, selected_rt_filenames = select_subset_from_files(segments_rt_all_filt, SUBSET_SIZE_XML_FILES, RANDOM_SEED_REAL_SUBSET, "RT")
    X_stmo_subset_flat_np, selected_stmo_filenames = select_subset_from_files(segments_stmo_all_filt, SUBSET_SIZE_XML_FILES, RANDOM_SEED_MODEL_OUTPUT_SUBSET, "STMO")

    # --- 3. Fit PCA on Input/Real Data & Project All Subsets ---
    print(f"\n--- Fitting PCA Model on STO and RT Subsets ---")
    pca_model = PCA(n_components=N_COMPONENTS_FOR_PCA)

    # Combine ONLY original synthetic and real data to define the PCA subspace
    data_for_pca_fitting = []
    if X_sto_subset_flat_np.shape[0] > 0: data_for_pca_fitting.append(X_sto_subset_flat_np)
    if X_rt_subset_flat_np.shape[0] > 0: data_for_pca_fitting.append(X_rt_subset_flat_np)

    # Initialize empty tensors for the projected data
    Z_sto_subset_torch = torch.empty((0, N_COMPONENTS_FOR_PCA), device=device)
    Z_rt_subset_torch = torch.empty((0, N_COMPONENTS_FOR_PCA), device=device)
    Z_stmo_subset_torch = torch.empty((0, N_COMPONENTS_FOR_PCA), device=device)
    X_for_pca_fit_np = np.array([]) # For logging purposes

    if data_for_pca_fitting:
        X_for_pca_fit_np = np.vstack(data_for_pca_fitting)
        print(f"Fitting PCA on a combined STO+RT dataset of {X_for_pca_fit_np.shape[0]} samples.")

        if X_for_pca_fit_np.shape[0] >= N_COMPONENTS_FOR_PCA:
            # Standardize and fit PCA on the combined STO+RT data
            X_for_pca_fit_std, pca_mean, pca_std = standardize_data(X_for_pca_fit_np)
            pca_model.fit(X_for_pca_fit_std)
            plot_scree(pca_model.explained_variance_ratio_, OUTPUT_DIR_VISUALIZATIONS, "PCA on STO+RT", "pca_sto_rt_fit", N_COMPONENTS_FOR_PCA, FONT_FAMILY, FONT_SIZE)

            # Project STO and RT subsets (which were used for fitting)
            if X_sto_subset_flat_np.shape[0] > 0:
                X_sto_std, _, _ = standardize_data(X_sto_subset_flat_np, pca_mean, pca_std)
                Z_sto_subset_torch = torch.from_numpy(pca_model.transform(X_sto_std)).to(device)
            if X_rt_subset_flat_np.shape[0] > 0:
                X_rt_std, _, _ = standardize_data(X_rt_subset_flat_np, pca_mean, pca_std)
                Z_rt_subset_torch = torch.from_numpy(pca_model.transform(X_rt_std)).to(device)

            # Project STMO subset (model output) into the EXISTING subspace
            if X_stmo_subset_flat_np.shape[0] > 0:
                print("Projecting STMO data into the existing STO+RT PCA subspace...")
                X_stmo_std, _, _ = standardize_data(X_stmo_subset_flat_np, pca_mean, pca_std)
                Z_stmo_subset_torch = torch.from_numpy(pca_model.transform(X_stmo_std)).to(device)
            print("Successfully projected all subsets.")
        else:
            print("Not enough combined STO+RT data for PCA fitting.")
    else:
        print("STO and RT data subsets are empty. Cannot create PCA subspace.")


    # --- 4. Calculate Wasserstein Distances ---
    print("\n--- Calculating Wasserstein Distances ---")

    # Input Gap: Original Synthetic vs. Real
    w_input_dist_np, w_input_sqrt_dist_np, w_input_log_ot, w_input_plan = calculate_ot_distance(
        Z_sto_subset_torch, Z_rt_subset_torch, SINKHORN_REGULARIZATION, SINKHORN_METHOD,
        SINKHORN_MAX_ITER, SINKHORN_STOP_THRESHOLD, device.type,
        pair_name="Input Gap (STO_subset vs RT_subset)"
    )
    print(f"W2_approx (Input Gap): {w_input_sqrt_dist_np:.4f}")

    # Output Gap: Model Output vs. Real (THE MAIN REQUEST)
    w_output_dist_np, w_output_sqrt_dist_np, w_output_log_ot, w_output_plan = calculate_ot_distance(
        Z_stmo_subset_torch, Z_rt_subset_torch, SINKHORN_REGULARIZATION, SINKHORN_METHOD,
        SINKHORN_MAX_ITER, SINKHORN_STOP_THRESHOLD, device.type,
        pair_name="Output Gap (STMO_subset vs RT_subset)"
    )
    print(f"W2_approx (Output Gap): {w_output_sqrt_dist_np:.4f}")

    # --- 5. Generate Visualizations ---
    print("\n--- Generating OT Matrix Visualizations ---")
    if w_input_plan is not None:
        plot_ot_matrix(w_input_plan, Z_sto_subset_torch.cpu().numpy(), Z_rt_subset_torch.cpu().numpy(),
                       f"Transport Plan (STO vs RT, {N_COMPONENTS_FOR_PCA}D PCA)", "sto_vs_rt_matrix",
                       OUTPUT_DIR_VISUALIZATIONS, max_plot_size=500)

    if w_output_plan is not None:
        plot_ot_matrix(w_output_plan, Z_stmo_subset_torch.cpu().numpy(), Z_rt_subset_torch.cpu().numpy(),
                       f"Transport Plan (STMO vs RT, {N_COMPONENTS_FOR_PCA}D PCA)", "stmo_vs_rt_matrix",
                       OUTPUT_DIR_VISUALIZATIONS, max_plot_size=500)

    # --- 6. Save Combined Results ---
    print("\n--- Saving Final Results Summary ---")
    summary_data = {
        'experiment_summary': 'Comparison of domain gaps using a fixed PCA subspace defined by STO+RT data.',
        'paths': {
            'synthetic_original': str(Path(XML_DIR_SYNTHETIC_ORIGINAL)),
            'real_test': str(Path(XML_DIR_REAL_TEST)),
            'synthetic_model_output': str(Path(XML_DIR_SYNTHETIC_MODEL_OUTPUT)),
        },
        'subset_config': {
            'subset_size_xml_files': SUBSET_SIZE_XML_FILES if SUBSET_SIZE_XML_FILES is not None else "all",
            'sto_subset': {'seed': RANDOM_SEED_SYNTHETIC_ORIGINAL_SUBSET, 'num_files': len(selected_sto_filenames), 'num_segments': X_sto_subset_flat_np.shape[0]},
            'rt_subset': {'seed': RANDOM_SEED_REAL_SUBSET, 'num_files': len(selected_rt_filenames), 'num_segments': X_rt_subset_flat_np.shape[0]},
            'stmo_subset': {'seed': RANDOM_SEED_MODEL_OUTPUT_SUBSET, 'num_files': len(selected_stmo_filenames), 'num_segments': X_stmo_subset_flat_np.shape[0]},
        },
        'pca_config': {
            'components': N_COMPONENTS_FOR_PCA,
            'fitted_on': f'Union of STO and RT subsets (Total N={X_for_pca_fit_np.shape[0]})',
            'explained_variance_ratio': pca_model.explained_variance_ratio_.tolist() if hasattr(pca_model, 'explained_variance_ratio_') else "N/A"
        },
        'sinkhorn_config': {
            'regularization': SINKHORN_REGULARIZATION, 'max_iter': SINKHORN_MAX_ITER,
            'stop_thr': SINKHORN_STOP_THRESHOLD, 'method': SINKHORN_METHOD, 'backend': f"pytorch_{device.type}"
        },
        'input_domain_gap_sto_vs_rt': {
            'W2_squared_approx': w_input_dist_np, 'W2_approx': w_input_sqrt_dist_np,
            'log_details': serialize_log(w_input_log_ot)
        },
        'output_domain_gap_stmo_vs_rt': {
            'W2_squared_approx': w_output_dist_np, 'W2_approx': w_output_sqrt_dist_np,
            'log_details': serialize_log(w_output_log_ot)
        }
    }
    summary_filename = f"ot_experiment_summary_{results_suffix}.json"
    summary_path = OUTPUT_DIR_RESULTS / summary_filename
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved experiment summary to: {summary_path}")

    print("\nAnalysis complete.")