import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import numpy as np

def generate_dummy_inputs(B=2, A=3, T=30, T_future=20, F=3, P=5, N=10):
    torch.manual_seed(0)
    inputs = torch.randn(B, A, T, F)
    sigma = torch.exp(torch.normal(mean=-1.2, std=1.2, size=(B,)))
    sigma_sampled, noisy_inputs = sample_noise(inputs)


    map_polylines = torch.randn(P, N, 2)
    polyline_masks = torch.randint(0, 2, (P,), dtype=torch.bool)

    pred_traj = noisy_inputs[0][:, -T_future:, :]
    gt_traj = inputs[0][:, -T_future:, :]
    noisy_traj = pred_traj.clone()
    trajectory_mask = torch.ones(A, T_future, dtype=torch.bool)
    ego_id = 0

    return {
        "inputs": inputs,
        "sigma": sigma,
        "map_polylines": map_polylines,
        "polyline_masks": polyline_masks,
        "pred_traj": pred_traj,
        "gt_traj": gt_traj,
        "noisy_traj": noisy_traj,
        "trajectory_mask": trajectory_mask,
        "ego_id": ego_id
    }
 
def plot_trajectories(
    map_polylines, polyline_masks, pred_traj, gt_traj, noisy_traj, trajectory_mask, ego_id,
    save_path=None, eval=False, title_suffix="", epoch=None, num_train_files=None, step=None, extra_info=None
):
    """
    Plot predicted and ground truth trajectories for all agents.
    Args:
        map_polylines: [P, N, 2] map coordinates.
        polyline_masks: [P] mask for valid polylines.
        pred_traj, gt_traj, noisy_traj: [A, T_future, 3] agent trajectories.
        trajectory_mask: [A, T_future] valid time steps.
        ego_id: ego agent ID.
        save_path: optional path to save figure.
        eval: if True, returns figure.
        title_suffix: extra title text.
        epoch: current epoch (int, optional).
        num_train_files: number of training files (int, optional).
        step: current training step (int, optional).
        extra_info: dict of additional info to display (optional).
    Returns:
        None
    """
    # Convert tensors to numpy
    to_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    map_polylines, polyline_masks = to_np(map_polylines), to_np(polyline_masks)
    pred_traj, gt_traj, noisy_traj = to_np(pred_traj), to_np(gt_traj), to_np(noisy_traj)
    trajectory_mask = to_np(trajectory_mask)

    # Warn if NaN/Inf
    for name, arr in [("map_polylines", map_polylines), ("pred_traj", pred_traj), ("gt_traj", gt_traj), ("noisy_traj", noisy_traj)]:
        if np.any(np.isnan(arr)): print(f"Warning: {name} contains NaN")
        if np.any(np.isinf(arr)): print(f"Warning: {name} contains Inf")

    A = pred_traj.shape[0]
    # Collect valid map and trajectory points
    valid_polys = [map_polylines[i] for i in range(map_polylines.shape[0]) if polyline_masks[i]]
    all_x = np.concatenate([p[:, 0] for p in valid_polys if p.shape[0] > 0]) if valid_polys else np.array([])
    all_y = np.concatenate([p[:, 1] for p in valid_polys if p.shape[0] > 0]) if valid_polys else np.array([])
    for a in range(A):
        mask = trajectory_mask[a].astype(bool)
        if np.any(mask):
            all_x = np.concatenate([all_x, pred_traj[a, mask, 0], gt_traj[a, mask, 0]])
            all_y = np.concatenate([all_y, pred_traj[a, mask, 1], gt_traj[a, mask, 1]])
    # Plot limits
    finite_x, finite_y = all_x[np.isfinite(all_x)], all_y[np.isfinite(all_y)]
    if finite_x.size and finite_y.size:
        xmin, xmax = np.min(finite_x), np.max(finite_x)
        ymin, ymax = np.min(finite_y), np.max(finite_y)
        rng = max(xmax - xmin, ymax - ymin, 1.0)
        pad = rng * 0.1
        xlim, ylim = (xmin - pad, xmax + pad), (ymin - pad, ymax + pad)
    else:
        xlim, ylim = (-20, 20), (-20, 20)

    # Plot
    fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax in [ax_pred, ax_gt]:
        for i, poly in enumerate(map_polylines):
            if polyline_masks[i] and poly.shape[0] > 0 and not np.allclose(poly, 0.0):
                ax.plot(poly[:, 0], poly[:, 1], color='gray', alpha=0.5, linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X (meters)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
    ax_pred.set_ylabel('Y (meters)')
    cmap = plt.get_cmap('hsv', max(A, 2))

    def plot_traj(ax, traj, label):
        for a in range(A):
            mask = trajectory_mask[a].astype(bool)
            if np.any(mask):
                x, y, theta = traj[a, mask, 0], traj[a, mask, 1], traj[a, mask, 2]
                ax.plot(x, y, "o", color=cmap(a % cmap.N), linewidth=1.5, alpha=0.7, label=f"{label} Agent {a}" if label else None)
                ax.arrow(x[0], y[0], 2.0 * np.cos(theta[0]), 2.0 * np.sin(theta[0]),
                         color=cmap(a % cmap.N), head_width=0.5, head_length=1.0, alpha=0.3, length_includes_head=True)
    plot_traj(ax_pred, pred_traj, 'Predicted')
    plot_traj(ax_gt, gt_traj, 'Ground Truth')

    ax_pred.set_title('Predicted Trajectories')
    ax_gt.set_title('Ground Truth Trajectories')
    sup_title = f'Future Trajectories (Ego ID: {ego_id})'
    if title_suffix: sup_title += f" - {title_suffix}"

    # Legend info
    legend_lines = []
    if epoch is not None:
        legend_lines.append(f"Epoch: {epoch}")
    if num_train_files is not None:
        legend_lines.append(f"Train Files: {num_train_files}")
    if step is not None:
        legend_lines.append(f"Step: {step}")
    legend_lines.append(f"Agents: {A}")
    if extra_info is not None:
        for k, v in extra_info.items():
            legend_lines.append(f"{k}: {v}")

    # Add legend box to both axes
    legend_text = "\n".join(legend_lines)
    for ax in [ax_pred, ax_gt]:
        ax.text(
            0.98, 0.02, legend_text,
            transform=ax.transAxes,
            fontsize=10, color='black',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    fig.suptitle(sup_title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close(fig)
    return None

def sample_noise(inputs):
    """
    Adds noise to future parts of the trajectory and returns noise level sigma.

    Args:
        inputs (torch.Tensor): [B, A, T, F], with T >= 10

    Returns:
        sigma (torch.Tensor): [B], noise level per sample in batch
        noised_inputs (torch.Tensor): [B, A, T, F], with noise added from t=10 onwards
    """
    B, A, T, F = inputs.size()
    T_obs = 10
    T_future = T - T_obs

    # Sample log sigma and convert to sigma
    ln_sigma = torch.normal(mean=-1.2, std=1.2, size=(B,), device=inputs.device)  # [B]
    sigma = torch.exp(ln_sigma)  # [B]

    # Noise tensor
    epsilon = torch.randn(B, A, T_future, F, device=inputs.device)  # [B, A, T_future, F]

    # Clone inputs and add noise to future timesteps
    noised_inputs = inputs.clone()
    noised_inputs[:, :, T_obs:, :] += epsilon * sigma[:, None, None, None]  # broadcast sigma

    return sigma, noised_inputs


def embed_features(inputs, sigma, embedding_dim=256, T_obs=10, eval=False):
    """
    Embed features using sinusoidal positional encodings for diffusion time tau, scenario time t,
    and agent states x, y, theta.
    
    Args:
        inputs (torch.Tensor): Input tensor [B, A, T, F], where F=3 for x, y, theta.
                              Contains observed states (xobs) for t<T_obs and noisy states (xlat,τ) for t>=T_obs.
        sigma (torch.Tensor): Noise levels [B,], one per batch.
        embedding_dim (int): Dimension of each encoding vector, default 256.
        T_obs (int): Number of observed time steps (default: 20).
        eval (bool): Evaluation mode flag.
    
    Returns:
        torch.Tensor: Embedded features [B, A, T, 5 * embedding_dim]
    """
    # Extract dimensions
    B, A, T, F = inputs.size()
    assert F == 3, "Expected F=3 for x, y, theta"
    assert embedding_dim % 2 == 0, "embedding_dim must be even"
    device = inputs.device

    # Generate scenario time t as integers [0, T-1]
    t = torch.arange(0, T, dtype=torch.float32, device=device)  # [T,]

    # Helper function for sinusoidal encoding
    def sinusoidal_encoding(values, min_period, max_period):
        num_freqs = embedding_dim // 2  # e.g., 128 for embedding_dim=256
        i = torch.arange(num_freqs, device=device, dtype=torch.float32)
        exp_term = i / (num_freqs - 1) if num_freqs > 1 else torch.zeros_like(i)
        wavelengths = min_period * (max_period / min_period) ** exp_term
        angular_freqs = 2 * torch.pi / wavelengths
        phases = values[..., None] * angular_freqs
        sin_enc = torch.sin(phases)
        cos_enc = torch.cos(phases)
        encoding = torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)  # [..., embedding_dim]
        return encoding

    # 1. Encode agent state features (x, y, theta)
    state_encodings = []
    for f in range(F):  # F=3
        feat = inputs[:, :, :, f]  # [B, A, T]
        enc = sinusoidal_encoding(feat, min_period=0.01, max_period=10)  # [B, A, T, embedding_dim]
        state_encodings.append(enc)

    # 2. Encode scenario time t
    t_enc = sinusoidal_encoding(t, min_period=1, max_period=100)  # [T, embedding_dim]
    t_enc = t_enc.expand(B, A, T, embedding_dim)  # [B, A, T, embedding_dim]

    # 3. Encode diffusion time tau
   # if not eval:
   #     sigma = sigma.squeeze()  # [B,]
    # Create tau tensor: 0 for t < T_obs, sigma for t >= T_obs
    latent_mask = (t >= T_obs).float()  # [T,], 0 for observed, 1 for latent
    tau = sigma[:, None, None] * latent_mask[None, None, :]  # [B, 1, T]
    tau = tau.expand(B, A, T)  # [B, A, T], broadcast across agents
    tau_enc = sinusoidal_encoding(tau, min_period=0.1, max_period=10000)  # [B, A, T, embedding_dim]

    # Concatenate all encodings: x, y, theta, t, tau
    all_encodings = state_encodings + [t_enc, tau_enc]  # 5 tensors, each [B, A, T, embedding_dim]
    embedded = torch.cat(all_encodings, dim=-1)  # [B, A, T, 5 * embedding_dim], e.g., [B, A, T, 1280]
    
    return embedded

if __name__ == "__main__":
    print("all good!")
    data = generate_dummy_inputs()
    embedded = embed_features(data["inputs"], data["sigma"])
    print(embedded.shape)
    plot_trajectories(
        data["map_polylines"],
        data["polyline_masks"],
        data["pred_traj"],
        data["gt_traj"],
        data["noisy_traj"],
        data["trajectory_mask"],
        data["ego_id"]
    )