"""
Modified utilities for trajectory prediction and plotting.

This module contains functions for plotting predicted and ground‑truth
trajectories, sampling diffusion noise, and embedding features for a
diffusion model.  The `plot_trajectories` function has been updated
to avoid recursion errors during input validation by safely converting
inputs to numeric NumPy arrays before checking for NaN or Inf values.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Set device for PyTorch operations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _check_nan_inf(x):
    """
    Safely check whether a given input contains NaN or infinite values.

    Attempts to coerce the input to a float64 NumPy array.  If
    conversion fails or triggers a recursion error, the function
    prints a warning and returns (False, False).

    Args:
        x: Any object that should be checked for NaNs or Infs.

    Returns:
        Tuple of booleans (has_nan, has_inf).
    """
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError, RecursionError) as e:
        print(f"Warning: could not convert object of type {type(x).__name__} to a float array for NaN/Inf check ({e}).")
        return False, False
    return np.isnan(arr).any(), np.isinf(arr).any()


def plot_trajectories(
    map_polylines,
    polyline_masks,
    pred_traj,
    gt_traj,
    noisy_traj,
    trajectory_mask,
    ego_id,
    save_path=None,
    eval=False,
    title_suffix="",
):
    """
    Plot predicted and ground truth trajectories side by side in a single figure
    for all agents.  Handles both PyTorch tensors and NumPy arrays as input.

    Args:
        map_polylines (torch.Tensor or np.ndarray): Map polylines of shape
            ``[P, N, 2]`` (x, y coordinates).
        polyline_masks (torch.Tensor or np.ndarray): Mask for valid polylines
            of shape ``[P]``.
        pred_traj (torch.Tensor or np.ndarray): Predicted trajectories of shape
            ``[A, T_future, 3]`` (x, y, theta).
        gt_traj (torch.Tensor or np.ndarray): Ground truth trajectories of
            shape ``[A, T_future, 3]``.
        noisy_traj (torch.Tensor or np.ndarray): Noisy trajectories of shape
            ``[A, T_future, 3]``.  Can be used for visualisation.
        trajectory_mask (torch.Tensor or np.ndarray): Mask for valid
            trajectory timesteps of shape ``[A, T_future]``.
        ego_id (int or str): Identifier for the ego agent.
        save_path (str, optional): Base path (without extension) to save the
            generated figure.  If ``None`` and ``eval`` is ``False``, the
            function will display the figure instead of saving it.
        eval (bool): If ``True``, the function returns the ``matplotlib``
            ``Figure`` object instead of showing or saving it.  Useful for
            integration with TensorBoard.
        title_suffix (str, optional): Additional text appended to the figure
            title.

    Returns:
        matplotlib.figure.Figure or None: If ``eval`` is ``True``, returns
        the figure object.  Otherwise, returns ``None``.
    """
    # --- Convert inputs to NumPy if they are tensors ---
    if isinstance(map_polylines, torch.Tensor):
        map_polylines = map_polylines.detach().cpu().numpy()
    if isinstance(polyline_masks, torch.Tensor):
        polyline_masks = polyline_masks.detach().cpu().numpy()
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.detach().cpu().numpy()
    if isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.detach().cpu().numpy()
    if isinstance(noisy_traj, torch.Tensor):
        noisy_traj = noisy_traj.detach().cpu().numpy()
    if isinstance(trajectory_mask, torch.Tensor):
        trajectory_mask = trajectory_mask.detach().cpu().numpy()

    # --- Input Validation: check for NaN/Inf using safe helper ---
    for name, arr in [
        ("map_polylines", map_polylines),
        ("pred_traj", pred_traj),
        ("gt_traj", gt_traj),
        ("noisy_traj", noisy_traj),
    ]:
        has_nan, has_inf = _check_nan_inf(arr)
        if has_nan:
            print(f"Warning: {name} contains NaN values")
        if has_inf:
            print(f"Warning: {name} contains infinite values")

    # Number of agents
    A = pred_traj.shape[0]

    # --- Compute plot limits based on map and valid trajectories ---
    valid_polylines_list = [
        map_polylines[i] for i in range(map_polylines.shape[0]) if polyline_masks[i]
    ]
    all_x_road = (
        np.concatenate([p[:, 0] for p in valid_polylines_list if p.shape[0] > 0])
        if valid_polylines_list
        else np.array([])
    )
    all_y_road = (
        np.concatenate([p[:, 1] for p in valid_polylines_list if p.shape[0] > 0])
        if valid_polylines_list
        else np.array([])
    )

    all_x_traj, all_y_traj = [], []
    for a in range(A):
        # Use Python's built‑in any() to avoid NumPy internals that may
        # trigger issues in some environments.  valid_mask_a is a 1‑D
        # boolean array indicating which timesteps are valid.
        valid_mask_a = trajectory_mask[a].astype(bool)
        if any(valid_mask_a):
            all_x_traj.append(pred_traj[a, valid_mask_a, 0])
            all_y_traj.append(pred_traj[a, valid_mask_a, 1])
            all_x_traj.append(gt_traj[a, valid_mask_a, 0])
            all_y_traj.append(gt_traj[a, valid_mask_a, 1])

    all_x_traj_np = np.concatenate(all_x_traj) if all_x_traj else np.array([])
    all_y_traj_np = np.concatenate(all_y_traj) if all_y_traj else np.array([])

    all_x = np.concatenate([all_x_road, all_x_traj_np])
    all_y = np.concatenate([all_y_road, all_y_traj_np])

    # Determine axis limits, handling empty data gracefully
    if all_x.size > 0 and all_y.size > 0:
        finite_x = all_x[np.isfinite(all_x)]
        finite_y = all_y[np.isfinite(all_y)]
        if finite_x.size > 0 and finite_y.size > 0:
            xmin, xmax = np.min(finite_x), np.max(finite_x)
            ymin, ymax = np.min(finite_y), np.max(finite_y)
            x_range = max(xmax - xmin, 1.0)
            y_range = max(ymax - ymin, 1.0)
            padding = max(x_range, y_range) * 0.1
            xlim = (xmin - padding, xmax + padding)
            ylim = (ymin - padding, ymax + padding)
        else:
            print(
                "Warning: No finite data points found for limit calculation. Using default limits."
            )
            xlim = (-20, 20)
            ylim = (-20, 20)
    else:
        print(
            "Warning: No map or trajectory data to determine plot limits. Using default limits."
        )
        xlim = (-20, 20)
        ylim = (-20, 20)

    # --- Plotting set‑up ---
    fig, (ax_pred, ax_gt) = plt.subplots(
        1, 2, figsize=(12, 6), sharex=True, sharey=True
    )

    # Plot map polylines on both axes
    for ax in [ax_pred, ax_gt]:
        for i in range(map_polylines.shape[0]):
            if polyline_masks[i]:
                poly = map_polylines[i]
                if poly.shape[0] > 0 and not np.allclose(poly, 0.0):
                    ax.plot(poly[:, 0], poly[:, 1], color="gray", alpha=0.5, linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("X (meters)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
    ax_pred.set_ylabel("Y (meters)")

    # Colour map for different agents
    cmap = plt.get_cmap("hsv", A if A > 1 else 2)

    # Plot predicted trajectories on the left
    for a in range(A):
        valid_mask_a = trajectory_mask[a].astype(bool)
        # Use Python's built‑in any() instead of np.any() to avoid NumPy internal
        # reductions that may fail if NumPy is misconfigured.
        if any(valid_mask_a):
            x = pred_traj[a, valid_mask_a, 0]
            y = pred_traj[a, valid_mask_a, 1]
            theta = pred_traj[a, valid_mask_a, 2]
            ax_pred.plot(
                x, y, color=cmap(a % cmap.N), linewidth=1.5, alpha=0.9
            )
            x_start, y_start, theta_start = x[0], y[0], theta[0]
            arrow_length = 2.0
            ax_pred.arrow(
                x_start,
                y_start,
                arrow_length * np.cos(theta_start),
                arrow_length * np.sin(theta_start),
                color=cmap(a % cmap.N),
                head_width=0.5,
                head_length=1.0,
                alpha=0.9,
                length_includes_head=True,
            )

    # Plot ground‑truth trajectories on the right
    for a in range(A):
        valid_mask_a = trajectory_mask[a].astype(bool)
        if any(valid_mask_a):
            x = gt_traj[a, valid_mask_a, 0]
            y = gt_traj[a, valid_mask_a, 1]
            theta = gt_traj[a, valid_mask_a, 2]
            ax_gt.plot(
                x, y, color=cmap(a % cmap.N), linewidth=1.5, alpha=0.9
            )
            x_start, y_start, theta_start = x[0], y[0], theta[0]
            arrow_length = 2.0
            ax_gt.arrow(
                x_start,
                y_start,
                arrow_length * np.cos(theta_start),
                arrow_length * np.sin(theta_start),
                color=cmap(a % cmap.N),
                head_width=0.5,
                head_length=1.0,
                alpha=0.9,
                length_includes_head=True,
            )

    # Titles and layout adjustments
    ax_pred.set_title("Predicted Trajectories")
    ax_gt.set_title("Ground Truth Trajectories")
    sup_title = f"Future Trajectories (Ego ID: {ego_id})"
    if title_suffix:
        sup_title += f" - {title_suffix}"
    fig.suptitle(sup_title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Output handling ---
    if eval:
        return fig
    elif save_path:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                fig.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}.png")
        except Exception as e:
            print(f"Failed to save plot {save_path}.png: {e}")
        plt.close(fig)
        return None
    else:
        plt.show()
        plt.close(fig)
        return None


def sample_noise(inputs):
    """
    Adds noise to future parts of the trajectory and returns noise level sigma.

    Args:
        inputs (torch.Tensor): Tensor of shape ``[B, A, T, F]`` with ``T >= 10``.

    Returns:
        Tuple (sigma, noised_inputs): ``sigma`` is a tensor of shape ``[B]``
        containing the noise level per batch element, and ``noised_inputs``
        is the inputs with noise added to the future part of the trajectory.
    """
    B, A, T, F = inputs.size()
    T_obs = 10
    T_future = T - T_obs

    ln_sigma = torch.normal(mean=-1.2, std=1.2, size=(B,), device=inputs.device)
    sigma = torch.exp(ln_sigma)
    epsilon = torch.randn(B, A, T_future, F, device=inputs.device)
    noised_inputs = inputs.clone()
    noised_inputs[:, :, T_obs:, :] += epsilon * sigma[:, None, None, None]
    return sigma, noised_inputs


def embed_features(inputs, sigma, embedding_dim=256, T_obs=10, eval=False):
    """
    Embed features using sinusoidal positional encodings for diffusion time
    tau, scenario time t, and agent states x, y, theta.

    Args:
        inputs (torch.Tensor): Input tensor ``[B, A, T, F]`` with ``F=3``
            representing x, y, theta.  Contains observed states for
            ``t < T_obs`` and noisy states for ``t >= T_obs``.
        sigma (torch.Tensor): Noise levels ``[B]``, one per batch element.
        embedding_dim (int): Dimension of each encoding vector (must be even).
        T_obs (int): Number of observed time steps.
        eval (bool): Evaluation mode flag.

    Returns:
        torch.Tensor: Embedded features of shape ``[B, A, T, 5 * embedding_dim]``.
    """
    B, A, T, F = inputs.size()
    assert F == 3, "Expected F=3 for x, y, theta"
    assert embedding_dim % 2 == 0, "embedding_dim must be even"
    device = inputs.device

    # Scenario time indices
    t = torch.arange(0, T, dtype=torch.float32, device=device)

    def sinusoidal_encoding(values, min_period, max_period):
        num_freqs = embedding_dim // 2
        i = torch.arange(num_freqs, device=device, dtype=torch.float32)
        exp_term = i / (num_freqs - 1) if num_freqs > 1 else torch.zeros_like(i)
        wavelengths = min_period * (max_period / min_period) ** exp_term
        angular_freqs = 2 * torch.pi / wavelengths
        phases = values[..., None] * angular_freqs
        sin_enc = torch.sin(phases)
        cos_enc = torch.cos(phases)
        encoding = torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)
        return encoding

    # Encode agent state features (x, y, theta)
    state_encodings = []
    for f in range(F):
        feat = inputs[:, :, :, f]
        enc = sinusoidal_encoding(feat, min_period=0.01, max_period=10)
        state_encodings.append(enc)

    # Encode scenario time t
    t_enc = sinusoidal_encoding(t, min_period=1, max_period=100)
    t_enc = t_enc.expand(B, A, T, embedding_dim)

    # Encode diffusion time tau
    latent_mask = (t >= T_obs).float()
    tau = sigma[:, None, None] * latent_mask[None, None, :]
    tau = tau.expand(B, A, T)
    tau_enc = sinusoidal_encoding(tau, min_period=0.1, max_period=10000)

    all_encodings = state_encodings + [t_enc, tau_enc]
    embedded = torch.cat(all_encodings, dim=-1)
    return embedded


if __name__ == "__main__":
    print("utils_modified loaded successfully.")