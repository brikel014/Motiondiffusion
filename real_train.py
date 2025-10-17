import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import amp

from map_pre_old import MapDataset
from networks_2 import Denoiser
from infer_2 import calculate_validation_loss_and_plot  # unchanged import
from utils import embed_features

# ------------------- Helper: Validation-Plot speichern ------------------- #
# Function to save validation plots, with a limit on the number of saves per run.
def save_val_fig(fig, epoch, writer=None, max_saves=5, base_dir="runs/val_plots"):
    """Speichert höchstens `max_saves` Validierungsplots pro Training in einem Run-Ordner."""
    if fig is None:
        warnings.warn(f"[VAL] Keine Figure erzeugt (epoch={epoch}).")
        return
    # state am Funktionsobjekt (einmalig pro Run)
    if not hasattr(save_val_fig, "_dir"):
        ts = time.strftime("%Y%m%d-%H%M%S")
        save_val_fig._dir = os.path.join(base_dir, f"train_{ts}")
        os.makedirs(save_val_fig._dir, exist_ok=True)
        save_val_fig._count = 0
        print(f"[VAL] Plots werden gespeichert unter: {save_val_fig._dir}")
    if save_val_fig._count >= max_saves:
        return
    out_path = os.path.join(save_val_fig._dir, f"val_epoch_{int(epoch):03d}.png")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[VAL] Plot saved -> {out_path}")
    save_val_fig._count += 1
    if writer is not None:
        writer.add_figure(f"Inference/Validation_Epoch_{epoch}", fig, epoch)

# ------------------- Device & DataLoader Settings ------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None
def _safe_num_workers():
    return 1 if torch.cuda.is_available() else 0


# ------------------- Conditioning helpers (minimal, additive) ------------------- #
EMBED_DX = 5 * 256  # utils.embed_features default => 5 * embedding_dim(=256) = 1280 
P_UNCOND  = 0.20    # classifier-free guidance drop prob
TURN_THRESH_DEG = 10.0  # how much θ change to call left/right

class CondProj(torch.nn.Module):
    """One-hot(3) → Dx (1280) via small MLP. Output size = EMBED_DX to allow additive conditioning."""
    def __init__(self, out_dim=EMBED_DX):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, out_dim)
        )
    def forward(self, y_onehot):  # [B,A,3] or [B,3]
        if y_onehot.dim() == 2:
            y_onehot = y_onehot.unsqueeze(1)  # [B,1,3]
        return self.net(y_onehot)  # [B,A,Dx]

def _angle_wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi

@torch.no_grad()
# Convert future angles to one-hot direction commands.
# The Function take B,A,T tensor and returns B,A,3 one-hot tensor
def dir_onehot_from_theta(theta_future: torch.Tensor, thresh_deg: float = TURN_THRESH_DEG) -> torch.Tensor:
    """
    theta_future: [B,A,Tp], in radians. Returns one-hot [B,A,3] = [right, straight, left].
    Uses summed Δtheta over the future horizon with wrap handling.
    """
    dtheta = _angle_wrap(theta_future[..., 1:] - theta_future[..., :-1])  # [B,A,Tp-1]
    total = dtheta.sum(dim=-1)  # [B,A]
    th = math.radians(thresh_deg)
    left = (total > th)
    right = (total < -th)
    straight = ~(left | right)
    y = torch.zeros((*total.shape, 3), device=theta_future.device, dtype=torch.float32)
    y[..., 0] = right.float()
    y[..., 1] = straight.float()
    y[..., 2] = left.float()
    return y
# Convert one-hot direction command back to string labels in German.
def _drop_condition(y_onehot: torch.Tensor, p_uncond: float = P_UNCOND) -> torch.Tensor:
    """Classifier-free guidance: randomly drop the condition to zero."""
    drop = (torch.rand(y_onehot.shape[:2], device=y_onehot.device) < p_uncond)  # [B,A]
    y = y_onehot.clone()
    y[drop] = 0.0
    return y
# Add condition (same width as x_embed) only on the future timesteps.
def _add_condition_to_embed(x_embed: torch.Tensor, cond_time: torch.Tensor, obs_len: int) -> torch.Tensor:
    """
    Add condition (same width as x_embed) only on the future timesteps.
      x_embed: [B,A,T,Dx], cond_time: [B,A,Tp,Dx]
    """
    out = x_embed
    out[:, :, obs_len:, :] = out[:, :, obs_len:, :] + cond_time
    return out


# ------------------- Training ------------------- #
def train(model, writer):
    start_time = time.time()
    print(f"Using device: {device}")
    
    # Improves performance for consistent input sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    dataset = MapDataset(
        xml_dir="/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test",
        obs_len=10, pred_len=20, max_radius=100,
        num_timesteps=30, num_polylines=500, num_points=10,
        save_plots=False, max_agents=32
    )
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        num_workers=_safe_num_workers(), pin_memory=(device.type=='cuda'),
        persistent_workers=False
    )

    model = model.to(device)
    model.train()

    # NEW: condition projector (tiny MLP). We keep Denoiser unchanged.
    cond_proj = CondProj(out_dim=EMBED_DX).to(device)

    # Optimizer now includes cond_proj params (rest unchanged)
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.add_param_group({"params": cond_proj.parameters(), "lr": 0.0})  # same LR schedule

    end_time=time.time()
    print(f"Time taken to set up dataset and dataloader: {end_time - start_time} seconds")

    steps_per_epoch = len(dataloader)
    ramp_up_steps = max(1, int(0.1 * steps_per_epoch))
    target_lr = 3e-4
    sigma_data = 0.5

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,} (+ {sum(p.numel() for p in cond_proj.parameters()):,} cond)")

    num_epochs = 1000
    global_step = 0
    parts = 5
    milestones = sorted({math.ceil(num_epochs * i / parts) for i in range(1, parts + 1)})

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")  
        print(device) 
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            (ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask,
             observed, observed_masks, ground_truth, ground_truth_masks,
             scene_means, scene_stds) = batch

            feature_tensor = feature_tensor.to(device, non_blocking=True)   # [B,A,T,3]  (x,y,theta)
            feature_mask   = feature_mask.to(device, non_blocking=True)     # [B,A,T]
            roadgraph_tensor = roadgraph_tensor.to(device, non_blocking=True)
            roadgraph_mask   = roadgraph_mask.to(device, non_blocking=True)
            scene_means = scene_means.to(device, non_blocking=True)
            scene_stds  = scene_stds.to(device, non_blocking=True)

            B, A, T, F = feature_tensor.shape
            assert F == 3, "Expected x,y,theta as 3 features"
            To = 10  # obs_len fixed in dataset above
            Tp = T - To

            global_step += 1
            optimizer.zero_grad()
            
            with amp.autocast('cuda', enabled=(device.type == 'cuda')):

                # Noise-Training (EDM-ähnlich)
                sigma, noised_tensor = sample_noise(feature_tensor)

                c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
                c_out  = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
                c_in   = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)
                c_noise = torch.log(sigma) * 0.25  # [B]
                # Precondition the latent (future) part only (your original logic)
                result = noised_tensor.clone()
                c_in_broadcast = c_in[:, None, None, None]
                result[:, :, To:, :] = c_in_broadcast * noised_tensor[:, :, To:, :]

                # ---- NEW: build direction condition from future theta ----
                theta_future = feature_tensor[:, :, To:, 2]  # [B,A,Tp]
                y_onehot = dir_onehot_from_theta(theta_future, TURN_THRESH_DEG)  # [B,A,3]
                y_cf = _drop_condition(y_onehot, P_UNCOND)                        # CFG drop

                # Embed features (unchanged API)
                embedded = embed_features(result, c_noise)  # [B,A,T,EMBED_DX]

                # Project condition to Dx and add only on the future window (no shape change)
                cond_embed = cond_proj(y_cf)                             # [B,A,EMBED_DX]
                cond_time  = cond_embed.unsqueeze(2).expand(-1,-1,Tp,-1) # [B,A,Tp,EMBED_DX]
                embedded   = _add_condition_to_embed(embedded, cond_time, To)

                # Forward pass (unchanged)
                model_out = model(embedded, roadgraph_tensor, feature_mask, roadgraph_mask)[:, :, To:, :]

                gt_pred   = feature_tensor[:, :, To:, :]
                mask_pred = feature_mask[:, :, To:]
                valid_mask = mask_pred.unsqueeze(-1).expand_as(gt_pred)

                recon = model_out * c_out[:, None, None, None] + noised_tensor[:, :, To:, :] * c_skip[:, None, None, None]

                squared_diff = (recon - gt_pred) ** 2
                masked_squared_diff = squared_diff * valid_mask

                loss_per_batch = masked_squared_diff.sum(dim=[1, 2, 3]) / valid_mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)

                weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
                weighted_loss = (loss_per_batch * weight).mean()
            
            if scaler:
                scaler.scale(weighted_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                weighted_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
            writer.add_scalar("GradNorm", float(grad_norm), global_step)

            lr = (global_step / ramp_up_steps) * target_lr if global_step < ramp_up_steps else target_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            writer.add_scalar("LR", lr, global_step)
            
            writer.add_scalar("Loss/Iteration", weighted_loss.item(), global_step)
            writer.add_scalar("Sigma/Example0", sigma[0].item(), global_step)

            train_losses.append(weighted_loss.item())
            epoch_loss += weighted_loss.item()
            num_batches += 1
            global_step += 1

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        writer.add_scalar("Loss/Epoch", avg_epoch_loss, epoch)
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")

        # ------------------- Validation (alle 20 Epochen) --------------
        if epoch % 20 == 0:
            
            ckpt = {
                "model": model.state_dict(),
                "cond_proj": cond_proj.state_dict(),
                "epoch": epoch,
            }
            os.makedirs("ckpts", exist_ok=True)
            torch.save(ckpt, f"ckpts/model_epoch_{epoch}_REAL.pt")

            avg_val_loss, val_fig = calculate_validation_loss_and_plot(
                model=model,
                val_xml_dir="/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test",
                val_batch_size=32,
                obs_len=10, pred_len=20,
                max_radius=100,
                num_polylines=500, num_points=10,
                max_agents=32,
                sigma_data=0.5,
                device=device,
                direction_command=True,             # see next point
                cond_scale=2.0,
                turn_thresh_deg=TURN_THRESH_DEG,    # <— 10.0 to match training
                cond_proj_state_dict=cond_proj.state_dict(),  # <— use the trained projector
            )

            if not np.isnan(avg_val_loss):
                val_losses.append(avg_val_loss)
                writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
                save_val_fig(val_fig, epoch, writer=writer)

    print("Training finished.")


# ------------------- Noise sampler (your original helper) ------------------- #
def sample_noise(feature_tensor):
    """
    Return per-batch sigma and a noised tensor, compatible with your training loop.
    """
    B = feature_tensor.size(0)
    # EDM-like sampling — keep whatever you had; placeholder:
    sigma_min, sigma_max, rho = 0.002, 20.0, 7.0
    u = torch.rand(B, device=feature_tensor.device)
    sigma = (sigma_max**(1/rho) + u * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho  # [B]
    noise = torch.randn_like(feature_tensor)
    noised = feature_tensor.clone()
    noised[:, :, 10:, :] = feature_tensor[:, :, 10:, :] + sigma[:, None, None, None] * noise[:, :, 10:, :]
    return sigma, noised


# ------------------- Main ------------------- #
if __name__ == "__main__":
    writer = None
    try:
        writer = SummaryWriter(log_dir="./runs_5/REAL")
        model = Denoiser()
        train(model, writer)
    finally:
        if writer is not None:
            writer.close()
        print("Training completed and TensorBoard writer closed.")
