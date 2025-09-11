
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

from map_pre_old import MapDataset
from networks_2 import Denoiser
from utils import plot_trajectories, sample_noise, embed_features
from infer_2 import calculate_validation_loss_and_plot

import time

start_time = time.time()
# ------------------- Helper: Validation-Plot speichern ------------------- #
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

end_time=time.time()
print(f"Time taken to define save_val_fig: {end_time - start_time} seconds")

# ------------------- Device & DataLoader Settings ------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO: check which cuda is the best one  and print during the training and vali which is being used 

def _safe_num_workers():
    return 1 if torch.cuda.is_available() else 0


# ------------------- Training ------------------- #
def train(model, writer):
    start_time = time.time()
    dataset = MapDataset(
        xml_dir="/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test",
        obs_len=10, pred_len=20, max_radius=100,
        num_timesteps=30, num_polylines=500, num_points=10,
        save_plots=False, max_agents=32
    )
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        num_workers=_safe_num_workers(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0)  # Warmup setzt LR später
    end_time=time.time()
    print(f"Time taken to set up dataset and dataloader: {end_time - start_time} seconds")

    steps_per_epoch = len(dataloader)
    ramp_up_steps = max(1, int(0.1 * steps_per_epoch))
    target_lr = 3e-4
    sigma_data = 0.5

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    num_epochs = 200
    global_step = 0
    parts = 5
    milestones = sorted({math.ceil(num_epochs * i / parts) for i in range(1, parts + 1)})

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")  
        print(device) 
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            (ego_ids, feature_tensor, feature_mask,
             roadgraph_tensor, roadgraph_mask,
             observed, observed_masks,
             ground_truth, ground_truth_masks,
             scene_means, scene_stds) = batch

            feature_tensor = feature_tensor.to(device)
            feature_mask   = feature_mask.to(device)
            roadgraph_tensor = roadgraph_tensor.to(device)
            roadgraph_mask   = roadgraph_mask.to(device)
            scene_means = scene_means.to(device)
            scene_stds  = scene_stds.to(device)
            start_time = time.time()
            # Noise-Training (EDM-ähnlich)
            sigma, noised_tensor = sample_noise(feature_tensor)  # sigma: [B]

            c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)                # [B]
            c_out  = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2) # [B]
            c_in   = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)                # [B]

            # Eingabe skalieren (nur Zukunft)
            result = noised_tensor.clone()
            c_in_broadcast = c_in[:, None, None, None]
            result[:, :, 10:, :] = c_in_broadcast * noised_tensor[:, :, 10:, :]

            # WICHTIG: embed_features erwartet σ, NICHT c_noise
            embedded = embed_features(result, sigma)  # σ als [B] passt

            model_out = model(embedded, roadgraph_tensor, feature_mask, roadgraph_mask)[:, :, 10:, :]  # [B,A,20,F]

            gt_pred   = feature_tensor[:, :, 10:, :]                  # [B,A,20,F]
            mask_pred = feature_mask[:, :, 10:]                       # [B,A,20]
            valid_mask = mask_pred.unsqueeze(-1).expand_as(gt_pred)   # [B,A,20,F]
            end_time = time.time()
            print(f"Time taken for forward pass: {end_time - start_time} seconds")
            start_time = time.time()
            # Rekonstruktion (Denoiser-Formel)
            recon = model_out * c_out[:, None, None, None] + noised_tensor[:, :, 10:, :] * c_skip[:, None, None, None]

            squared_diff = (recon - gt_pred) ** 2
            masked_squared_diff = squared_diff * valid_mask.float()

            loss_per_batch = masked_squared_diff.sum(dim=[1, 2, 3]) / valid_mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)

            # EDM Loss-Gewichtung
            weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
            weighted_loss = (loss_per_batch * weight).mean()
            end_time = time.time()
            print(f"Time taken for loss computation: {end_time - start_time} seconds")
            start_time = time.time()
            # Logging (robust)
            writer.add_scalar("Loss/WeightMean", weight.mean().item(), global_step)

            optimizer.zero_grad()
            weighted_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            writer.add_scalar("GradNorm", float(grad_norm), global_step)

            # Warmup LR
            lr = (global_step / ramp_up_steps) * target_lr if global_step < ramp_up_steps else target_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            writer.add_scalar("LR", lr, global_step)

            optimizer.step()
            end_time = time.time()
            print(f"Time taken for optimization step: {end_time - start_time} seconds")
            start_time = time.time()
            writer.add_scalar("Loss/Iteration", weighted_loss.item(), global_step)
            writer.add_scalar("Sigma/Example0", sigma[0].item(), global_step)

            train_losses.append(weighted_loss.item())
            epoch_loss += weighted_loss.item()
            num_batches += 1
            global_step += 1
            end_time = time.time()
            print(f"Time taken for one batch: {end_time - start_time} seconds") 

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        writer.add_scalar("Loss/Epoch", avg_epoch_loss, epoch)
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")

        # ------------------- Validation (alle 5 Epochen) ------------------- #
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}_REAL_500_poly_10_pts.pt")

            avg_val_loss, val_fig = calculate_validation_loss_and_plot(
                model=model,
                val_xml_dir="/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test",
                val_batch_size=4,
                obs_len=10, pred_len=20,
                max_radius=100,
                num_polylines=500, num_points=10,
                max_agents=32,
                sigma_data=0.5,
            )

            print(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")
            if not np.isnan(avg_val_loss):
                val_losses.append(avg_val_loss)
                writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            if (epoch + 1) in milestones:
             
                # Speichern (max. 5x pro Run) + schließen
                save_val_fig(val_fig, epoch=epoch, writer=writer)
                if val_fig is not None:
                 plt.close(val_fig)

    # ------------------- Zusammenfassung: Loss-Plot ------------------- #
  # === Loss curves: one figure, two graphs ===
    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(12, 5))

# Training loss (per iteration)
    ax_tr.plot(range(len(train_losses)), train_losses, label="Training Loss")
    ax_tr.set_xlabel("Iteration")
    ax_tr.set_ylabel("Loss")
    ax_tr.set_title("Training Loss (per iteration)")
    ax_tr.grid(True, linestyle="--", alpha=0.5)
    ax_tr.legend()

# Validation loss (every 5 epochs)
    if val_losses:
    # map each validation point to its epoch index (0,5,10,...) and to iteration index
     epochs_with_val = list(range(0, num_epochs, 5))[:len(val_losses)]
     iter_with_val = [e * steps_per_epoch for e in epochs_with_val]
     ax_val.plot(iter_with_val, val_losses, marker="o", label="Validation Loss")
     ax_val.set_xlabel("Iteration (epoch × steps_per_epoch)")
     ax_val.set_title("Validation Loss (every 5 epochs)")
     ax_val.grid(True, linestyle="--", alpha=0.5)
     ax_val.legend()
    else:
     ax_val.set_axis_off()
     ax_val.text(0.5, 0.5, "No validation data yet", ha="center", va="center")

    plt.suptitle("Training & Validation Loss", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("training_validation_losses.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)



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
