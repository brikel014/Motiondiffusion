import torch
from torch import nn
from map_pre_old import MapDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks_2 import Denoiser
from utils import plot_trajectories, sample_noise, embed_features
import matplotlib.pyplot as plt
import numpy as np
from infer_2 import calculate_validation_loss_and_plot

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, writer):
    dataset = MapDataset(
        xml_dir='/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test',
        obs_len=10,
        pred_len=20,
        max_radius=100,
        num_timesteps=30,
        num_polylines=500,
        num_points=10,
        save_plots=False,
        max_agents=32
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0)

    steps_per_epoch = len(dataloader)
    ramp_up_steps = int(0.1 * steps_per_epoch)
    target_lr = 3e-4

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    num_epochs = 20
    global_step = 0

    # Arrays zum Tracken der Losses
    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask, observed, observed_masks, ground_truth, ground_truth_masks, scene_means, scene_stds = batch

            feature_tensor = torch.cat([feature_tensor[:, :, :10, :], feature_tensor[:, :, 10:, :]], dim=2).to(device)
            feature_mask = torch.cat([feature_mask[:, :, :10], feature_mask[:, :, 10:]], dim=2).to(device)

            roadgraph_tensor = roadgraph_tensor.to(device)
            roadgraph_mask = roadgraph_mask.to(device)
            ground_truth = ground_truth.to(device)
            scene_means = scene_means.to(device)
            scene_stds = scene_stds.to(device)

            sigma, noised_tensor = sample_noise(feature_tensor)
            c_skip = 0.5**2 / (sigma**2 + (0.5**2))
            c_out = sigma * 0.5 / torch.sqrt((0.5**2) + sigma**2)
            c_in = 1 / torch.sqrt((sigma**2) + 0.5**2)
            c_noise = 1/4 * torch.log(sigma)

            result = noised_tensor.clone()
            c_in = c_in[:, None, None, None] 
            result[:, :, 10:, :] = c_in * noised_tensor[:, :, 10:, :]
            embedded = embed_features(result, c_noise)
            model_out = model(embedded, roadgraph_tensor, feature_mask, roadgraph_mask)[:, :, 10:, :]

            gt_pred = feature_tensor[:, :, 10:, :]
            mask_pred = feature_mask[:, :, 10:]
            valid_mask = mask_pred.unsqueeze(-1).expand_as(gt_pred)

            squared_diff = (model_out * c_out[:, None, None, None] + noised_tensor[:, :, 10:, :] * c_skip[:, None, None, None]  - gt_pred) ** 2
            masked_squared_diff = squared_diff * valid_mask.float()
            loss_per_batch = masked_squared_diff.sum(dim=[1, 2, 3]) / valid_mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)

            sigma_data = 0.5
            weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
            weighted_loss = (loss_per_batch * weight).mean()

            writer.add_scalar("Loss/Weight", weight[0].item(), global_step)

            optimizer.zero_grad()
            weighted_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            writer.add_scalar("GradNorm", grad_norm.item(), global_step)

            if global_step < ramp_up_steps:
                lr = (global_step / ramp_up_steps) * target_lr
            else:
                lr = target_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()

            writer.add_scalar("Loss/Iteration", weighted_loss.item(), global_step)
            writer.add_scalar("Sigma", sigma[0].item(), global_step)

            # Training Loss speichern
            train_losses.append(weighted_loss.item())
            global_step += 1

            epoch_loss += weighted_loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        writer.add_scalar("Loss/Epoch", avg_epoch_loss, i)
        print(f"Epoch {i} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Validation Loss alle 5 Epochen
        if i % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{i}_REAL_500_poly_10_pts.pt")
            avg_val_loss, val_fig = calculate_validation_loss_and_plot(
                model=model,
                val_xml_dir='/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test',
                val_batch_size=4,
                obs_len=10,
                pred_len=20,
                max_radius=100,
                num_polylines=500,
                num_points=10,
                max_agents=32,
                sigma_data=0.5
            )
            print(f"Epoch {i}: Validation Loss = {avg_val_loss:.4f}")

            # Validation Loss speichern
            if not np.isnan(avg_val_loss):
                val_losses.append(avg_val_loss)
                writer.add_scalar("Loss/Validation", avg_val_loss, i)
            if val_fig:
                writer.add_figure(f"Inference/Validation_Epoch_{i}", val_fig, i)
                plt.close(val_fig)

    # Plot der Training Losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        val_x = [j * steps_per_epoch * 5 for j in range(len(val_losses))]  # alle 5 Epochen
        plt.plot(val_x, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_plot.png")
    plt.show()
    plt.close()

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