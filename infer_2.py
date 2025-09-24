# infer_2.py (CFG + metrics, fixe Settings)
import time
import warnings
import math

import torch
import numpy as np
from torch.utils.data import DataLoader
from map_pre_old import MapDataset
from networks_2 import Denoiser
from utils import plot_trajectories, embed_features
import torch.cuda.amp as amp

# =========================
# FIXED SETTINGS (edit here)
# =========================
VAL_XML_DIR = "/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test"
OBS_LEN = 10
PRED_LEN = 20
NUM_TIMESTEPS = OBS_LEN + PRED_LEN
MAX_RADIUS = 100
NUM_POLYLINES = 500
NUM_POINTS = 10
MAX_AGENTS = 32

VAL_BATCH_SIZE = 2
SIGMA_DATA = 0.5

# Command & CFG
DIRECTION_COMMAND = "rechts"   # "rechts" | "links" | "gerade"
COND_SCALE = 2.0
COND_DIM = 128

# =========================
# Conditioning helpers (same as train)
# =========================
def angle_wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi

@torch.no_grad()
def onehot_from_command(cmd: str, device) -> torch.Tensor:
    cmd = cmd.strip().lower()
    v = torch.zeros(3, dtype=torch.float32, device=device)
    if cmd in ("r","right","rechts"): v[0] = 1.0
    elif cmd in ("s","straight","gerade","forward","vor"): v[1] = 1.0
    elif cmd in ("l","left","links"): v[2] = 1.0
    else: raise ValueError(f"Unknown command '{cmd}'")
    return v

class DiscreteCondProj(torch.nn.Module):
    def __init__(self, in_dim=3, hid=128, out_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.SiLU(),
            torch.nn.Linear(hid, out_dim)
        )
    def forward(self, y_onehot):
        if y_onehot.dim()==2: y_onehot = y_onehot.unsqueeze(1)
        return self.net(y_onehot)

class DirectionHead(torch.nn.Module):
    def __init__(self, hidden=128, bidirectional=True):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=2, hidden_size=hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.proj = torch.nn.Sequential(torch.nn.Linear(out_dim, out_dim), torch.nn.SiLU(), torch.nn.Linear(out_dim, 3))
    def forward(self, xy_future, mask=None):
        B,A,T,_ = xy_future.shape
        x = xy_future.view(B*A, T, 2)
        h,_ = self.gru(x)
        if mask is not None:
            m = mask.view(B*A, T, 1).float()
            h = (h*m).sum(1) / (m.sum(1).clamp_min(1.0))
        else:
            h = h.mean(1)
        return self.proj(h).view(B,A,3)

# =========================
# Utilities
# =========================
def _move_batch_to_device(batch, device):
    def _mv(x):
        if isinstance(x, torch.Tensor): return x.to(device, non_blocking=True)
        if isinstance(x, (list, tuple)): return type(x)(_mv(t) for t in x)
        if isinstance(x, dict): return {k: _mv(v) for k, v in x.items()}
        return x
    return _mv(batch)

def _assert_same_device(*tensors):
    devs = {str(t.device) for t in tensors if isinstance(t, torch.Tensor)}
    if len(devs) != 1:
        raise RuntimeError(f"Mixed devices in val batch: {devs}")

def _to_np(t: torch.Tensor):
    return t.detach().to('cpu').numpy()

def get_T_optimized(sigma_max: float, sigma_min: float, rho: float, N: int, device: torch.device):
    if N <= 1:
        return torch.as_tensor([sigma_max], device=device, dtype=torch.float32)
    t_vals = torch.arange(N, device=device, dtype=torch.float32)
    a = (sigma_min**(1.0 / rho) - sigma_max**(1.0 / rho)) / (N - 1)
    return (sigma_max**(1.0 / rho) + t_vals * a) ** rho

def _normalize_like_train(x, scene_means, scene_stds, eps=1e-6):
    return (x - scene_means[:, None, None, :]) / (scene_stds[:, None, None, :] + eps)

# =========================
# Validation with CFG command
# =========================
def calculate_validation_loss_and_plot(
    model,
    val_xml_dir,
    val_batch_size,
    obs_len, pred_len,
    max_radius,
    num_polylines, num_points,
    max_agents,
    sigma_data,
    direction_command: str = "rechts",
    cond_scale: float = 2.0,
    device=torch.device('cpu'),
):
    print(f"[VAL] Start: {val_xml_dir}")
    model = model.to(device).eval()
    torch.backends.cudnn.benchmark = (device.type == 'cuda')

    # Dataset/Loader (DEIN Stil)
    val_dataset = MapDataset(
        xml_dir=val_xml_dir,
        obs_len=obs_len, pred_len=pred_len, max_radius=max_radius,
        num_timesteps=obs_len + pred_len, num_polylines=num_polylines, num_points=num_points,
        save_plots=False, max_agents=max_agents
    )
    if len(val_dataset) == 0:
        warnings.warn(f"[VAL] Leeres Dataset: {val_xml_dir}")
        return float('nan'), None

    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    # Build projector/head on the fly (or load checkpoints if du willst)
    cond_proj = DiscreteCondProj(out_dim=COND_DIM).to(device).eval()
    dir_head  = DirectionHead().to(device).eval()

    sigma_data_t = torch.as_tensor(float(sigma_data), device=device)
    N_inference_steps = 50
    sigma_max, sigma_min, rho = 20.0, 0.002, 7.0

    total_loss = 0.0
    total_valid = 0.0
    total_acc  = 0.0
    total_count = 0
    fig_to_return = None
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        for batch_idx, batch in enumerate(val_loader):
            batch = _move_batch_to_device(batch, device)
            (ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask,
             observed, observed_masks, ground_truth, ground_truth_masks,
             scene_means, scene_stds) = batch

            _assert_same_device(feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask,
                                observed, observed_masks, ground_truth, ground_truth_masks,
                                scene_means, scene_stds)

            B, A, T, F = feature_tensor.shape
            Tp = pred_len
            T_inference = get_T_optimized(sigma_max, sigma_min, rho, N_inference_steps, device)

            gt_future_part = feature_tensor[:, :, obs_len:, :]     # [B,A,Tp,F]
            gt_future_mask = feature_mask[:, :, obs_len:]          # [B,A,Tp]
            observed_past  = feature_tensor[:, :, :obs_len, :]     # [B,A,To,F]

            obs_mask   = feature_mask[:, :, :obs_len]
            pred_mask0 = torch.zeros_like(feature_mask[:, :, obs_len:])
            full_mask_in = torch.cat([obs_mask, pred_mask0], dim=2)

            # Command conditioning
            cmd_onehot = onehot_from_command(direction_command, device)  # [3]
            y_cmd = cmd_onehot.view(1,1,3).expand(B, A, 3)               # [B,A,3]
            cond_embed_cmd = cond_proj(y_cmd)                            # [B,A,dc]
            cond_time_cmd  = cond_embed_cmd.unsqueeze(2).expand(-1, -1, Tp, -1)  # [B,A,Tp,dc]

            # init noise
            x = torch.randn((B, A, Tp, F), device=device) * T_inference[0]

            # Loop (Heun + CFG)
            for i in range(N_inference_steps - 1):
                ti      = T_inference[i]
                ti_next = T_inference[i + 1]

                ti2_s2 = ti*ti + sigma_data_t*sigma_data_t
                c_skip = (sigma_data_t*sigma_data_t) / ti2_s2
                c_out  = ti * sigma_data_t / torch.sqrt(ti2_s2)
                c_in   = 1.0 / torch.sqrt(ti2_s2)

                c_noise = (0.25 * torch.log(ti)).expand(B)

                # uncond
                full_seq_u = torch.cat([observed_past, x], dim=2)
                model_in_u = full_seq_u.clone(); model_in_u[:, :, obs_len:, :] *= c_in
                model_in_u = _normalize_like_train(model_in_u, scene_means, scene_stds)

                x_embed_u = embed_features(model_in_u, c_noise, eval=True)
                zeros_time = torch.zeros_like(cond_time_cmd)
                To = obs_len
                Dc = COND_DIM
                zeros_past = torch.zeros(B, A, To, Dc, device=device, dtype=x_embed_u.dtype)
                past_cat_u = torch.cat([x_embed_u[:, :, :obs_len, :], zeros_past], dim=-1)
                future_cat_u = torch.cat([x_embed_u[:, :, obs_len:, :], zeros_time], dim=-1)
                x_embed_u = torch.cat([past_cat_u, future_cat_u], dim=2)
                out_u = model(x_embed_u, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                # cond
                full_seq_c = torch.cat([observed_past, x], dim=2)
                model_in_c = full_seq_c.clone(); model_in_c[:, :, obs_len:, :] *= c_in
                model_in_c = _normalize_like_train(model_in_c, scene_means, scene_stds)
                x_embed_c = embed_features(model_in_c, c_noise, eval=True)
                past_cat_c = torch.cat([x_embed_c[:, :, :obs_len, :], zeros_past], dim=-1)
                future_cat_c = torch.cat([x_embed_c[:, :, obs_len:, :], cond_time_cmd], dim=-1)
                x_embed_c = torch.cat([past_cat_c, future_cat_c], dim=2)
                out_c = model(x_embed_c, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                model_out = out_u + COND_SCALE * (out_c - out_u)

                D_theta = c_skip * x + c_out * model_out
                di = (x - D_theta) / ti
                x_tilde = x + (ti_next - ti) * di

                if i < N_inference_steps - 2:
                    ti2_next_s2 = ti_next*ti_next + sigma_data_t*sigma_data_t
                    c_in_next   = 1.0 / torch.sqrt(ti2_next_s2)
                    c_noise_next = (0.25 * torch.log(ti_next)).expand(B)

                    # mid uncond
                    full_seq_u_t = torch.cat([observed_past, x_tilde], dim=2)
                    model_in_u_t = full_seq_u_t.clone(); model_in_u_t[:, :, obs_len:, :] *= c_in_next
                    model_in_u_t = _normalize_like_train(model_in_u_t, scene_means, scene_stds)
                    x_embed_u_t = embed_features(model_in_u_t, c_noise_next, eval=True)
                    past_cat_u_t = torch.cat([x_embed_u_t[:, :, :obs_len, :], zeros_past], dim=-1)
                    future_cat_u_t = torch.cat([x_embed_u_t[:, :, obs_len:, :], zeros_time], dim=-1)
                    x_embed_u_t = torch.cat([past_cat_u_t, future_cat_u_t], dim=2)
                    out_u_t = model(x_embed_u_t, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                    # mid cond
                    full_seq_c_t = torch.cat([observed_past, x_tilde], dim=2)
                    model_in_c_t = full_seq_c_t.clone(); model_in_c_t[:, :, obs_len:, :] *= c_in_next
                    model_in_c_t = _normalize_like_train(model_in_c_t, scene_means, scene_stds)
                    x_embed_c_t = embed_features(model_in_c_t, c_noise_next, eval=True)
                    past_cat_c_t = torch.cat([x_embed_c_t[:, :, :obs_len, :], zeros_past], dim=-1)
                    future_cat_c_t = torch.cat([x_embed_c_t[:, :, obs_len:, :], cond_time_cmd], dim=-1)
                    x_embed_c_t = torch.cat([past_cat_c_t, future_cat_c_t], dim=2)
                    out_c_t = model(x_embed_c_t, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                    model_out_t = out_u_t + COND_SCALE * (out_c_t - out_u_t)

                    D_theta_t = (sigma_data_t*sigma_data_t / ti2_next_s2) * x_tilde + \
                                (ti_next * sigma_data_t / torch.sqrt(ti2_next_s2)) * model_out_t
                    d_prime_i = (x_tilde - D_theta_t) / ti_next
                    x = x + (ti_next - ti) * 0.5 * (di + d_prime_i)
                else:
                    x = x_tilde

            # final pass
            final_sigma = T_inference[-1]
            fs2 = final_sigma * final_sigma
            s2  = SIGMA_DATA * SIGMA_DATA
            fs2_s2 = fs2 + s2

            c_in_final   = 1.0 / torch.sqrt(fs2_s2)
            c_skip_final = s2 / fs2_s2
            c_out_final  = final_sigma * SIGMA_DATA / torch.sqrt(fs2_s2)

            c_noise_final = (0.25 * torch.log(final_sigma)).expand(B)
            full_seq_final = torch.cat([observed_past, x], dim=2)
            model_in_final = full_seq_final.clone()
            model_in_final[:, :, obs_len:, :] *= c_in_final
            model_in_final = _normalize_like_train(model_in_final, scene_means, scene_stds)

            # uncond final
            x_embed_u_f = embed_features(model_in_final, c_noise_final, eval=True)
            zeros_time = torch.zeros_like(cond_time_cmd)
            past_cat_u_f = torch.cat([x_embed_u_f[:, :, :obs_len, :], zeros_past], dim=-1)
            future_cat_u_f = torch.cat([x_embed_u_f[:, :, obs_len:, :], zeros_time], dim=-1)
            x_embed_u_f = torch.cat([past_cat_u_f, future_cat_u_f], dim=2)
            out_u_f = model(x_embed_u_f, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

            # cond final
            x_embed_c_f = embed_features(model_in_final, c_noise_final, eval=True)
            past_cat_c_f = torch.cat([x_embed_c_f[:, :, :obs_len, :], zeros_past], dim=-1)
            future_cat_c_f = torch.cat([x_embed_c_f[:, :, obs_len:, :], cond_time_cmd], dim=-1)
            x_embed_c_f = torch.cat([past_cat_c_f, future_cat_c_f], dim=2)
            out_c_f = model(x_embed_c_f, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

            model_out_final = out_u_f + COND_SCALE * (out_c_f - out_u_f)
            final_predicted_x0 = c_skip_final * x + c_out_final * model_out_final

            valid_mask_loss = gt_future_mask.unsqueeze(-1)
            loss = (final_predicted_x0 - gt_future_part).pow(2) * valid_mask_loss
            total_loss  += loss.sum().item()
            total_valid += valid_mask_loss.sum().item()

            # Direction success
            dir_head = DirectionHead().to(device).eval()
            logits = dir_head(final_predicted_x0[..., :2], mask=gt_future_mask)  # [B,A,3]
            acc_mat = (logits.argmax(-1) == y_cmd.argmax(-1)).float()
            agent_valid = gt_future_mask.any(dim=-1)
            acc = acc_mat[agent_valid].mean().item() if agent_valid.any() else 0.0
            total_acc += acc
            total_count += 1

            # one nice plot (first batch)
            if batch_idx == 0:
                pred_traj_unscaled = _to_np(final_predicted_x0[0] * scene_stds[0][None, None, :])
                gt_traj_unscaled   = _to_np(gt_future_part[0]      * scene_stds[0][None, None, :])
                initial_noise_unscaled = _to_np(torch.randn_like(gt_future_part[0]) * T_inference[0] * scene_stds[0][None, None, :])

                map_polylines_unscaled = _to_np(roadgraph_tensor[0] * scene_stds[0][:2])
                poly_mask_np           = _to_np(roadgraph_mask[0])
                traj_mask_np           = _to_np(gt_future_mask[0])

                ego0 = (ego_ids[0].item()
                        if isinstance(ego_ids, torch.Tensor) and ego_ids.dim() == 1
                        else (int(ego_ids[0]) if isinstance(ego_ids, (list, tuple)) else 0))

                _ = plot_trajectories(
                    map_polylines=map_polylines_unscaled,
                    polyline_masks=poly_mask_np,
                    pred_traj=pred_traj_unscaled,
                    gt_traj=gt_traj_unscaled,
                    noisy_traj=initial_noise_unscaled,
                    trajectory_mask=traj_mask_np,
                    ego_id=ego0,
                    eval=True,
                    save_path="/Users/brikelkeputa/Downloads/Master-Thesis-main/MA",
                    title_suffix=f"Validation (CMD={DIRECTION_COMMAND})"
                )

            if (batch_idx + 1) % 20 == 0:
                avg_acc = total_acc / max(1, total_count)
                print(f"[VAL] Batch {batch_idx+1} ... dir_acc={avg_acc:.3f}")

    avg_loss = total_loss / (total_valid + 1e-6)
    avg_acc  = total_acc / max(1,total_count)
    print(f"[VAL] Done. Avg Loss: {avg_loss:.6f} | dir_acc={avg_acc:.3f}")

    return avg_loss, None


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model init (DEIN Stil)
    model = Denoiser().to(dev).eval()

    # Run validation once
    _ = calculate_validation_loss_and_plot(
        model=model,
        val_xml_dir=VAL_XML_DIR,
        val_batch_size=VAL_BATCH_SIZE,
        obs_len=OBS_LEN, pred_len=PRED_LEN,
        max_radius=MAX_RADIUS,
        num_polylines=NUM_POLYLINES, num_points=NUM_POINTS,
        max_agents=MAX_AGENTS,
        sigma_data=SIGMA_DATA,
        direction_command=DIRECTION_COMMAND,
        cond_scale=COND_SCALE,
        device=dev
    )
