# real_train.py
import time
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim, amp

from map_pre_old import MapDataset
from networks_2 import Denoiser
from utils import embed_features

# =========================
# FIXED SETTINGS (edit here)
# =========================
XML_DIR = "/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test"
OBS_LEN = 10
PRED_LEN = 20
NUM_TIMESTEPS = OBS_LEN + PRED_LEN
MAX_RADIUS = 100
NUM_POLYLINES = 500
NUM_POINTS = 10
MAX_AGENTS = 32

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SIGMA_DATA = 0.5

# Conditioning / CFG
TURN_THRESH_DEG = 10.0
P_UNCOND = 0.2
COND_DIM = 128
LAMBDA_DIR = 0.1

# =========================
# Conditioning helpers
# =========================
def angle_wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _unwrap_diff(theta: torch.Tensor, dim: int = -1) -> torch.Tensor:
    d = torch.diff(theta, dim=dim)
    return angle_wrap(d)

@torch.no_grad()
def direction_onehot_from_theta(theta_seq: torch.Tensor, turn_thresh_deg: float = 10.0) -> torch.Tensor:
    """
    theta_seq: [B,A,T] (rad) Zukunftsfenster
    return onehot [B,A,3] = [right, straight, left]
    """
    dtheta = _unwrap_diff(theta_seq, dim=-1)       # [B,A,T-1]
    total = dtheta.sum(dim=-1)                     # [B,A]
    th = math.radians(turn_thresh_deg)
    left = (total > th)
    right = (total < -th)
    straight = ~(left | right)
    y = torch.zeros((*total.shape, 3), device=theta_seq.device, dtype=torch.float32)
    y[..., 0] = right.float()
    y[..., 1] = straight.float()
    y[..., 2] = left.float()
    return y

class DiscreteCondProj(nn.Module):
    """ onehot(3) -> cond_embed (dc) """
    def __init__(self, in_dim: int = 3, hid: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, y_onehot: torch.Tensor) -> torch.Tensor:
        # y_onehot: [B,A,3] oder [B,3]
        if y_onehot.dim() == 2:
            y_onehot = y_onehot.unsqueeze(1)  # [B,1,3]
        return self.net(y_onehot)             # [B,A,dc]

def drop_condition_cfg(y_onehot: torch.Tensor, p_uncond: float) -> torch.Tensor:
    drop_mask = (torch.rand(y_onehot.shape[:2], device=y_onehot.device) < p_uncond)
    y_drop = y_onehot.clone()
    y_drop[drop_mask] = 0.0
    return y_drop

def build_cond_time(cond_embed: torch.Tensor, pred_len: int) -> torch.Tensor:
    return cond_embed.unsqueeze(2).expand(-1, -1, pred_len, -1)

def concat_condition_to_embed(x_embed: torch.Tensor, cond_time: torch.Tensor, obs_len: int) -> torch.Tensor:
    """
    Make feature dim consistent across all timesteps:
    - past gets Dc zeros
    - future gets the actual cond_time
    """
    B, A, T, Dx = x_embed.shape
    Tp = cond_time.size(2)                   # [B,A,Tp,Dc]
    Dc = cond_time.size(-1)
    assert obs_len + Tp == T, "obs_len + pred_len must equal T"

    # pad past with zeros in cond channel
    zeros_past = torch.zeros(B, A, obs_len, Dc, device=x_embed.device, dtype=x_embed.dtype)
    past_cat   = torch.cat([x_embed[:, :, :obs_len, :], zeros_past], dim=-1)          # [B,A,To,Dx+Dc]
    future_cat = torch.cat([x_embed[:, :, obs_len:, :], cond_time], dim=-1)           # [B,A,Tp,Dx+Dc]

    out = torch.cat([past_cat, future_cat], dim=2)                                    # [B,A,T,Dx+Dc]
    return out

class DirectionHead(nn.Module):
    """ Aux-Head: Zukunfts-(x,y) -> logits [B,A,3] """
    def __init__(self, hidden: int = 128, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, num_layers=1,
                          batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, 3))
    def forward(self, xy_future: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B,A,T,_ = xy_future.shape
        x = xy_future.view(B*A, T, 2)
        h, _ = self.gru(x)
        if mask is not None:
            m = mask.view(B*A, T, 1).float()
            h = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        else:
            h = h.mean(dim=1)
        logits = self.proj(h).view(B, A, 3)
        return logits

# =========================
# EDM helpers (match infer)
# =========================
def edm_coeffs(sigma_t: torch.Tensor, sigma_data: float, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s2 = torch.as_tensor(sigma_data * sigma_data, device=device, dtype=torch.float32)
    ti2 = sigma_t * sigma_t
    ti2_s2 = ti2 + s2
    c_in   = 1.0 / torch.sqrt(ti2_s2)
    c_skip = s2 / ti2_s2
    c_out  = sigma_t * torch.sqrt(s2) / torch.sqrt(ti2_s2)  # == ti * sigma_data / sqrt(ti^2 + s2)
    return c_in, c_skip, c_out

def sample_sigmas_edm(batch_size: int, sigma_min: float, sigma_max: float, device) -> torch.Tensor:
    rho = 7.0
    u = torch.rand(batch_size, device=device)
    return (sigma_max**(1/rho) + u * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho

def _normalize_like_train(x, scene_means, scene_stds, eps=1e-6):
    return (x - scene_means[:, None, None, :]) / (scene_stds[:, None, None, :] + eps)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Loader (DEIN Stil)
    dataset = MapDataset(
        xml_dir=XML_DIR,
        obs_len=OBS_LEN, pred_len=PRED_LEN, max_radius=MAX_RADIUS,
        num_timesteps=NUM_TIMESTEPS, num_polylines=NUM_POLYLINES, num_points=NUM_POINTS,
        save_plots=False, max_agents=MAX_AGENTS
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    # Model + conditioning modules
    model = Denoiser().to(device)
    cond_proj = DiscreteCondProj(out_dim=COND_DIM).to(device)
    dir_head  = DirectionHead().to(device)

    opt = optim.AdamW(list(model.parameters()) + list(cond_proj.parameters()) + list(dir_head.parameters()),
                      lr=LR, weight_decay=1e-4)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    sigma_min, sigma_max = 0.002, 20.0
    model.train()

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        run_recon = 0.0
        run_ce    = 0.0
        n_steps   = 0

        for batch in dataloader:
            (ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask,
             observed, observed_masks, ground_truth, ground_truth_masks,
             scene_means, scene_stds) = batch

            feature_tensor = feature_tensor.to(device, non_blocking=True)   # [B,A,T,F] (x,y,theta,...)
            feature_mask   = feature_mask.to(device, non_blocking=True)     # [B,A,T]
            roadgraph_tensor = roadgraph_tensor.to(device, non_blocking=True)
            roadgraph_mask   = roadgraph_mask.to(device, non_blocking=True)
            ground_truth     = ground_truth.to(device, non_blocking=True)
            ground_truth_masks = ground_truth_masks.to(device, non_blocking=True)
            scene_means = scene_means.to(device, non_blocking=True)
            scene_stds  = scene_stds.to(device, non_blocking=True)

            B,A,T,FeatDim = feature_tensor.shape
            To, Tp  = OBS_LEN, PRED_LEN

            # Splits
            past  = feature_tensor[:, :, :To, :]          # [B,A,To,F]
            fut   = feature_tensor[:, :, To:, :]          # [B,A,Tp,F]
            fut_m = feature_mask[:, :, To:]               # [B,A,Tp]

            # Conditioning labels aus theta (Zukunft)
            theta_future = fut[..., 2]                    # [B,A,Tp]
            y_onehot = direction_onehot_from_theta(theta_future, turn_thresh_deg=TURN_THRESH_DEG)  # [B,A,3]
            y_drop   = drop_condition_cfg(y_onehot, p_uncond=P_UNCOND)

            # Cond-Embedding
            cond_embed = cond_proj(y_drop)                # [B,A,dc]
            cond_time  = build_cond_time(cond_embed, Tp)  # [B,A,Tp,dc]

            # EDM sigma pro Item
            sigmas = sample_sigmas_edm(B, sigma_min, sigma_max, device)     # [B]
            sigma_b = sigmas.view(B,1,1,1)                                  # [B,1,1,1]

            # Noisy future
            noise = torch.randn_like(fut)
            x_noisy_future = fut + sigma_b * noise

            # Preconditioning
            c_in, c_skip, c_out = edm_coeffs(sigmas, SIGMA_DATA, device)

            full_seq = torch.cat([past, x_noisy_future], dim=2)             # [B,A,T,F]
            full_scaled = full_seq.clone()
            full_scaled[:, :, To:, :] *= c_in.view(B,1,1,1)

            # Normalisieren (selbe Stelle wie in infer)
            full_scaled = _normalize_like_train(full_scaled, scene_means, scene_stds)

            # Embed
            c_noise = (0.25 * torch.log(sigmas)).to(device)  # [B]
            x_embed = embed_features(full_scaled, c_noise, eval=False)       # [B,A,T,Dx]

            # Condition in Zukunft anhängen
            x_embed = concat_condition_to_embed(x_embed, cond_time, To)     # [B,A,T,Dx+dc]

            with amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                out = model(x_embed, roadgraph_tensor, feature_mask, roadgraph_mask)  # [B,A,T,F]
                out_future = out[:, :, To:, :]                                       # [B,A,Tp,F]

                # x0_hat rekonstruieren (wie in infer)
                x0_hat = c_skip.view(B,1,1,1) * x_noisy_future + c_out.view(B,1,1,1) * out_future

                # Recon-Loss (nur gültige Zukunft)
                valid = fut_m.unsqueeze(-1)
                recon_loss = (((x0_hat - fut) ** 2) * valid).sum() / (valid.sum() + 1e-6)

                # Aux Direction CE-Loss (auf XY der Vorhersage)
                dir_logits = dir_head(x0_hat[..., :2].detach(), mask=fut_m)  # [B,A,3]
                ce_loss = -(y_onehot * F.log_softmax(dir_logits, dim=-1)).sum(dim=-1).mean()

                total_loss = recon_loss + LAMBDA_DIR * ce_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()

            run_recon += recon_loss.item()
            run_ce    += ce_loss.item()
            n_steps   += 1

        dt = time.time() - t0
        print(f"[TRAIN] epoch {epoch:03d} | recon={run_recon/max(1,n_steps):.5f} | dirCE={run_ce/max(1,n_steps):.5f} | dt={dt:.1f}s")

    # Optional: speichern
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "cond_proj": cond_proj.state_dict(),
        "dir_head": dir_head.state_dict(),
        "epoch": EPOCHS
    }, "./checkpoints/final.pt")
    print("[DONE] saved ./checkpoints/final.pt")
