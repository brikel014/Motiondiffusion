# direction_conditioning.py
# ------------------------------------------------------------
# Richtungs-Conditioning (rechts/gerade/links) mit Aux-Classifier-Head,
# CFG-Training (Condition-Dropout), Metriken und Integrations-Helfern.
# ------------------------------------------------------------

from __future__ import annotations
import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Winkel / Labels / Helpers
# =========================

def angle_wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrappe Winkel auf [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _unwrap_diff(theta: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Diskrete Differenz entlang 'dim' und Wrap auf [-pi, pi]."""
    d = torch.diff(theta, dim=dim)
    return angle_wrap(d)


@torch.no_grad()
def direction_onehot_from_theta(
    theta_seq: torch.Tensor,
    turn_thresh_deg: float = 10.0
) -> torch.Tensor:
    """
    Erzeuge diskrete Richtungslabels aus einer Winkel-Sequenz.
    theta_seq: [B,A,T] (rad) – typischerweise Zukunftsfenster
    return: onehot [B,A,3] in Ordnung [right, straight, left]
    """
    assert theta_seq.dim() == 3, "theta_seq must be [B,A,T]"
    dtheta = _unwrap_diff(theta_seq, dim=-1)     # [B,A,T-1]
    total = dtheta.sum(dim=-1)                   # [B,A]
    th = math.radians(turn_thresh_deg)
    left = (total > th)
    right = (total < -th)
    straight = ~(left | right)

    y = torch.zeros((*total.shape, 3), device=theta_seq.device, dtype=torch.float32)
    y[..., 0] = right.float()
    y[..., 1] = straight.float()
    y[..., 2] = left.float()
    return y


@torch.no_grad()
def direction_onehot_from_traj_xy(
    xy_seq: torch.Tensor,
    turn_thresh_deg: float = 10.0
) -> torch.Tensor:
    """
    Erzeuge diskrete Richtungslabels aus (x,y)-Trajektorie (Geschwindigkeitsrichtung).
    xy_seq: [B,A,T,2]
    return: onehot [B,A,3]
    """
    assert xy_seq.dim() == 4 and xy_seq.size(-1) == 2, "xy_seq must be [B,A,T,2]"
    vx = xy_seq[..., 1:, 0] - xy_seq[..., :-1, 0]
    vy = xy_seq[..., 1:, 1] - xy_seq[..., :-1, 1]
    theta = torch.atan2(vy, vx)  # [B,A,T-1]
    # Aus theta Sequenz wieder Labels bauen: wir verwenden die Winkel über das Zukunftsfenster
    # Dazu "füllen" wir eine Pseudo-Sequenz, damit shape [B,A,T] erfüllt ist:
    # (letzter Schritt doppeln – only for label extraction)
    theta_pad = torch.cat([theta, theta[..., -1:].clone()], dim=-1)  # [B,A,T]
    return direction_onehot_from_theta(theta_pad, turn_thresh_deg=turn_thresh_deg)


def onehot_from_command(
    cmd: str,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Mappe Textkommando -> onehot [3] in Ordnung [right, straight, left].
    """
    cmd = cmd.strip().lower()
    v = torch.zeros(3, dtype=torch.float32, device=device)
    if cmd in ("r", "right", "rechts"):
        v[0] = 1.0
    elif cmd in ("s", "straight", "gerade", "forward", "vor"):
        v[1] = 1.0
    elif cmd in ("l", "left", "links"):
        v[2] = 1.0
    else:
        raise ValueError(f"Unknown command '{cmd}', use one of: rechts/gerade/links")
    return v


# =========================
# Condition-Embedding / CFG
# =========================

class DiscreteCondProj(nn.Module):
    """
    Kleiner MLP-Projektor: onehot(3) -> cond_embed (dc).
    """
    def __init__(self, in_dim: int = 3, hid: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(),
            nn.Linear(hid, out_dim)
        )

    def forward(self, y_onehot: torch.Tensor) -> torch.Tensor:
        # y_onehot: [B,A,3] oder [B,3] oder [3]
        if y_onehot.dim() == 1:
            y_onehot = y_onehot.unsqueeze(0)  # [1,3]
        if y_onehot.dim() == 2:
            y_onehot = y_onehot.unsqueeze(1)  # [B,1,3]
        return self.net(y_onehot)  # [B,A,dc]


def drop_condition_cfg(y_onehot: torch.Tensor, p_uncond: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dropout der Condition für CFG-Training.
    y_onehot: [B,A,3]
    returns: (y_dropped, drop_mask [B,A] Bool)
    """
    assert y_onehot.dim() == 3 and y_onehot.size(-1) == 3, "y_onehot must be [B,A,3]"
    drop_mask = (torch.rand(y_onehot.shape[:2], device=y_onehot.device) < p_uncond)
    y_drop = y_onehot.clone()
    y_drop[drop_mask] = 0.0
    return y_drop, drop_mask


def build_cond_time(cond_embed: torch.Tensor, pred_len: int) -> torch.Tensor:
    """
    Broadcast cond_embed [B,A,dc] über die Zukunftszeitachse -> [B,A,Tp,dc].
    """
    assert cond_embed.dim() == 3, "cond_embed must be [B,A,dc]"
    return cond_embed.unsqueeze(2).expand(-1, -1, pred_len, -1)


def concat_condition_to_embed(
    x_embed: torch.Tensor,
    cond_time: torch.Tensor,
    obs_len: int
) -> torch.Tensor:
    """
    Concat condition-FEATURES im Zukunftsfenster an deine eingebetteten Features.
    x_embed: [B,A,T,Dx], cond_time: [B,A,Tp,Dc]
    returns: [B,A,T,Dx+Dc]
    """
    B, A, T, Dx = x_embed.shape
    Tp = cond_time.size(2)
    assert obs_len + Tp == T, "obs_len + pred_len must equal T"
    future = torch.cat([x_embed[:, :, obs_len:, :], cond_time], dim=-1)
    out = torch.cat([x_embed[:, :, :obs_len, :], future], dim=-1)
    return out


# =========================
# Aux-Classifier-Head (sauber)
# =========================

class DirectionHead(nn.Module):
    """
    Aux-Head, der aus prädizierten Zukunfts-(x,y)-Sequenzen eine Richtungsklasse prediktet.
    Gibt logits [B,A,3] zurück.
    """
    def __init__(self, in_per_step: int = 2, hidden: int = 128, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_per_step,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, 3)
        )

    def forward(self, xy_future: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        xy_future: [B,A,Tp,2]
        mask:     [B,A,Tp] (1=valid), optional (für gepooltes Mittel)
        return logits: [B,A,3]
        """
        B, A, T, _ = xy_future.shape
        x = xy_future.view(B * A, T, 2)  # pack Agenten in Batch
        h, _ = self.gru(x)               # [B*A,T,H]
        if mask is not None:
            m = mask.view(B * A, T).unsqueeze(-1).float()  # [B*A,T,1]
            h = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))  # masked mean pooling
        else:
            h = h.mean(dim=1)  # global mean
        logits = self.proj(h).view(B, A, 3)
        return logits


def direction_ce_loss_from_logits(
    logits: torch.Tensor,           # [B,A,3]
    targets_onehot: torch.Tensor,   # [B,A,3]
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    CE-Loss (stabile Variante) mit Onehot-Zielen.
    """
    # Weiche Onehot-Targets -> log_softmax
    logp = F.log_softmax(logits, dim=-1)  # [B,A,3]
    if class_weights is not None:
        # class_weights: [3] -> broadcast
        logp = logp * class_weights.view(1, 1, 3)
    loss = -(targets_onehot * logp).sum(dim=-1).mean()
    return loss


# =========================
# Metriken
# =========================

@torch.no_grad()
def direction_metrics(
    pred_logits: torch.Tensor,     # [B,A,3]
    target_onehot: torch.Tensor,   # [B,A,3]
    mask_agents: Optional[torch.Tensor] = None  # [B,A] 1=valid
) -> Dict[str, float]:
    """
    Berechnet Trefferquote und Confusion Matrix (3x3) als flache Zahlen.
    """
    pred_cls = pred_logits.argmax(dim=-1)   # [B,A]
    targ_cls = target_onehot.argmax(dim=-1) # [B,A]
    if mask_agents is not None:
        valid = mask_agents.bool()
        pred_cls = pred_cls[valid]
        targ_cls = targ_cls[valid]

    total = max(1, pred_cls.numel())
    acc = (pred_cls == targ_cls).float().sum().item() / total

    # Confusion
    cm = torch.zeros(3, 3, dtype=torch.long, device=pred_cls.device)
    for i in range(pred_cls.numel()):
        cm[targ_cls[i], pred_cls[i]] += 1

    # als einfache Kennzahlen
    cm_np = cm.cpu().numpy()
    out = {
        "dir_acc": acc,
        "cm_tt": int(cm_np[0, 0]), "cm_ts": int(cm_np[0, 1]), "cm_tl": int(cm_np[0, 2]),
        "cm_st": int(cm_np[1, 0]), "cm_ss": int(cm_np[1, 1]), "cm_sl": int(cm_np[1, 2]),
        "cm_lt": int(cm_np[2, 0]), "cm_ls": int(cm_np[2, 1]), "cm_ll": int(cm_np[2, 2]),
    }
    return out


# =========================
# INTEGRATION: TRAINING
# =========================
#
# In real_train.py, nach dem Batch-Load:
#
#   theta_future = feature_tensor[:, :, obs_len:, 2]                 # [B,A,Tp]
#   y_onehot = direction_onehot_from_theta(theta_future, 10.0)       # [B,A,3]
#
#   # CFG-Training (p_uncond z.B. 0.2):
#   y_drop, _ = drop_condition_cfg(y_onehot, p_uncond=0.2)           # [B,A,3]
#
#   # Condition-Embedding:
#   cond_proj = ... (einmalig im __init__/vorher) : DiscreteCondProj(out_dim=128)
#   cond_embed = cond_proj(y_drop)                                   # [B,A,dc]
#   cond_time  = build_cond_time(cond_embed, pred_len)               # [B,A,Tp,dc]
#
#   # Deine pipeline:
#   # x_embed = embed_features(...), shape [B,A,T,Dx]
#   # Dann Future-Teil mit cond_time concat:
#   x_embed = concat_condition_to_embed(x_embed, cond_time, obs_len) # -> [B,A,T,Dx+dc]
#
#   # Denoiser vorwärts -> predicted_x0_future: [B,A,Tp,F]
#
#   # Aux-Classifier-Head:
#   dir_head = ... (einmalig anlegen): DirectionHead()
#   logits = dir_head(predicted_x0_future[..., :2], mask=gt_future_mask)  # [B,A,3]
#   ce_loss = direction_ce_loss_from_logits(logits, y_onehot)
#
#   total_loss = recon_loss + lambda_dir * ce_loss
#
# =========================
# INTEGRATION: VALIDIERUNG / SAMPLING
# =========================
#
# In infer_2.py, vor dem Heun-Loop, baue den Befehl:
#
#   # Beispiel: überall "rechts" kommandieren:
#   cmd = onehot_from_command("rechts", device=device)               # [3]
#   y_cmd = cmd.view(1,1,3).expand(B, A, 3)                          # [B,A,3]
#   cond_embed_cmd = cond_proj(y_cmd)                                # [B,A,dc]
#   cond_time_cmd  = build_cond_time(cond_embed_cmd, pred_len)       # [B,A,Tp,dc]
#   cond_scale = 2.0
#
# Im Heun-Loop je Forward ZWEI Inputs bauen:
#
#   # 1) uncond:
#   x_embed_u = embed_features(model_in_u, c_noise, eval=True)
#   zeros_time = torch.zeros_like(cond_time_cmd)
#   x_embed_u  = concat_condition_to_embed(x_embed_u, zeros_time, obs_len)
#   out_u = model(x_embed_u, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]
#
#   # 2) cond:
#   x_embed_c = embed_features(model_in_c, c_noise, eval=True)
#   x_embed_c = concat_condition_to_embed(x_embed_c, cond_time_cmd, obs_len)
#   out_c = model(x_embed_c, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]
#
#   # CFG-Kombination:
#   model_out = out_u + cond_scale * (out_c - out_u)
#
# Nach dem letzten Schritt:
#
#   predicted_x0_future = ...  # wie bisher (deine EDM-Formeln)
#
#   # Aux-Head für Metriken (wie gut wurde Befehl befolgt?):
#   logits = dir_head(predicted_x0_future[..., :2], mask=gt_future_mask)  # [B,A,3]
#   # „Ziel“ ist hier der gegebene Befehl y_cmd:
#   val_ce = direction_ce_loss_from_logits(logits, y_cmd)
#
#   # Und Erfolgsquote:
#   mets = direction_metrics(logits, y_cmd, mask_agents=gt_future_mask.any(dim=-1))
#   print("[VAL] dir_acc={:.3f}".format(mets["dir_acc"]))
#
# =========================
# OPTIONAL: θ-Loss (sanfter Winkel)
# =========================
#
# Falls du θ im Loss berücksichtigen willst (train/val), nutze:
#
def heading_align_loss(theta_pred: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
    """
    Quadratischer Fehler auf gewrapptem Winkel.
    """
    d = angle_wrap(theta_pred - theta_ref)
    return (d ** 2).mean()


def heading_smooth_loss(theta_pred: torch.Tensor) -> torch.Tensor:
    """
    Glättung via 2. Differenz über die Zeit.
    """
    d1 = theta_pred[..., 1:] - theta_pred[..., :-1]
    d2 = d1[..., 1:] - d1[..., :-1]
    return (d2 ** 2).mean()
