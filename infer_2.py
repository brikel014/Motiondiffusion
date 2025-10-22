# infer_2_short_fixed.py
import os, time, warnings, math
from typing import Optional, Tuple, Dict, Any, Union, List
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import amp

from map_pre_old import MapDataset
from networks_2 import Denoiser          # model is passed in, fallback only if None
from utils import plot_trajectories, embed_features

# -------------------- constants --------------------
EMBED_DX = 1280  # must match networks_2.FeatureMLP(input_dim)

# -------------------- small helpers --------------------

# Move batch to device. We got some errors because of that
def _move_batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    return batch

def _to_np(t: torch.Tensor):
    return t.detach().to("cpu").numpy()

# Optimized noise schedule from Karras et al. 2022
# https://arxiv.org/abs/2206.00364 (Appendix C.
def get_T_optimized(sigma_max: float, sigma_min: float, rho: float, N: int, device: torch.device):
    if N <= 1:
        return torch.as_tensor([sigma_max], device=device, dtype=torch.float32)
    t = torch.arange(N, device=device, dtype=torch.float32)
    a = (sigma_min**(1.0 / rho) - sigma_max**(1.0 / rho)) / (N - 1)
    return (sigma_max**(1.0 / rho) + t * a) ** rho
#------------------------------ Function about angles ------------------------------

def angle_wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi
# The Function below converts a sequence of angles (theta_seq) into a one-hot encoded direction command based on the total change in angle.
# The Function take B,A,T tensor and returns B,A,3 one-hot tensor
def direction_onehot_from_theta(theta_seq: torch.Tensor, turn_thresh_deg: float = 15.0) -> torch.Tensor:
    """
    theta_seq: (..., T) in radians. Returns one-hot (..., 3) with order [right, straight, left].
    """
    dtheta = angle_wrap(theta_seq[..., 1:] - theta_seq[..., :-1])
    total = dtheta.sum(dim=-1)
    th = math.radians(turn_thresh_deg)
    left  = total >  th
    right = total < -th
    straight = ~(left | right)
    y = torch.zeros((*total.shape, 3), device=theta_seq.device, dtype=torch.float32)
    y[..., 0] = right.float()
    y[..., 1] = straight.float()
    y[..., 2] = left.float()
    return y

# Convert one-hot direction command back to string labels in German.
def dir_onehot_to_str_de(y_onehot: torch.Tensor) -> Union[str, List]:
    """
    One-hot (...,3) [right, straight, left] -> "rechts"/"gerade"/"links".
    """
    mapping = {0: "rechts", 1: "gerade", 2: "links"}
    idx = y_onehot.argmax(dim=-1)  # (...,)
    if idx.ndim == 0:
        return mapping[int(idx.item())]
    def _map(obj):
        if isinstance(obj, list): return [_map(x) for x in obj]
        return mapping[int(obj)]
    return _map(idx.detach().cpu().tolist())

class CondProj_Additive(torch.nn.Module):
    """[B,A,3] -> [B,A,EMBED_DX] for additive conditioning."""
    def __init__(self, out_dim=EMBED_DX):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, out_dim)
        )
    def forward(self, y_onehot):
        if y_onehot.dim() == 2:
            y_onehot = y_onehot.unsqueeze(1)  # [B,1,3]
        return self.net(y_onehot)             # [B,A,EMBED_DX]

def _add_condition_to_embed(x_embed: torch.Tensor, cond_time: torch.Tensor, obs_len: int) -> torch.Tensor:
    """
    Add cond only on future: x_embed + [0..0, cond_future].
    x_embed: [B,A,T,D], cond_time: [B,A,Tp,D]
    """
    out = x_embed.clone()
    out[:, :, obs_len:, :] = out[:, :, obs_len:, :] + cond_time
    return out

# ---- safe partial loader for old cond_proj checkpoints (loads matching shapes only) ----
def _map_old_cond_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped = {}
    for k, v in sd.items():
        if k == 'fc1.weight': mapped['net.0.weight'] = v
        elif k == 'fc1.bias': mapped['net.0.bias'] = v
        elif k == 'fc2.weight': mapped['net.2.weight'] = v
        elif k == 'fc2.bias': mapped['net.2.bias'] = v
        else: mapped[k] = v
    return mapped

def load_cond_proj_partial(module: torch.nn.Module, sd_raw: Dict[str, torch.Tensor]) -> None:
    if not sd_raw: return
    sd_mapped = _map_old_cond_keys(sd_raw)
    current = module.state_dict()
    filtered = {k: v for k, v in sd_mapped.items() if (k in current) and (current[k].shape == v.shape)}
    msg = module.load_state_dict(filtered, strict=False)
    print(f"[COND_PROJ] Loaded keys: {sorted(filtered.keys())}")
    if msg.missing_keys or msg.unexpected_keys:
        print(f"[COND_PROJ] missing: {msg.missing_keys}, unexpected: {msg.unexpected_keys}")

# ...

# ------------------------------------------------------------
# Hauptfunktion: Validation + Plot (mit Conditioning und CFG)
# ------------------------------------------------------------
def calculate_validation_loss_and_plot(
    model,
    val_xml_dir,
    val_batch_size,
    obs_len, pred_len,
    max_radius,
    num_polylines, num_points,
    max_agents,
    sigma_data,
    device=torch.device('cpu'),
    direction_command=True,  # NEU: Bedingung (z.B. "left")
    cond_scale: float = 2.0,                 # NEU: CFG-Skalierung (z.B. 2.0)
    cond_proj_state_dict: Optional[Dict[str, Any]] = None, # NEU: Zum Laden des Cond-Projektors
    turn_thresh_deg: float = 15.0            # NEU: Für die Berechnung der Ground-Truth-Richtung
):
    """
    Führt eine bedingte Validierungsrunde aus und gibt (avg_val_loss, figure) zurück.
    """

    print(f"[VAL] Start (AUTO={direction_command}, CFG scale={cond_scale}) on: {val_xml_dir}")

    model = Denoiser().to(device).eval()
    
  
    # NEU: Conditional Projection (CondProj) für die Konditionierung initialisieren/laden
 
    # condition projector only if AUTO
    cond_proj = None
    if direction_command is True:
        cond_proj = CondProj_Additive(EMBED_DX).to(device).eval()
        if cond_proj_state_dict:
            load_cond_proj_partial(cond_proj, cond_proj_state_dict) #Warum aber hier eigentlich

    # ... [Loader-Setup ist dasselbe, ausgelassen für Kürze] ...
    try:
        val_dataset = MapDataset(
            xml_dir=val_xml_dir,
            obs_len=obs_len, pred_len=pred_len, max_radius=max_radius,
            num_timesteps=obs_len + pred_len, num_polylines=num_polylines, num_points=num_points,
            save_plots=False, max_agents=max_agents
        )
        # ... [Loader-Setup wird fortgesetzt] ...
        if len(val_dataset) == 0:
            warnings.warn(f"[VAL] Leeres Dataset unter {val_xml_dir}.")
            model.train()
            return float('nan'), None
        num_workers = 2 if torch.cuda.is_available() else 0
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0), prefetch_factor=(2 if num_workers > 0 else None), drop_last=False
        )
    except Exception as e:
        warnings.warn(f"[VAL] Fehler beim Erstellen von Dataset/Loader: {e}")
        model.train()
        return float('nan'), None
    # --- ENDE Loader-Setup

    # --- Konstanten auf Device
    sigma_data_t = torch.as_tensor(float(sigma_data), device=device)
    N_inference_steps = 50
    sigma_max, sigma_min, rho = 20.0, 0.002, 7.0

    total_loss = 0.0
    total_valid = 0.0
    fig_to_return = None
    t0 = time.time()



    with torch.no_grad(), amp.autocast('cuda', enabled=(device.type == 'cuda')):
        for batch_idx, batch in enumerate(val_loader):
            try:
                # ... (Batch-Setup ist dasselbe) ...
                batch = _move_batch_to_device(batch, device)
                (ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask,
                 observed, observed_masks, ground_truth, ground_truth_masks,
                 scene_means, scene_stds) = batch

                # ... (Shapes + Splits sind dasselbe) ...
                B, A, T, F = feature_tensor.shape
                Tp = pred_len
                T_inference = get_T_optimized(sigma_max, sigma_min, rho, N_inference_steps, device)
                
                # ... (Masken und GT-Splits sind dasselbe) ...
                obs_mask       = feature_mask[:, :, :obs_len]
                gt_future_mask = feature_mask[:, :, obs_len:]
                observed_past  = feature_tensor[:, :, :obs_len, :]
                gt_future_part = feature_tensor[:, :, obs_len:, :]
                full_mask_in   = torch.cat([obs_mask, torch.zeros_like(gt_future_mask)], dim=2)

                # build cond tensors
                current_c_cond_time = None
                uncond_time_tensor = torch.zeros(B, A, Tp, EMBED_DX, device=device)

                cond_time_tensor_input_i = uncond_time_tensor  # default to uncond if no direction_command
                if direction_command is True:
                    theta_future = feature_tensor[:, :, obs_len:, 2]                           # (B,A,Tp)
                    y_onehot = direction_onehot_from_theta(theta_future, turn_thresh_deg)      # (B,A,3)
                    c_embed  = cond_proj(y_onehot)                                            # (B,A,1280)
                    current_c_cond_time = c_embed.unsqueeze(2).expand(-1, -1, Tp, -1)         # (B,A,Tp,1280)

                # diffusion schedule
                T_inference = get_T_optimized(sigma_max, sigma_min, rho, N_inference_steps, device)
                x = torch.randn((B, A, Tp, F), device=device) * T_inference[0]

                for i in range(N_inference_steps - 1):
                    ti      = T_inference[i]
                    ti_next = T_inference[i + 1]

                    ti2_s2 = ti*ti + sigma_data_t*sigma_data_t
                    c_skip = (sigma_data_t*sigma_data_t) / ti2_s2
                    c_out  = ti * sigma_data_t / torch.sqrt(ti2_s2)
                    c_in   = 1.0 / torch.sqrt(ti2_s2)
              
                    # optionaler zeit-embedding-scalar pro Item
                    c_noise = (0.25 * torch.log(ti)).expand(B)   # [B]
                    full_seq_base = torch.cat([observed_past, x], dim=2) 
                    
                    # ----------------------------------------------------
                    # --- CFG Step 1: Additive Conditioning ---
                    # ----------------------------------------------------
                    
                    # Der Tensor für die bedingte Vorhersage (entweder GT-Condition oder Null-Condition)
                    cond_tensor_i = current_c_cond_time if current_c_cond_time is not None else uncond_time_tensor
                    
                    # 1. Bedingte Vorhersage
                    # helper: embed with preconditioning on future
                    def _embed_precond(seq, c_in_scalar, c_noise_scalar):
                        seq = seq.clone()
                        seq[:, :, obs_len:, :] *= c_in_scalar
                        return embed_features(seq, c_noise_scalar, eval=True)  # (...,1280)
                    # If direction_command is False, we do unconditional prediction only
                    if direction_command is True:
                        # conditional
                        base_emb_cond = _embed_precond(full_seq_base, c_in, c_noise)
                        emb_cond = _add_condition_to_embed(base_emb_cond, current_c_cond_time, obs_len)
                        model_out_cond = model(emb_cond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]
                        # unconditional
                        base_emb_uncond = _embed_precond(full_seq_base, c_in, c_noise)
                        emb_uncond = _add_condition_to_embed(base_emb_uncond, uncond_time_tensor, obs_len)
                        model_out_uncond = model(emb_uncond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]
                        # CFG
                        model_out = model_out_uncond + cond_scale * (model_out_cond - model_out_uncond)
                    else:
                        base_emb_uncond = _embed_precond(full_seq_base, c_in, c_noise)
                        emb_uncond = _add_condition_to_embed(base_emb_uncond, uncond_time_tensor, obs_len)
                        model_out = model(emb_uncond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                    D_theta = c_skip * x + c_out * model_out
                    di = (x - D_theta) / ti
                    x_tilde = x + (ti_next - ti) * di

                    if i < N_inference_steps - 2:
                        # 2. Second step of Heun's method
                        ti2_next_s2 = ti_next*ti_next + sigma_data_t*sigma_data_t
                        c_in_next   = 1.0 / torch.sqrt(ti2_next_s2)
                        c_noise_next = (0.25 * torch.log(ti_next)).expand(B)

                        full_seq_t = torch.cat([observed_past, x_tilde], dim=2)
                        model_in_t = full_seq_t.clone()
                        model_in_t[:, :, obs_len:, :] *= c_in_next

                        if direction_command is True:
                            base_emb_t_cond = _embed_precond(full_seq_t, c_in_next, c_noise_next)
                            emb_t_cond = _add_condition_to_embed(base_emb_t_cond, current_c_cond_time, obs_len)
                            model_out_t_cond = model(emb_t_cond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                            base_emb_t_uncond = _embed_precond(full_seq_t, c_in_next, c_noise_next)
                            emb_t_uncond = _add_condition_to_embed(base_emb_t_uncond, uncond_time_tensor, obs_len)
                            model_out_t_uncond = model(emb_t_uncond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                            model_out_t = model_out_t_uncond + cond_scale * (model_out_t_cond - model_out_t_uncond)
                        else:
                            base_emb_t_uncond = _embed_precond(full_seq_t, c_in_next, c_noise_next)
                            emb_t_uncond = _add_condition_to_embed(base_emb_t_uncond, uncond_time_tensor, obs_len)
                            model_out_t = model(emb_t_uncond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                        D_theta_t = (sigma_data_t*sigma_data_t / ti2_next_s2) * x_tilde + \
                                    (ti_next * sigma_data_t / torch.sqrt(ti2_next_s2)) * model_out_t
                        d_prime = (x_tilde - D_theta_t) / ti_next
                        x = x + (ti_next - ti) * 0.5 * (di + d_prime)
                    else:
                        x = x_tilde

                # Loss Calculation (Final Step)
                final_sigma = T_inference[-1]
                fs2 = final_sigma * final_sigma
                s2  = sigma_data_t * sigma_data_t
                fs2_s2 = fs2 + s2

                c_in_final   = 1.0 / torch.sqrt(fs2_s2)
                c_skip_final = s2 / fs2_s2
                c_out_final  = final_sigma * sigma_data_t / torch.sqrt(fs2_s2)

                # Final forward pass
                c_noise_final = (0.25 * torch.log(final_sigma)).expand(B)
                full_seq_final = torch.cat([observed_past, x], dim=2)
                model_in_final = full_seq_final.clone()
                model_in_final[:, :, obs_len:, :] *= c_in_final
                base_emb_final = embed_features(model_in_final, c_noise_final, eval=True)
                
                if direction_command is True:
                    cond_tensor_final = current_c_cond_time
                    emb_final_cond = _add_condition_to_embed(base_emb_final, cond_tensor_final, obs_len)
                    out_final_cond = model(emb_final_cond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                    emb_final_uncond = _add_condition_to_embed(base_emb_final, uncond_time_tensor, obs_len)
                    out_final_uncond = model(emb_final_uncond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                    model_out_final = out_final_uncond + cond_scale * (out_final_cond - out_final_uncond)
                else:
                    emb_final_uncond = _add_condition_to_embed(base_emb_final, uncond_time_tensor, obs_len)
                    model_out_final = model(emb_final_uncond, roadgraph_tensor, full_mask_in, roadgraph_mask)[:, :, obs_len:, :]

                final_pred_x0 = c_skip_final * x + c_out_final * model_out_final

                # Loss (nur wo Maske==1)
                valid_mask_loss = gt_future_mask.unsqueeze(-1)  # [B,A,Tp,1]
                loss = (final_pred_x0 - gt_future_part).pow(2) * valid_mask_loss
                total_loss  += loss.sum().item()
                total_valid += valid_mask_loss.sum().item()

                # --- PLOT-BLOCK (FEHLERFREI) ---
                # one figure only (batch 0)
                if (fig_to_return is None) and (B > 0):
                    pred_traj_unscaled = _to_np(final_pred_x0[0] * scene_stds[0][None, None, :])
                    gt_traj_unscaled   = _to_np(gt_future_part[0]  * scene_stds[0][None, None, :])
                    initial_noise_unscaled = _to_np(torch.randn_like(gt_future_part[0]) * T_inference[0] * scene_stds[0][None, None, :])

                    map_polylines_unscaled = _to_np(roadgraph_tensor[0] * scene_stds[0][:2])
                    poly_mask_np           = _to_np(roadgraph_mask[0])
                    traj_mask_np           = _to_np(gt_future_mask[0])

                    ego0 = (ego_ids[0].item() if isinstance(ego_ids, torch.Tensor) and ego_ids.dim() == 1
                            else (int(ego_ids[0]) if isinstance(ego_ids, (list, tuple)) else 0))

                    if direction_command is True:
                        theta_future_b0 = gt_future_part[0, :, :, 2]  # [A,Tp]
                        y_onehot_b0 = direction_onehot_from_theta(theta_future_b0, turn_thresh_deg)  # [A,3]
                        ego_idx = ego0 if (0 <= ego0 < y_onehot_b0.size(0)) else 0
                        cond_label_de = dir_onehot_to_str_de(y_onehot_b0[ego_idx])
                    else:
                        cond_label_de = "unbedingte Vorhersage"

                    fig_to_return = plot_trajectories(
                        map_polylines=map_polylines_unscaled,
                        polyline_masks=poly_mask_np,
                        pred_traj=pred_traj_unscaled,
                        gt_traj=gt_traj_unscaled,
                        noisy_traj=initial_noise_unscaled,
                        trajectory_mask=traj_mask_np,
                        ego_id=ego0,
                        eval=True,
                        title_suffix=f"Condition: {cond_label_de} and Ego ID {ego0}"
                    )
                                        
                    # Versuch zu speichern (wie in Ihrem Code) HIER SVG STATT PNG
                    try:
                        out_png = os.path.join(os.getcwd(), "MA.png")
                        fig_to_return.savefig(out_png, dpi=150, bbox_inches="tight"); print(f" Visualization saved: {out_png}")
                    except Exception as e:
                        warnings.warn(f"Failed to save plot: {e}")
                # --- ENDE PLOT-BLOCK ---

                if (batch_idx + 1) % 20 == 0:
                    print(f"[VAL] Batch {batch_idx+1} verarbeitet...")

            except Exception as e:
                warnings.warn(f"[VAL] Fehler in Batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / (total_valid + 1e-6)
    print(f"[VAL] Fertig in {time.time() - t0:.2f}s. Avg Loss: {avg_loss:.6f}")
    model.train()
    return avg_loss, fig_to_return
# ------------------------------------------------------------
# Optional: Mini-Smoke-Test (nur wenn direkt gestartet)
# ------------------------------------------------------------
if __name__ == "__main__":
    from networks_2 import DiscreteCondProj # Sicherstellen, dass der Import oben steht
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[SMOKE] Device:", dev)
    
    try:
        # Denoiser muss importiert/definiert sein.
        model = Denoiser().to(dev).eval()
        
        # Optional: cond_proj aus Checkpoint laden
        cond_state = None

        ckpt_path = "./checkpoints/final.pt" # PFAD ANPASSEN
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
                cond_state_raw = ckpt.get("cond_proj", None)
                
                if cond_state_raw:
                    # FIX: UMBENENNEN DES ALTEN STATE DICT (fc1, fc2) AUF DIE NEUE STRUKTUR (net.0, net.2)
                    cond_state = {}
                    for k, v in cond_state_raw.items():
                        if k == 'fc1.weight': cond_state['net.0.weight'] = v
                        elif k == 'fc1.bias': cond_state['net.0.bias'] = v
                        elif k == 'fc2.weight': cond_state['net.2.weight'] = v
                        elif k == 'fc2.bias': cond_state['net.2.bias'] = v
                        else: cond_state[k] = v # andere Schlüssel beibehalten (falls vorhanden)

                print("[SMOKE] cond_proj state loaded and keys mapped.")
            except Exception as e:
                print(f"[SMOKE] Could not load or map cond_proj from checkpoint: {e}")
        
        # Der Val-Aufruf verwendet die neuen, korrigierten Parameter
        loss, fig = calculate_validation_loss_and_plot(
            model=model,
            val_xml_dir="/Users/brikelkeputa/Downloads/singapore_split/cleaneddata/test", # PFAD ANPASSEN
            val_batch_size=32,
            obs_len=10, pred_len=20,
            max_radius=100,
            num_polylines=500, num_points=10,
            max_agents=32,
            sigma_data=0.5,
            device=dev,
            direction_command=True, 
            cond_scale=2.0,
            cond_proj_state_dict=cond_state, # Jetzt der korrigierte State Dict
            turn_thresh_deg=15.0,
        )
        
        print("[SMOKE] Loss:", loss)
        if fig is not None:
            # Beispiel zum Speichern
            fig.savefig("MA_test.svg", dpi=350, bbox_inches="tight")
            print("Visualization saved to MA_test.svg")

    except Exception as e:
        print(f"[SMOKE] Failed: {e}")