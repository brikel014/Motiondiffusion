"""
Simple XML → Plot converter for your scenario files.

What it does
------------
• Reads dynamic obstacles from your XML (id, time, x, y, optional theta)
• Draws full trajectories per agent + last pose arrow
• One PNG per XML input

Usage
-----
python xml_to_plots.py \
  --input /path/to/folder_or_file.xml \
  --out   /path/to/output_dir \
  --dpi   300 \
  --arrow-scale 3.0

Notes
-----
• If orientation is missing, the heading is estimated from the last segment.
• Axes are in the same units as the XML (usually meters).
"""
from __future__ import annotations

import argparse
import os
import math
import glob
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrow

ColorMap = cm.get_cmap("tab20")


def _safe_float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except Exception:
        return None


def _parse_dynamic_obstacles(xml_path: str) -> Dict[str, np.ndarray]:
    """Parse XML and return {agent_id: array[[t, x, y, theta?], ...] sorted by t}.
       Theta may be NaN if missing.
    """
    root = ET.parse(xml_path).getroot()
    out: Dict[str, List[Tuple[int, float, float, float]]] = {}

    for dob in root.findall('.//dynamicObstacle'):
        aid = dob.get('id') or "unknown"
        traj = dob.find('trajectory')
        if traj is None:
            continue
        rows = []
        for state in traj.findall('state'):
            t_el = state.find('time/exact')
            try:
                t = int(t_el.text) if (t_el is not None and t_el.text is not None) else None
            except Exception:
                t = None
            px = _safe_float((state.find('position/point/x') or state.find('position/x') or state.find('point/x')) and (state.find('position/point/x') or state.find('position/x') or state.find('point/x')).text)
            py = _safe_float((state.find('position/point/y') or state.find('position/y') or state.find('point/y')) and (state.find('position/point/y') or state.find('position/y') or state.find('point/y')).text)
            th = _safe_float((state.find('orientation/exact') or state.find('orientation') or state.find('theta')) and (state.find('orientation/exact') or state.find('orientation') or state.find('theta')).text)
            if (t is None) or (px is None) or (py is None):
                continue
            rows.append((t, px, py, (math.nan if th is None else th)))
        if rows:
            rows.sort(key=lambda r: r[0])
            out[aid] = np.array(rows, dtype=float)

    return out


def _estimate_heading(seg: np.ndarray) -> float:
    """Estimate heading (radians) from last displacement; returns 0 if too short."""
    if len(seg) < 2:
        return 0.0
    dx = seg[-1, 1] - seg[-2, 1]
    dy = seg[-1, 2] - seg[-2, 2]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.atan2(dy, dx)


def _color_for_index(i: int):
    return ColorMap((i % 20) / 20.0)


def plot_xml(
    xml_path: str,
    out_png: str,
    dpi: int = 300,
    arrow_scale: float = 3.0,
    min_padding: float = 10.0,
):
    data = _parse_dynamic_obstacles(xml_path)
    if not data:
        print(f"[WARN] No dynamicObstacle trajectories found in {os.path.basename(xml_path)}")
        return

    # gather bounds
    xs, ys = [], []
    for arr in data.values():
        xs.extend(arr[:, 1].tolist())
        ys.extend(arr[:, 2].tolist())
    if not xs:
        print(f"[WARN] No XY coordinates in {os.path.basename(xml_path)}")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # trajectories per agent
    for i, (aid, arr) in enumerate(sorted(data.items(), key=lambda kv: kv[0])):
        col = _color_for_index(i)
        ax.plot(arr[:, 1], arr[:, 2], '-', lw=1.2, alpha=0.9, color=col, label=f"id={aid}")
        ax.plot(arr[0:1, 1], arr[0:1, 2], 'o', ms=3, color=col, alpha=0.9)  # start dot

        # last pose arrow
        th = arr[-1, 3]
        if math.isnan(th):
            th = _estimate_heading(arr)
        x0, y0 = arr[-1, 1], arr[-1, 2]
        ax.add_patch(FancyArrow(x0, y0, math.cos(th) * arrow_scale, math.sin(th) * arrow_scale,
                                width=0.4, length_includes_head=True, head_width=1.2, head_length=1.8,
                                color=col, alpha=0.9))

    # bounds & cosmetics
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = 0.1 * max(x_max - x_min, y_max - y_min, min_padding)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(os.path.basename(xml_path))
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.legend(fontsize='small', ncol=2, framealpha=0.9)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved plot → {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Convert scenario XML to trajectory plots")
    ap.add_argument('--input', required=True, help='XML file or directory containing XMLs')
    ap.add_argument('--out',   required=False, default='xml_plots', help='Output folder for PNGs')
    ap.add_argument('--dpi',   type=int, default=300, help='Figure DPI')
    ap.add_argument('--arrow-scale', type=float, default=3.0, help='Length of heading arrows (in XY units)')
    args = ap.parse_args()

    in_path = args.input
    out_dir = args.out

    xml_files: List[str] = []
    if os.path.isdir(in_path):
        xml_files = sorted(glob.glob(os.path.join(in_path, '*.xml')))
    elif os.path.isfile(in_path) and in_path.lower().endswith('.xml'):
        xml_files = [in_path]
    else:
        print('[ERR] --input must be an XML file or a directory containing XMLs')
        return

    if not xml_files:
        print('[WARN] No XML files found')
        return

    for xp in xml_files:
        base = os.path.splitext(os.path.basename(xp))[0]
        out_png = os.path.join(out_dir, f"{base}.png")
        try:
            plot_xml(xp, out_png, dpi=args.dpi, arrow_scale=args.arrow_scale)
        except Exception as e:
            print(f"[ERR] Failed on {xp}: {e}")


if __name__ == '__main__':
    main()
