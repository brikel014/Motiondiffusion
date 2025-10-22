# plot_tr.py
import os
import xml.etree.ElementTree as ET
import numpy as np
from utils import plot_trajectories  # dein utils.py im PYTHONPATH halten!

XML_DIR  = "/Users/brikelkeputa/Downloads/Master-Thesis-main/carla_new->real_higher_dpi/predicted_xmls"
OBS_LEN  = 10
PRED_LEN = 20

def parse_xml_future_window(xml_path, obs_len=10, pred_len=20):
    """
    Liest ein XML und gibt (pred_traj, gt_traj, noisy_traj, mask) zurück
    mit Shapes:
      pred_traj, gt_traj, noisy_traj: [A, pred_len, 3]
      mask:                           [A, pred_len]  (bool)
    Wir nehmen für jeden Agenten die Zeiten sortiert und schneiden
    ab (min_t + obs_len) die nächsten pred_len Schritte (mit Padding falls nötig).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    agents_time_xyz = []  # Liste pro Agent: [(t, x, y, theta), ...]
    for dob in root.findall(".//dynamicObstacle"):
        traj_elem = dob.find("trajectory")
        if traj_elem is None:
            continue
        seq = []
        for state in traj_elem.findall("state"):
            t_el = state.find("time/exact")
            x_el = state.find("position/point/x")
            y_el = state.find("position/point/y")
            th  = state.find("orientation/exact")
            if t_el is None or x_el is None or y_el is None or th is None:
                continue
            try:
                t = int(t_el.text)
                x = float(x_el.text); y = float(y_el.text); theta = float(th.text)
            except (TypeError, ValueError):
                continue
            seq.append((t, x, y, theta))
        if seq:
            # Nach Zeit sortieren
            seq.sort(key=lambda r: r[0])
            agents_time_xyz.append(seq)

    if not agents_time_xyz:
        # Keine Daten -> leere Arrays zurückgeben
        return (np.zeros((0, pred_len, 3), dtype=float),
                np.zeros((0, pred_len, 3), dtype=float),
                np.zeros((0, pred_len, 3), dtype=float),
                np.zeros((0, pred_len), dtype=bool))

    # Gemeinsames min_t finden, um obs/pred-Fenster auszurichten
    min_t_global = min(seq[0][0] for seq in agents_time_xyz)
    start_pred_t = min_t_global + obs_len

    pred_list, mask_list = [], []
    for seq in agents_time_xyz:
        # Zukunft ab start_pred_t herausfiltern
        future = [(t, x, y, th) for (t, x, y, th) in seq if t >= start_pred_t]

        # In Arrays der Länge pred_len bringen (abschneiden/padden)
        arr = np.zeros((pred_len, 3), dtype=float)
        msk = np.zeros((pred_len,), dtype=bool)

        n = min(len(future), pred_len)
        if n > 0:
            fut_slice = np.array([[x, y, th] for (_, x, y, th) in future[:pred_len]], dtype=float)
            arr[:n] = fut_slice[:n]
            msk[:n] = True

        pred_list.append(arr)
        mask_list.append(msk)

    pred_traj  = np.stack(pred_list, axis=0)     # [A, pred_len, 3]
    gt_traj    = pred_traj.copy()                # falls keine separaten GTs vorhanden sind
    noisy_traj = pred_traj.copy()                # reine Visualhilfe
    mask       = np.stack(mask_list, axis=0)     # [A, pred_len]

    return pred_traj, gt_traj, noisy_traj, mask

def main():
    # Dummy-Map (keine Straßen -> leerer Plot-Hintergrund)
    map_polylines  = np.zeros((1, 2, 2), dtype=float)
    polyline_masks = np.array([False])

    for fname in os.listdir(XML_DIR):
        if not fname.endswith(".xml"):
            continue
        xml_path = os.path.join(XML_DIR, fname)
        print("Plotting:", xml_path)

        pred_traj, gt_traj, noisy_traj, mask = parse_xml_future_window(
            xml_path, obs_len=OBS_LEN, pred_len=PRED_LEN
        )

        # Nichts zu plotten?
        if pred_traj.shape[0] == 0:
            print("  -> keine gültigen Trajektorien gefunden, überspringe.")
            continue

        fig = plot_trajectories(
            map_polylines=map_polylines,
            polyline_masks=polyline_masks,
            pred_traj=pred_traj,
            gt_traj=gt_traj,
            noisy_traj=noisy_traj,
            trajectory_mask=mask,
            ego_id="0",
            eval=True
        )
        out_path = xml_path.replace(".xml", "_plot.png")
        if fig is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print("  -> gespeichert:", out_path)

if __name__ == "__main__":
    main()
