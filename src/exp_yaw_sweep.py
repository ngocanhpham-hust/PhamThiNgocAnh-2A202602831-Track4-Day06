"""Topic A: đo độ nhạy của projection khi extrinsic bị lệch yaw.

Ngoài CSV tổng hợp theo frame, script ghi thêm CSV theo từng vật thể để có thể
phân tích theo class và khoảng cách (phần mở rộng so với script mẫu codelab).

Ví dụ:
    python -m src.exp_yaw_sweep
    python -m src.exp_yaw_sweep --frames 000008 000011 000049 --yaw-levels 0 0.5 1 2 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from starter.datasets import load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image, velo_to_cam

CLASSES = ("Car", "Van", "Pedestrian", "Cyclist")


def points_in_box(points_cam: np.ndarray, obj) -> np.ndarray:
    """Trả về mask các điểm camera-frame nằm trong 3D box KITTI ``obj``."""
    h, w, length = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ rotation
    return ((np.abs(local[:, 0]) <= length / 2)
            & (local[:, 1] <= 0) & (local[:, 1] >= -h)
            & (np.abs(local[:, 2]) <= w / 2))


def distance_bin(distance_m: float) -> str:
    if distance_m < 15:
        return "0-15m"
    if distance_m < 30:
        return "15-30m"
    return ">=30m"


def run_one(fr: dict, yaw_deg: float) -> tuple[dict, list[dict]]:
    points = fr["points"][np.isfinite(fr["points"]).all(axis=1)]
    points_cam_true = velo_to_cam(points[:, :3], fr["calib"])
    perturbed = perturb_extrinsic(fr["calib"], yaw_deg=yaw_deg)
    uv, _, mask = project_velo_to_image(points, perturbed, fr["image"].shape)
    uv_all = np.full((len(points), 2), np.nan)
    uv_all[mask] = uv

    total_object_points = 0
    total_hits = 0
    details: list[dict] = []
    for object_id, obj in enumerate(fr["labels"]):
        if obj.type not in CLASSES:
            continue
        selected = points_in_box(points_cam_true, obj) & mask
        u, v = uv_all[selected, 0], uv_all[selected, 1]
        x1, y1, x2, y2 = obj.bbox
        hits = int(((u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)).sum())
        object_points = int(selected.sum())
        distance_m = float(np.hypot(obj.location[0], obj.location[2]))
        details.append({
            "object_id": object_id,
            "class": obj.type,
            "distance_m": round(distance_m, 2),
            "distance_bin": distance_bin(distance_m),
            "object_points": object_points,
            "hits": hits,
            "hit_ratio": round(hits / object_points, 4) if object_points else float("nan"),
        })
        total_object_points += object_points
        total_hits += hits

    summary = {
        "n_points": len(points),
        "inside_image": int(mask.sum()),
        "object_points": total_object_points,
        "hits": total_hits,
        "hit_ratio": round(total_hits / total_object_points, 4) if total_object_points else float("nan"),
    }
    return summary, details


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quét lệch yaw và đo tỉ lệ điểm vật thể nằm đúng trong 2D box")
    parser.add_argument("--data-root", default="data/kitti_mini", help="thư mục dữ liệu KITTI")
    parser.add_argument("--frames", nargs="+", default=["000008", "000011", "000049"], help="các frame cần đo")
    parser.add_argument("--yaw-levels", nargs="+", type=float, default=[0, 0.5, 1, 2, 3], help="các mức yaw (độ)")
    parser.add_argument("--out", default="results/yaw_perturb_sweep.csv", help="CSV tổng hợp theo frame")
    parser.add_argument("--detail-out", default="results/yaw_perturb_objects.csv", help="CSV chi tiết theo vật thể")
    args = parser.parse_args()

    rows: list[dict] = []
    detail_rows: list[dict] = []
    for frame_id in args.frames:
        frame = load_frame(args.data_root, frame_id)
        for yaw_deg in args.yaw_levels:
            summary, details = run_one(frame, yaw_deg)
            common = {"dataset": Path(args.data_root).name, "frame": frame_id, "yaw_deg": yaw_deg}
            row = {**common, **summary}
            rows.append(row)
            detail_rows.extend({**common, **detail} for detail in details)
            print(row)

    write_csv(Path(args.out), rows)
    write_csv(Path(args.detail_out), detail_rows)
    print(f"-> {args.out} ({len(rows)} dòng)")
    print(f"-> {args.detail_out} ({len(detail_rows)} dòng, phân tích theo class/khoảng cách)")


if __name__ == "__main__":
    main()
