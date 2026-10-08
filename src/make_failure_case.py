"""Tạo ảnh failure case: người đi bộ khi calibration lệch yaw.

Chạy từ thư mục gốc repo:
    python -m src.make_failure_case
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.exp_yaw_sweep import points_in_box
from starter.datasets import load_frame
from starter.projection import draw_box2d, overlay_points, perturb_extrinsic, project_velo_to_image, velo_to_cam


def render(frame: dict, yaw_deg: float) -> tuple[np.ndarray, list[dict]]:
    points = frame["points"][np.isfinite(frame["points"]).all(axis=1)]
    points_cam_true = velo_to_cam(points[:, :3], frame["calib"])
    calibration = perturb_extrinsic(frame["calib"], yaw_deg=yaw_deg)
    uv, depth, mask = project_velo_to_image(points, calibration, frame["image"].shape)
    uv_all = np.full((len(points), 2), np.nan)
    uv_all[mask] = uv
    image = overlay_points(frame["image"], uv, depth)

    pedestrian_metrics = []
    for object_id, obj in enumerate(frame["labels"]):
        color = (0, 0, 255) if obj.type == "Pedestrian" else (0, 255, 0)
        image = draw_box2d(image, obj.bbox, color=color, label=obj.type)
        if obj.type != "Pedestrian":
            continue
        selected = points_in_box(points_cam_true, obj) & mask
        u, v = uv_all[selected, 0], uv_all[selected, 1]
        x1, y1, x2, y2 = obj.bbox
        hits = int(((u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)).sum())
        count = int(selected.sum())
        pedestrian_metrics.append({
            "object_id": object_id,
            "bbox": obj.bbox,
            "points": count,
            "hits": hits,
            "hit_ratio": hits / count if count else float("nan"),
            "distance_m": float(np.hypot(obj.location[0], obj.location[2])),
        })
    return image, pedestrian_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo ảnh so sánh baseline và yaw drift cho failure case")
    parser.add_argument("--data-root", default="data/kitti_mini", help="thư mục dữ liệu KITTI")
    parser.add_argument("--frame", default="000011", help="frame có người đi bộ")
    parser.add_argument("--yaw-deg", type=float, default=2.0, help="mức lệch yaw cần minh hoạ")
    parser.add_argument("--out", default="results/figures/fail_01_yaw_2deg_pedestrian.png", help="ảnh đầu ra")
    args = parser.parse_args()

    frame = load_frame(args.data_root, args.frame)
    baseline, metrics_ok = render(frame, 0.0)
    failure, metrics_fail = render(frame, args.yaw_deg)
    if not metrics_fail:
        raise RuntimeError(f"Frame {args.frame} không có Pedestrian để minh hoạ")

    worst = min(metrics_fail, key=lambda item: item["hit_ratio"] if np.isfinite(item["hit_ratio"]) else 2.0)
    x1, y1, x2, y2 = (int(round(value)) for value in worst["bbox"])
    for image in (baseline, failure):
        cv2.rectangle(image, (max(0, x1 - 8), max(0, y1 - 8)),
                      (min(image.shape[1] - 1, x2 + 8), min(image.shape[0] - 1, y2 + 8)), (0, 255, 255), 3)

    ok_lookup = {item["object_id"]: item for item in metrics_ok}
    baseline_ratio = ok_lookup[worst["object_id"]]["hit_ratio"]
    cv2.putText(baseline, f"Baseline yaw=0 deg | selected pedestrian hit={100 * baseline_ratio:.1f}%",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(failure, f"FAIL yaw={args.yaw_deg:g} deg | selected pedestrian hit={100 * worst['hit_ratio']:.1f}%",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(failure, f"distance={worst['distance_m']:.1f} m | Geometry: wrong extrinsic",
                (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    separator = np.full((baseline.shape[0], 8, 3), 255, dtype=np.uint8)
    comparison = np.hstack((baseline, separator, failure))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), comparison):
        raise OSError(f"Không ghi được ảnh {out}")
    print(f"Worst pedestrian object_id={worst['object_id']}, distance={worst['distance_m']:.1f} m, "
          f"hit={100 * baseline_ratio:.1f}% -> {100 * worst['hit_ratio']:.1f}%")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
