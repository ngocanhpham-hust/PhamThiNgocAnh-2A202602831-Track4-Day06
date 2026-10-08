"""Vẽ kết quả quét yaw tổng hợp và theo class.

Chạy từ thư mục gốc repo:
    python -m src.plot_yaw_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def weighted_group_ratio(group: pd.DataFrame) -> float:
    points = group["object_points"].sum()
    return float(group["hits"].sum() / points) if points else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ độ nhạy projection theo yaw")
    parser.add_argument("--input", default="results/yaw_perturb_sweep.csv", help="CSV tổng hợp")
    parser.add_argument("--detail-input", default="results/yaw_perturb_objects.csv", help="CSV theo vật thể")
    parser.add_argument("--out-dir", default="results/figures", help="thư mục chứa biểu đồ")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.input, dtype={"frame": str})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for frame_id, group in summary.groupby("frame"):
        ax.plot(group["yaw_deg"], 100 * group["hit_ratio"], marker="o", label=f"frame {frame_id}")
    ax.set_xlabel("Yaw error (degree)")
    ax.set_ylabel("Object LiDAR points inside 2D box (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    summary_path = out_dir / "yaw_sweep.png"
    fig.savefig(summary_path, dpi=160)
    plt.close(fig)

    details = pd.read_csv(args.detail_input, dtype={"frame": str})
    grouped = (details.groupby(["yaw_deg", "class"], as_index=False)
               .apply(lambda group: pd.Series({"hit_ratio": weighted_group_ratio(group)}), include_groups=False))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for class_name, group in grouped.groupby("class"):
        ax.plot(group["yaw_deg"], 100 * group["hit_ratio"], marker="o", label=class_name)
    ax.set_xlabel("Yaw error (degree)")
    ax.set_ylabel("Object LiDAR points inside 2D box (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    class_path = out_dir / "yaw_sweep_by_class.png"
    fig.savefig(class_path, dpi=160)
    plt.close(fig)

    print(f"-> {summary_path}")
    print(f"-> {class_path}")


if __name__ == "__main__":
    main()
