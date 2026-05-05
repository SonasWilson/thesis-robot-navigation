import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def rolling_mean(series, window=100):
    return series.rolling(window=window, min_periods=1).mean()


def parse_binary_column(series, name):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.clip(lower=0.0, upper=1.0)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    )
    if mapped.notna().any():
        return mapped
    raise ValueError(f"Could not parse '{name}' column as binary data.")


def stage_train_csv(log_root: str, stage_id: int) -> Path:
    return Path(log_root) / f"stage_{stage_id}" / "train_monitor.csv"


def stage_gate_csv(log_root: str, stage_id: int) -> Path:
    return Path(log_root) / f"stage_{stage_id}" / "stage_gate_log.csv"


def load_stage_df(csv_path: Path, window: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=1)
    if df.empty:
        raise ValueError(f"No episode rows yet in {csv_path}")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "r": "episode_reward",
            "l": "episode_length_steps",
            "success": "success",
            "collision": "collision",
            "path_length": "path_length",
        }
    )
    expected = {"episode_reward", "episode_length_steps", "success", "collision", "path_length"}
    missing = expected - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df["success"] = parse_binary_column(df["success"], "success")
    df["collision"] = parse_binary_column(df["collision"], "collision")
    df["timesteps"] = df["episode_length_steps"].cumsum()
    df["success_rate"] = rolling_mean(df["success"], window=window)
    df["collision_rate"] = rolling_mean(df["collision"], window=window)
    df["avg_path_length"] = rolling_mean(df["path_length"], window=window)
    df["avg_reward"] = rolling_mean(df["episode_reward"], window=window)
    return df


def load_stage_df_from_gate(gate_csv: Path, window: int) -> pd.DataFrame:
    """
    Fallback: build a minimal stage DataFrame from the gate evaluation log.
    Used when train_monitor.csv is missing or empty (e.g. Stage 1 resumed
    from checkpoint with a fresh monitor file).
    Gate log columns: eval_index, train_steps, success_rate, collision_rate,
                      reward_mean, reward_std, episode_length_mean, decision
    """
    df = pd.read_csv(gate_csv)
    if df.empty:
        raise ValueError(f"Gate log also empty: {gate_csv}")
    df.columns = [c.strip() for c in df.columns]
    # Map gate columns to the standard names used by the plotter
    out = pd.DataFrame()
    out["timesteps"]      = df["train_steps"] if "train_steps" in df.columns else df.index * 25000
    out["success_rate"]   = df["success_rate"]
    out["collision_rate"] = df["collision_rate"]
    out["avg_reward"]     = df["reward_mean"]
    out["avg_path_length"] = df.get("episode_length_mean", pd.Series([0.0] * len(df)))
    return out


def build_global_df(stage_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    offset = 0.0
    for stage_id in sorted(stage_dfs.keys()):
        sdf = stage_dfs[stage_id].copy()
        sdf["global_timesteps"] = sdf["timesteps"] + offset
        sdf["stage_id"] = stage_id
        frames.append(sdf)
        offset = float(sdf["global_timesteps"].iloc[-1])
    return pd.concat(frames, ignore_index=True)


def plot_per_stage(stage_dfs: Dict[int, pd.DataFrame], out_dir: str, window: int):
    n = len(stage_dfs)
    fig, axes = plt.subplots(n, 4, figsize=(18, max(3.2 * n, 4.5)), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"Curriculum Training Metrics Per Stage (Rolling Window={window})",
        fontsize=15,
        fontweight="bold",
    )

    for row_idx, stage_id in enumerate(sorted(stage_dfs.keys())):
        df = stage_dfs[stage_id]
        ax0, ax1, ax2, ax3 = axes[row_idx]

        ax0.plot(df["timesteps"], df["success_rate"], color="tab:green")
        ax0.set_title(f"Stage {stage_id}: Success Rate", fontweight="bold")
        ax0.set_ylim(0, 1.0)
        ax0.grid(alpha=0.3)

        ax1.plot(df["timesteps"], df["collision_rate"], color="tab:red")
        ax1.set_title(f"Stage {stage_id}: Collision Rate", fontweight="bold")
        ax1.set_ylim(0, 1.0)
        ax1.grid(alpha=0.3)

        ax2.plot(df["timesteps"], df["avg_path_length"], color="tab:blue")
        ax2.set_title(f"Stage {stage_id}: Avg Path Length", fontweight="bold")
        ax2.grid(alpha=0.3)

        ax3.plot(df["timesteps"], df["avg_reward"], color="tab:purple")
        ax3.set_title(f"Stage {stage_id}: Avg Reward", fontweight="bold")
        ax3.grid(alpha=0.3)

        for ax in (ax0, ax1, ax2, ax3):
            ax.set_xlabel("Timesteps")

    png_path = os.path.join(out_dir, "curriculum_training_metrics_per_stage.png")
    pdf_path = os.path.join(out_dir, "curriculum_training_metrics_per_stage.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_global(global_df: pd.DataFrame, out_dir: str, window: int):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True, sharex=True)
    fig.suptitle(
        f"Curriculum Training Metrics (Global Timeline, Window={window})",
        fontsize=15,
        fontweight="bold",
    )

    x = global_df["global_timesteps"]
    axes[0, 0].plot(x, global_df["success_rate"], color="tab:green")
    axes[0, 0].set_title("Success Rate", fontweight="bold")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, global_df["collision_rate"], color="tab:red")
    axes[0, 1].set_title("Collision Rate", fontweight="bold")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, global_df["avg_path_length"], color="tab:blue")
    axes[1, 0].set_title("Average Path Length", fontweight="bold")
    axes[1, 0].set_xlabel("Global Timesteps")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, global_df["avg_reward"], color="tab:purple")
    axes[1, 1].set_title("Average Episode Reward", fontweight="bold")
    axes[1, 1].set_xlabel("Global Timesteps")
    axes[1, 1].grid(alpha=0.3)

    stage_starts = global_df.groupby("stage_id")["global_timesteps"].min().to_dict()
    for ax in axes.flatten():
        for stage_id, start_x in stage_starts.items():
            ax.axvline(start_x, linestyle="--", alpha=0.3, color="black")
            ax.text(start_x, ax.get_ylim()[1] * 0.95, f"S{stage_id}", fontsize=8, alpha=0.75)

    png_path = os.path.join(out_dir, "curriculum_training_metrics_global.png")
    pdf_path = os.path.join(out_dir, "curriculum_training_metrics_global.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def refresh_once(args):
    stage_dfs: Dict[int, pd.DataFrame] = {}
    for stage_id in range(args.stage_start, args.stage_end + 1):
        csv_path  = stage_train_csv(args.log_root, stage_id)
        gate_path = stage_gate_csv(args.log_root, stage_id)

        if csv_path.exists():
            try:
                stage_dfs[stage_id] = load_stage_df(csv_path, window=max(int(args.window), 1))
                print(f"[ok] Stage {stage_id}: loaded {len(stage_dfs[stage_id])} episodes from {csv_path}")
                continue
            except Exception as exc:
                print(f"[warn] Stage {stage_id} train_monitor failed ({exc}), trying gate log...")

        # Fallback to gate log
        if gate_path.exists():
            try:
                stage_dfs[stage_id] = load_stage_df_from_gate(gate_path, window=max(int(args.window), 1))
                print(f"[ok] Stage {stage_id}: loaded {len(stage_dfs[stage_id])} gate evals from {gate_path}")
            except Exception as exc:
                print(f"[skip] Stage {stage_id}: {exc}")
        else:
            print(f"[skip] Stage {stage_id}: no data found in {csv_path} or {gate_path}")

    if not stage_dfs:
        raise FileNotFoundError("No valid stage train_monitor.csv files found yet.")

    global_df = build_global_df(stage_dfs)
    p1, p2 = plot_per_stage(stage_dfs, args.out_dir, window=max(int(args.window), 1))
    p3, p4 = plot_global(global_df, args.out_dir, window=max(int(args.window), 1))

    print("\nSaved curriculum training plots:")
    print(f"- {p1}")
    print(f"- {p2}")
    print(f"- {p3}")
    print(f"- {p4}")
    print("\nTip: use --watch to auto-refresh while training.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot curriculum training metrics from stage train_monitor.csv files"
    )
    parser.add_argument("--log_root", type=str, default="./logs")
    parser.add_argument("--stage_start", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--stage_end", type=int, default=6, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--window", type=int, default=100, help="Rolling window in episodes")
    parser.add_argument("--out_dir", type=str, default="./logs")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh plots until interrupted (Ctrl+C)",
    )
    parser.add_argument(
        "--refresh_sec",
        type=float,
        default=30.0,
        help="Refresh interval in seconds when --watch is enabled",
    )
    args = parser.parse_args()

    if args.stage_start > args.stage_end:
        raise ValueError("--stage_start must be <= --stage_end")
    if args.refresh_sec <= 0.0:
        raise ValueError("--refresh_sec must be > 0")

    os.makedirs(args.out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    if not args.watch:
        refresh_once(args)
        return

    print(f"Watch mode enabled. Refreshing every {args.refresh_sec:.1f}s. Press Ctrl+C to stop.")
    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"\n========== Refresh #{iteration} ==========")
            try:
                refresh_once(args)
            except Exception as exc:
                print(f"[warn] refresh failed: {exc}")
            time.sleep(args.refresh_sec)
    except KeyboardInterrupt:
        print("\nStopped watch mode.")


if __name__ == "__main__":
    main()
