import argparse
from pathlib import Path
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


def resolve_paths(algo, env_type):
    algo = algo.lower()
    env_type = env_type.lower()
    key = (algo, env_type)

    mapping = {
        ("sac", "static2"): (
            "./logs_static2_sac",
            "./logs_static2_sac/training_metrics_static2_sac.png",
            "SAC",
            "Static2",
        ),
        ("sac", "dynamic2"): (
            "./logs_dynamic2_sac",
            "./logs_dynamic2_sac/training_metrics_dynamic2_sac.png",
            "SAC",
            "Dynamic2",
        ),
        ("td3", "static2"): (
            "./logs_static2_td3",
            "./logs_static2_td3/training_metrics_static2_td3.png",
            "TD3",
            "Static2",
        ),
        ("td3", "dynamic2"): (
            "./logs_dynamic2_td3",
            "./logs_dynamic2_td3/training_metrics_dynamic2_td3.png",
            "TD3",
            "Dynamic2",
        ),
    }
    if key in mapping:
        return mapping[key]

    # Backward-compatible fallback for older single-environment runs.
    if algo == "td3":
        return "./logs_td3", "./logs_td3/training_metrics.png", "TD3", "Default"
    if algo == "sac":
        return "./logs_sac", "./logs_sac/training_metrics.png", "SAC", "Default"
    raise ValueError(f"Unsupported algo '{algo}'. Use 'td3' or 'sac'.")


def resolve_monitor_csv(log_dir):
    log_path = Path(log_dir)
    train_csv = log_path / "train_monitor.csv"
    if train_csv.exists():
        return str(train_csv)

    candidates = sorted(log_path.glob("*monitor*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No monitor CSV found in {log_dir}. "
            "Run training first and ensure Monitor is writing logs."
        )
    return str(candidates[0])


def main():
    parser = argparse.ArgumentParser(description="Plot rolling training metrics from monitor logs")
    parser.add_argument("--algo", type=str, default="sac", choices=["td3", "sac"])
    parser.add_argument(
        "--env_type",
        type=str,
        default="dynamic2",
        choices=["static2", "dynamic2"],
        help="Training environment variant",
    )
    parser.add_argument("--window", type=int, default=100, help="Rolling window in episodes")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Optional custom log dir; overrides --algo/--env_type resolution",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional custom output image path",
    )
    args = parser.parse_args()

    default_log_dir, default_output_path, algo_name, env_name = resolve_paths(args.algo, args.env_type)
    log_dir = args.log_dir if args.log_dir else default_log_dir
    output_path = args.output_path if args.output_path else default_output_path
    monitor_path = resolve_monitor_csv(log_dir)
    df = pd.read_csv(monitor_path, skiprows=1)
    if df.empty:
        raise ValueError("train_monitor.csv has no episode data yet.")
    df.columns = [c.strip() for c in df.columns]

    # Rename for readability
    df = df.rename(
        columns={
            "r": "episode_reward",
            "l": "episode_length_steps",
            "success": "success",
            "collision": "collision",
            "path_length": "path_length",
        }
    )

    # Cumulative timesteps for x-axis
    df["timesteps"] = df["episode_length_steps"].cumsum()

    # Rolling averages/rates
    window = max(int(args.window), 1)
    if "success" not in df.columns or "collision" not in df.columns:
        raise KeyError(
            f"Expected 'success' and 'collision' columns in {monitor_path}, got {list(df.columns)}"
        )

    df["success"] = parse_binary_column(df["success"], "success")
    df["collision"] = parse_binary_column(df["collision"], "collision")
    df["success_rate"] = rolling_mean(df["success"], window=window)
    df["collision_rate"] = rolling_mean(df["collision"], window=window)
    df["avg_path_length"] = rolling_mean(df["path_length"], window=window)
    df["avg_reward"] = rolling_mean(df["episode_reward"], window=window)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"{algo_name} {env_name} Training Metrics (Rolling Window = {window} Episodes)"
    )

    axes[0, 0].plot(df["timesteps"], df["success_rate"])
    axes[0, 0].set_title("Success Rate")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df["timesteps"], df["collision_rate"], color="tab:red")
    axes[0, 1].set_title("Collision Rate")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(df["timesteps"], df["avg_path_length"], color="tab:green")
    axes[1, 0].set_title("Average Path Length")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(df["timesteps"], df["avg_reward"], color="tab:purple")
    axes[1, 1].set_title("Average Episode Reward")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Read monitor file: {monitor_path}")
    print(f"Saved plot to {output_path}")
    print(f"Raw success mean: {df['success'].mean():.3f} | unique: {sorted(df['success'].dropna().unique().tolist())[:10]}")
    print(f"Raw collision mean: {df['collision'].mean():.3f} | unique: {sorted(df['collision'].dropna().unique().tolist())[:10]}")
    print(f"Final rolling success rate: {df['success_rate'].iloc[-1]:.3f}")
    print(f"Final rolling collision rate: {df['collision_rate'].iloc[-1]:.3f}")
    print(f"Final rolling avg path length: {df['avg_path_length'].iloc[-1]:.3f}")
    print(f"Final rolling avg reward: {df['avg_reward'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
