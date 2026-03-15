from __future__ import annotations

import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

_LOG_PATTERN = re.compile(
    r"\[(DATA|ALERT)\]\s+"
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\|\s+"
    r"([^|]+)\|\s+"
    r"HR=([\d.]+)\s+bpm\s+\|\s+"
    r"SpO2=([\d.]+)%\s+\|\s+"
    r"Temp=([\d.]+)C\s+\|\s+"
    r"Activity=(\w+)"
)


def _parse_log(log_path: Path) -> pd.DataFrame:
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = _LOG_PATTERN.search(line)
        if m:
            rows.append(
                {
                    "type": m.group(1),
                    "timestamp": pd.to_datetime(m.group(2), utc=True),
                    "user": m.group(3).strip(),
                    "heart_rate": float(m.group(4)),
                    "spo2": float(m.group(5)),
                    "body_temperature": float(m.group(6)),
                    "activity": m.group(7),
                }
            )
    return pd.DataFrame(rows)


def plot_report(log_path: Path, output_path: Path) -> None:
    df = _parse_log(log_path)
    if df.empty:
        print(f"No parseable entries found in {log_path}.")
        return

    users = df["user"].unique().tolist()
    n_users = len(users)

    metrics: list[tuple[str, str, str]] = [
        ("heart_rate", "Heart Rate (bpm)", "steelblue"),
        ("spo2", "SpO2 (%)", "seagreen"),
        ("body_temperature", "Temperature (C)", "darkorange"),
    ]

    fig, axes = plt.subplots(
        3, n_users, figsize=(8 * n_users, 12), sharex="col", squeeze=False
    )
    fig.suptitle(
        "Wearable Health Stream - Anomaly Detection Report",
        fontsize=14,
        fontweight="bold",
    )

    for col, user in enumerate(users):
        udf = df[df["user"] == user].sort_values("timestamp")
        normal = udf[udf["type"] == "DATA"]
        anomalies = udf[udf["type"] == "ALERT"]

        for row, (metric, ylabel, color) in enumerate(metrics):
            ax = axes[row][col]
            ax.plot(
                normal["timestamp"],
                normal[metric],
                color=color,
                linewidth=1.2,
                label="Normal",
            )
            if not anomalies.empty:
                ax.scatter(
                    anomalies["timestamp"],
                    anomalies[metric],
                    color="crimson",
                    zorder=5,
                    s=70,
                    marker="x",
                    linewidths=1.8,
                    label="Anomaly",
                )
            ax.set_ylabel(ylabel, fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
            ax.legend(loc="upper right", fontsize=8)
            if row == 0:
                ax.set_title(user, fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Report saved to {output_path}")
