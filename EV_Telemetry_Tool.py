#!/usr/bin/env python3
"""
EV Telemetry Tool (Mock CAN-style Log Generator + Parser + CLI Dashboard)

Core features:
- Generate mock EV telemetry logs (CSV)
- Parse and analyze logs
- Detect basic anomalies (overheating, abnormal SOC drop, overspeed, etc.)
- Show a simple real-time style CLI dashboard using rich

This is intentionally kept single-file for easy review and sharing.
"""

import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
except ImportError:
    raise ImportError(
        "rich is required for this script.\n"
        "Install it with: pip install rich"
    )

console = Console()


# ====== CONFIGURATION ======

DATA_DIR = "data"
DEFAULT_LOG_FILE = os.path.join(DATA_DIR, "ev_mock_telemetry.csv")

# Telemetry ranges (rough, for a scooter-like EV)
SPEED_MAX = 90          # km/h
MOTOR_TEMP_WARN = 80    # °C
MOTOR_TEMP_MAX = 100    # °C
BATT_TEMP_WARN = 55     # °C
BATT_TEMP_MAX = 65      # °C
SOC_MIN = 5             # %
SOC_DROP_ABNORMAL = 8   # % drop per minute threshold


# ====== DATA CLASS FOR ANOMALIES ======

@dataclass
class AnomalyFlags:
    over_speed: bool
    batt_over_temp: bool
    motor_over_temp: bool
    soc_abnormal_drop: bool

    @property
    def any_anomaly(self) -> bool:
        return (
            self.over_speed
            or self.batt_over_temp
            or self.motor_over_temp
            or self.soc_abnormal_drop
        )


# ====== 1. MOCK LOG GENERATION ======

def generate_mock_log(
    filename: str = DEFAULT_LOG_FILE,
    duration_s: int = 600,
    dt_s: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a mock EV telemetry log and save as CSV.

    Columns:
        time_s, speed_kmph, throttle_pct, soc_pct,
        batt_temp_c, motor_temp_c, pack_current_a
    """
    np.random.seed(seed)

    num_points = duration_s // dt_s
    t = np.arange(0, duration_s, dt_s)

    # Speed profile: start slow, then cruise, then random traffic fluctuations
    speed = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti < 60:
            speed[i] = min(SPEED_MAX * 0.4, ti * 0.4)  # ramp up
        elif ti < 300:
            base = SPEED_MAX * 0.5
            speed[i] = base + 10 * np.sin(ti / 30) + np.random.normal(0, 2)
        elif ti < 540:
            base = SPEED_MAX * 0.35
            speed[i] = base + 15 * np.sin(ti / 20) + np.random.normal(0, 4)
        else:
            # deceleration
            rem = duration_s - ti
            speed[i] = max(0, rem * 0.5 + np.random.normal(0, 1))

    speed = np.clip(speed, 0, SPEED_MAX + 10)

    # Throttle roughly proportional to target speed + noise
    throttle = np.clip(speed / SPEED_MAX * 100 + np.random.normal(0, 5, size=num_points), 0, 100)

    # SOC: start at 100%, drop over time with some random variation
    base_soc_drop_per_min = 1.0  # 1% per minute average
    soc = np.zeros_like(t, dtype=float)
    soc[0] = 100.0
    for i in range(1, num_points):
        # baseline drop
        drop = base_soc_drop_per_min * dt_s / 60.0
        # influence of speed (higher speed, faster drop)
        drop *= (0.6 + 0.4 * (speed[i] / SPEED_MAX))
        # noise spikes to simulate harsh load
        drop *= np.random.uniform(0.8, 1.3)
        soc[i] = soc[i - 1] - drop
    soc = np.clip(soc, SOC_MIN, 100)

    # Battery temperature: slowly rising, influenced by current (speed)
    batt_temp = 28 + 0.03 * t + 0.015 * speed + np.random.normal(0, 0.5, size=num_points)
    # Inject mild overheating region
    batt_temp[(t > 350) & (t < 430)] += 8

    # Motor temperature: more sensitive to speed and throttle
    motor_temp = 35 + 0.05 * t + 0.03 * speed + np.random.normal(0, 1.0, size=num_points)
    # Inject more serious overheating period
    motor_temp[(t > 400) & (t < 520)] += 15

    # Pack current in Amps (simple synthetic model)
    pack_current = 10 + 0.8 * throttle + 0.2 * speed + np.random.normal(0, 3, size=num_points)

    df = pd.DataFrame(
        {
            "time_s": t,
            "speed_kmph": speed,
            "throttle_pct": throttle,
            "soc_pct": soc,
            "batt_temp_c": batt_temp,
            "motor_temp_c": motor_temp,
            "pack_current_a": pack_current,
        }
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    return df


# ====== 2. LOADING AND BASIC ANALYSIS ======

def load_log(filename: str = DEFAULT_LOG_FILE) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Log file '{filename}' not found. "
            f"Generate it first using --generate."
        )
    df = pd.read_csv(filename)
    required_cols = {
        "time_s",
        "speed_kmph",
        "throttle_pct",
        "soc_pct",
        "batt_temp_c",
        "motor_temp_c",
        "pack_current_a",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns. Found columns: {df.columns}")
    return df


def compute_basic_stats(df: pd.DataFrame) -> dict:
    stats = {
        "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
        "avg_speed": float(df["speed_kmph"].mean()),
        "max_speed": float(df["speed_kmph"].max()),
        "min_soc": float(df["soc_pct"].min()),
        "avg_batt_temp": float(df["batt_temp_c"].mean()),
        "max_batt_temp": float(df["batt_temp_c"].max()),
        "avg_motor_temp": float(df["motor_temp_c"].mean()),
        "max_motor_temp": float(df["motor_temp_c"].max()),
    }
    return stats


# ====== 3. ANOMALY DETECTION ======

def detect_anomaly_for_index(df: pd.DataFrame, idx: int) -> AnomalyFlags:
    """
    Check anomalies at a single timestamp.
    """
    row = df.iloc[idx]

    over_speed = row["speed_kmph"] > SPEED_MAX

    batt_over_temp = row["batt_temp_c"] > BATT_TEMP_WARN

    motor_over_temp = row["motor_temp_c"] > MOTOR_TEMP_WARN

    # SOC abnormal drop: compare to previous sample
    if idx == 0:
        soc_abnormal_drop = False
    else:
        dt = row["time_s"] - df["time_s"].iloc[idx - 1]
        if dt <= 0:
            soc_abnormal_drop = False
        else:
            soc_drop = df["soc_pct"].iloc[idx - 1] - row["soc_pct"]
            drop_per_min = soc_drop * 60.0 / dt
            soc_abnormal_drop = drop_per_min > SOC_DROP_ABNORMAL

    return AnomalyFlags(
        over_speed=over_speed,
        batt_over_temp=batt_over_temp,
        motor_over_temp=motor_over_temp,
        soc_abnormal_drop=soc_abnormal_drop,
    )


def precompute_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean columns for all anomalies.
    """
    over_speed = []
    batt_over_temp = []
    motor_over_temp = []
    soc_abnormal_drop = []

    for idx in range(len(df)):
        flags = detect_anomaly_for_index(df, idx)
        over_speed.append(flags.over_speed)
        batt_over_temp.append(flags.batt_over_temp)
        motor_over_temp.append(flags.motor_over_temp)
        soc_abnormal_drop.append(flags.soc_abnormal_drop)

    df = df.copy()
    df["flag_over_speed"] = over_speed
    df["flag_batt_over_temp"] = batt_over_temp
    df["flag_motor_over_temp"] = motor_over_temp
    df["flag_soc_abnormal_drop"] = soc_abnormal_drop
    df["flag_any_anomaly"] = (
        df["flag_over_speed"]
        | df["flag_batt_over_temp"]
        | df["flag_motor_over_temp"]
        | df["flag_soc_abnormal_drop"]
    )
    return df


# ====== 4. CLI DASHBOARD USING RICH ======

def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="upper", ratio=3),
        Layout(name="lower", ratio=1),
    )
    layout["upper"].split_row(
        Layout(name="telemetry"),
        Layout(name="anomalies"),
    )
    return layout


def render_telemetry_panel(row: pd.Series) -> Panel:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Parameter")
    table.add_column("Value")

    table.add_row("Time [s]", f"{row['time_s']:.0f}")
    table.add_row("Speed [km/h]", f"{row['speed_kmph']:.1f}")
    table.add_row("Throttle [%]", f"{row['throttle_pct']:.1f}")
    table.add_row("SOC [%]", f"{row['soc_pct']:.1f}")
    table.add_row("Batt Temp [°C]", f"{row['batt_temp_c']:.1f}")
    table.add_row("Motor Temp [°C]", f"{row['motor_temp_c']:.1f}")
    table.add_row("Pack Current [A]", f"{row['pack_current_a']:.1f}")

    return Panel(table, title="EV Telemetry Snapshot", border_style="cyan")


def render_anomaly_panel(row: pd.Series) -> Panel:
    lines = []

    if row["flag_over_speed"]:
        lines.append("Overspeed condition detected.")
    if row["flag_batt_over_temp"]:
        lines.append("Battery over-temperature warning.")
    if row["flag_motor_over_temp"]:
        lines.append("Motor over-temperature warning.")
    if row["flag_soc_abnormal_drop"]:
        lines.append("Abnormal SOC drop detected.")

    if not lines:
        text = Text("No active anomalies.", style="green")
    else:
        text = Text("\n".join(lines), style="red")

    return Panel(text, title="Anomalies", border_style="red")


def render_summary_panel(stats: dict, anomaly_rate: float) -> Panel:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Drive Duration [s]", f"{stats['duration_s']:.0f}")
    table.add_row("Avg Speed [km/h]", f"{stats['avg_speed']:.1f}")
    table.add_row("Max Speed [km/h]", f"{stats['max_speed']:.1f}")
    table.add_row("Min SOC [%]", f"{stats['min_soc']:.1f}")
    table.add_row("Avg Batt Temp [°C]", f"{stats['avg_batt_temp']:.1f}")
    table.add_row("Max Batt Temp [°C]", f"{stats['max_batt_temp']:.1f}")
    table.add_row("Avg Motor Temp [°C]", f"{stats['avg_motor_temp']:.1f}")
    table.add_row("Max Motor Temp [°C]", f"{stats['max_motor_temp']:.1f}")
    table.add_row("Anomaly Rate [% frames]", f"{anomaly_rate * 100:.1f}")

    return Panel(table, title="Session Summary", border_style="magenta")


def run_dashboard(df: pd.DataFrame, refresh_time: float = 0.1) -> None:
    """
    Iterate over rows and show live dashboard, like a simple real-time monitor.
    """
    stats = compute_basic_stats(df)
    anomaly_rate = df["flag_any_anomaly"].mean()

    layout = make_layout()

    with Live(layout, refresh_per_second=int(1 / refresh_time), screen=True):
        for _, row in df.iterrows():
            layout["upper"]["telemetry"].update(render_telemetry_panel(row))
            layout["upper"]["anomalies"].update(render_anomaly_panel(row))
            layout["lower"].update(render_summary_panel(stats, anomaly_rate))
            time.sleep(refresh_time)


# ====== 5. COMMAND-LINE INTERFACE ======

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="EV Telemetry Tool: Mock log generator + parser + CLI dashboard"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new telemetry log and exit.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help=f"Path to telemetry CSV file (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run the real-time style dashboard on the given log file.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print basic summary statistics and anomaly rate.",
    )
    args = parser.parse_args()

    if args.generate:
        console.print(f"Generating mock telemetry log at: {args.file}")
        df = generate_mock_log(args.file)
        console.print(f"Generated {len(df)} samples.")
        return

    # If not generated in this run, we load an existing file
    df = load_log(args.file)
    df = precompute_anomalies(df)

    if args.summary:
        stats = compute_basic_stats(df)
        anomaly_rate = df["flag_any_anomaly"].mean()

        console.print("\nBasic Telemetry Summary:\n", style="bold")
        for k, v in stats.items():
            console.print(f"{k}: {v:.2f}")

        console.print(f"\nAnomaly rate: {anomaly_rate * 100:.2f}% of frames")

        # Count each anomaly type
        console.print("\nAnomaly breakdown:")
        console.print(f"  Overspeed frames: {df['flag_over_speed'].sum()}")
        console.print(f"  Battery over-temp frames: {df['flag_batt_over_temp'].sum()}")
        console.print(f"  Motor over-temp frames: {df['flag_motor_over_temp'].sum()}")
        console.print(
            f"  Abnormal SOC drop frames: {df['flag_soc_abnormal_drop'].sum()}"
        )
        console.print()

    if args.dashboard:
        run_dashboard(df)


if __name__ == "__main__":
    main()
