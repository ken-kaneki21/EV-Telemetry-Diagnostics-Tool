# EV Telemetry Tool – Mock CAN-Style Log Generator and Diagnostics Dashboard

## Overview

This project implements a lightweight EV telemetry tool that simulates and analyzes
CAN-style vehicle data for an electric two-wheeler. It focuses on:

- Generating realistic mock telemetry logs for a scooter-like EV.
- Parsing and analyzing key parameters such as speed, SOC, temperatures, and pack current.
- Detecting basic anomalies (overspeed, overheating, abnormal SOC drop).
- Visualizing the data in a simple real-time style CLI dashboard using `rich`.

The goal is to prototype a diagnostic workflow that mirrors how EV teams monitor
vehicle health and performance via telemetry streams.

## Features

- Synthetic telemetry log generator (`.csv`):
  - `time_s`, `speed_kmph`, `throttle_pct`, `soc_pct`,
    `batt_temp_c`, `motor_temp_c`, `pack_current_a`
- Basic statistics computation for a drive session.
- Rule-based anomaly detection:
  - Overspeed detection
  - Battery over-temperature warning
  - Motor over-temperature warning
  - Abnormal SOC drop per minute
- CLI dashboard (using `rich`) that:
  - Streams telemetry sample-by-sample
  - Highlights active anomalies
  - Shows a session summary with anomaly rate

## Usage

```bash
pip install rich pandas numpy

# Generate mock log
python ev_telemetry_tool.py --generate

# Print summary statistics and anomaly breakdown
python ev_telemetry_tool.py --summary

# Run the live text-based dashboard
python ev_telemetry_tool.py --dashboard
