#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "🤖 ROBOTICS DATA RL CLUSTER SCHEDULER: ONE-COMMAND SHOWCASE"
echo "=========================================================================="

echo "[1/4] Running full automated test suite..."
pytest -v

echo -e "\n[2/4] Testing Cluster Scheduler Action Masking & Zero-Shot Scaling..."
python3 cli.py cluster ablation --timesteps 3000
python3 cli.py cluster scaling

echo -e "\n[3/4] Running Robotics Telemetry Simulation & Telemetry Ingestion..."
python3 cli.py robotics run-sim --robots 16 --duration 2.0

echo -e "\n[4/4] Executing Robotics Policy Benchmark & Chronos Failure Analysis..."
python3 cli.py robotics eval --scenarios 5

echo -e "\n=========================================================================="
echo "✅ Showcase Complete! To launch the interactive dashboard, run:"
echo "   streamlit run robotics_platform/dashboard/app.py"
echo "=========================================================================="
