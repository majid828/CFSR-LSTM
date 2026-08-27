"""Experiment 1: standard and peak-aware single-memory LSTM baselines."""
import subprocess, sys

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "training.train_lstm", "--config", "config.yaml"])
    subprocess.check_call([sys.executable, "-m", "training.train_lstm", "--config", "config.yaml", "--peak-aware"])
