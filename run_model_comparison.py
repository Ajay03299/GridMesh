"""Train the three mentor-deck forecasting backbones under the same FL protocol."""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="three rounds for a smoke test")
    args = ap.parse_args()
    rounds = 3 if args.quick else args.rounds
    for architecture in ("mlp", "gru", "lstm"):
        cmd = [sys.executable, "run_simulation.py", "--clients", str(args.clients),
               "--rounds", str(rounds), "--architecture", architecture,
               "--methods", "reliability_fedavg", "--quiet",
               "--tag", f"model_compare_{architecture}"]
        print(f"\n=== {architecture.upper()} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
