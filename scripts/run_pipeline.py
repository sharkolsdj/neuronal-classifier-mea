"""Entry point: run the full neuronal classifier pipeline."""
import argparse, yaml, sys
sys.path.insert(0, ".")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp_default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {args.config}")
    print("See notebooks/ for step-by-step execution.")

if __name__ == "__main__":
    main()
