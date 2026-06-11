"""GPU-adaptive experiment runner for the topology platform.

Detects hardware and runs at optimal settings.
Supports: CUDA GPU, MPS (Apple Silicon), CPU.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.topology_experiment import Config, ExperimentRunner
from src.topology_experiment.data.labeling import run_labeling_pipeline


def detect_device():
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        name = "Apple Silicon"
    else:
        device = "cpu"
        name = "CPU"
    print(f"  Device: {device} ({name})")
    return device


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Topology Experiment Platform")
    parser.add_argument("--label", type=int, default=0, metavar="N",
                        help="Run teacher labeling on N files from src/")
    parser.add_argument("--train", type=str, default="", metavar="LABELS",
                        help="Train on labels file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--paths", nargs="+", default=["cnn", "rnn", "gnn"])
    parser.add_argument("--review", type=str, default="", metavar="FILE",
                        help="Review a single file")
    parser.add_argument("--compare", type=str, default="", metavar="FILE",
                        help="Compare paths on a file")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models")
    args = parser.parse_args()

    device = detect_device()
    config = Config()

    if args.list_models:
        import requests
        try:
            resp = requests.get(f"{config.OLLAMA_BASE_URL.replace('/v1','')}/api/tags")
            models = resp.json().get("models", [])
            print(f"  Available Ollama models:")
            for m in models:
                print(f"    {m['name']:30s} {m['size']//1024**3:.1f}GB")
        except Exception as e:
            print(f"  Ollama not reachable: {e}")
        return

    runner = ExperimentRunner(config)

    if args.label:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Teacher Labeling ({args.label} files)")
        print(f"{'='*60}")
        run_labeling_pipeline("src", "teacher_labels.json", args.label)

    if args.train:
        import torch
        from torch.utils.data import DataLoader
        from src.topology_experiment.data.labeling import (
            CodeReviewDataset, DistillationTrainer
        )

        print(f"\n{'='*60}")
        print(f"  PHASE 2: Training {args.paths}")
        print(f"{'='*60}")

        with open(args.train, encoding="utf-8") as f:
            labeled = json.load(f)

        dataset = CodeReviewDataset(config, labeled)
        loader = DataLoader(
            dataset, batch_size=min(8, len(dataset)), shuffle=True,
            collate_fn=lambda b: {
                "tokens": torch.nn.utils.rnn.pad_sequence(
                    [s["tokens"] for s in b], batch_first=True, padding_value=0
                ),
                "labels": [s["labels"] for s in b],
                "files": [s["file"] for s in b],
            }
        )

        trainer = DistillationTrainer(config, runner.paths, runner.embedding)
        for pn in args.paths:
            print(f"\n  Training {pn}...")
            trainer.train_path(pn, loader, epochs=args.epochs)

        print(f"\n  Training complete. Saving state dicts...")
        os.makedirs("trained_models", exist_ok=True)
        for pn in args.paths:
            torch.save(runner.paths[pn].state_dict(),
                       f"trained_models/{pn}_path.pt")
        print(f"  Saved to trained_models/")

    if args.review:
        result = runner.review(args.review)
        runner.print_summary(result)

    if args.compare:
        runner.compare_paths(args.compare)

    if not any([args.label, args.train, args.review, args.compare, args.list_models]):
        parser.print_help()


if __name__ == "__main__":
    main()
