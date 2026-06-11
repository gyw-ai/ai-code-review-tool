"""CLI entry point for topology experiments."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.topology_experiment import Config, ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Topology Experiment Platform")
    sub = parser.add_subparsers(dest="command", help="Sub-commands")

    # review command
    rp = sub.add_parser("review", help="Review a file")
    rp.add_argument("file", help="Source file to analyze")
    rp.add_argument("--mode", choices=["single", "compare"], default="single")
    rp.add_argument("--paths", nargs="+",
                    help="Active paths (cnn, rnn, transformer, gnn)")
    rp.add_argument("--output", "-o", help="Output JSON path")

    # label command: run teacher labeling
    lp = sub.add_parser("label", help="Generate teacher labels for a codebase")
    lp.add_argument("root", help="Root directory to scan")
    lp.add_argument("--max-files", type=int, default=200)
    lp.add_argument("--output", default="teacher_labels.json")

    # train command: distill teacher labels into paths
    tp = sub.add_parser("train", help="Train paths on teacher labels")
    tp.add_argument("labels", help="Teacher labels JSON file")
    tp.add_argument("--paths", nargs="+", default=["cnn", "rnn"],
                    help="Paths to train")
    tp.add_argument("--epochs", type=int, default=5)
    tp.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    config = Config()
    runner = ExperimentRunner(config)

    if args.command == "review" or args.command is None:
        if args.command is None and not hasattr(args, "file"):
            parser.print_help()
            return 0
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return 1
        if args.mode == "compare":
            runner.compare_paths(args.file)
        else:
            result = runner.review(args.file, active_paths=args.paths)
            runner.print_summary(result)
            if args.output:
                runner.save_report(result, args.output)

    elif args.command == "label":
        from src.topology_experiment.data.labeling import run_labeling_pipeline
        run_labeling_pipeline(args.root, args.output, args.max_files)

    elif args.command == "train":
        import json
        from torch.utils.data import DataLoader
        from src.topology_experiment.data.labeling import CodeReviewDataset, DistillationTrainer

        with open(args.labels, encoding="utf-8") as f:
            labeled = json.load(f)
        dataset = CodeReviewDataset(config, labeled)
        loader = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=lambda b: {
                                "tokens": torch.nn.utils.rnn.pad_sequence(
                                    [s["tokens"] for s in b], batch_first=True, padding_value=0
                                ),
                                "labels": [s["labels"] for s in b],
                                "files": [s["file"] for s in b],
                            })
        import torch
        import torch.nn.functional as F

        trainer = DistillationTrainer(config, runner.paths, runner.embedding)
        for pn in args.paths:
            if pn in runner.paths:
                print(f"\nTraining {pn}...")
                trainer.train_path(pn, loader, epochs=args.epochs, lr=args.lr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
