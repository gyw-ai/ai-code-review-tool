"""Verify the topology experiment platform is ready to run."""

import importlib
import sys

modules = [
    "torch",
    "requests",
    "tree_sitter",
    "numpy",
]

print("=" * 50)
print("  Topology Experiment Platform - System Check")
print("=" * 50)

# Python version
print(f"\n  Python: {sys.version.split()[0]}")

# CUDA
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")

# Core modules
print(f"\n  Checking topology_experiment imports...")
try:
    from src.topology_experiment import Config, ExperimentRunner
    print(f"  ✓ ExperimentRunner")
    from src.topology_experiment.paths import CNNPath, RNNPath, TransformerPath, GNNPath
    print(f"  ✓ All 4 paths")
    from src.topology_experiment.data.labeling import TeacherLabeler, CodeReviewDataset, DistillationTrainer
    print(f"  ✓ Labeling pipeline")
    from src.topology_experiment.router import TopologyRouter
    print(f"  ✓ Router")
    from src.topology_experiment.fusion import FusionLayer
    print(f"  ✓ Fusion layer")
    print(f"  All imports OK")
except Exception as e:
    print(f"  ✗ Import failed: {e}")

# Ollama
print(f"\n  Checking Ollama...")
import requests
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    if r.status_code == 200:
        models = r.json().get("models", [])
        print(f"  ✓ Ollama running ({len(models)} models available)")
        for m in models:
            print(f"      {m['name']:30s} {m['size']//1024**3:.1f}GB")
    else:
        print(f"  ✗ Ollama returned status {r.status_code}")
except Exception as e:
    print(f"  ✗ Ollama not reachable: {e}")

print(f"\n{'='*50}")
print(f"  Platform {'READY' if torch.cuda.is_available() else 'READY (CPU mode)'}")
print(f"{'='*50}")
