"""Experiment runner: orchestrates topology experiments and tracks results."""

import json
import os
import time

import torch

from .config import Config
from .embedding import CodeEmbedding, SimpleTokenizer
from .data.code_parser import parse_file, build_ast_graph
from .router import TopologyRouter
from .paths import CNNPath, RNNPath, TransformerPath, GNNPath
from .fusion import FusionLayer


class ExperimentRunner:
    """Orchestrates multi-path topology experiments."""

    def __init__(self, config=None):
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared components
        self.embedding = CodeEmbedding(self.config).to(self.device)
        self.tokenizer = SimpleTokenizer()
        self.router = TopologyRouter(self.config).to(self.device)
        self.fusion = FusionLayer(self.config).to(self.device)

        # Topology paths
        self.paths = nn.ModuleDict({
            "cnn": CNNPath(self.config).to(self.device),
            "rnn": RNNPath(self.config).to(self.device),
            "transformer": TransformerPath(self.config).to(self.device),
            "gnn": GNNPath(self.config).to(self.device),
        })

        self.history = []

    def review(self, file_path, active_paths=None):
        """Run all active topology paths on a single file.

        Args:
            file_path: path to source code file
            active_paths: list of path names to activate, or None for all

        Returns:
            dict with merged results and per-path breakdown
        """
        start = time.time()
        parsed = parse_file(file_path)
        ast = build_ast_graph(file_path)

        # Tokenize
        tokens_ids = self.tokenizer.encode(parsed["content"], self.config.MAX_TOKENS)
        tokens_t = torch.tensor([tokens_ids], dtype=torch.long, device=self.device)

        # Embed
        with torch.no_grad():
            emb = self.embedding(tokens_t)  # (1, seq_len, embed_dim)

        # Route
        routing_weights = self.router.heuristic_fallback(parsed)

        if active_paths is None:
            active_paths = [p for p, w in routing_weights.items() if w > 0.3]

        # Run active paths
        path_results = []
        for name in active_paths:
            if name not in self.paths:
                continue

            weight = routing_weights.get(name, 0.5)
            if weight < 0.01:
                continue

            print(f"  [{name}] routing weight={weight:.2f}", end="", flush=True)
            p_start = time.time()

            path = self.paths[name]
            try:
                if name == "transformer":
                    result = path(None, ast_graph=ast, metadata=parsed)
                elif name == "gnn":
                    result = path(emb, ast_graph=ast, metadata=parsed)
                else:
                    result = path(emb, metadata=parsed)
            except Exception as e:
                print(f"  FAILED: {e}")
                result = {"issues": [], "representation": None, "confidence": None}

            result["path_name"] = name
            result["routing_weight"] = weight
            result["latency_ms"] = (time.time() - p_start) * 1000
            path_results.append(result)
            print(f"  {len(result.get('issues', []))} issues, {result['latency_ms']:.0f}ms")

        # Fuse
        fused_rep, merged_issues = self.fusion(path_results)

        total_latency = (time.time() - start) * 1000
        result = {
            "file": file_path,
            "language": parsed["language"],
            "total_latency_ms": total_latency,
            "n_functions": len(parsed["functions"]),
            "n_lines": len(parsed["lines"]),
            "n_tokens": len(parsed["tokens"]),
            "total_issues": len(merged_issues),
            "merged_issues": merged_issues,
            "path_results": [
                {
                    "path": pr["path_name"],
                    "weight": pr["routing_weight"],
                    "n_issues": len(pr.get("issues", [])),
                    "latency_ms": pr["latency_ms"],
                }
                for pr in path_results
            ],
        }

        self.history.append(result)
        return result

    def print_summary(self, result):
        """Pretty-print experiment result."""
        print(f"\n{'='*60}")
        print(f"  FILE: {result['file']}")
        print(f"  LANGUAGE: {result['language']}")
        print(f"  LINES: {result['n_lines']}")
        print(f"  TOTAL TIME: {result['total_latency_ms']:.0f}ms")
        print(f"{'='*60}")

        for pr in result["path_results"]:
            print(f"  [{pr['path']:>12}] weight={pr['weight']:.2f}  "
                  f"issues={pr['n_issues']}  latency={pr['latency_ms']:.0f}ms")

        print(f"\n  MERGED ISSUES: {result['total_issues']}")
        for i, issue in enumerate(result["merged_issues"], 1):
            print(f"\n  {i}. [{issue['severity'].upper():>8}] {issue['category']}  "
                  f"L{issue.get('line', '?')}")
            print(f"     Source: {issue.get('source', '?')}")
            print(f"     {issue.get('description', '')[:100]}")

        print(f"\n{'='*60}")

    def save_report(self, result, output_path=None):
        """Save experiment result to JSON."""
        path = output_path or self.config.DEFAULT_OUTPUT

        def serialize(obj):
            if isinstance(obj, torch.Tensor):
                return None
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=serialize)
        print(f"\n  Report saved to {path}")

    def compare_paths(self, file_path):
        """Run each path independently and compare results.

        This is the key experiment: which topology finds which issues?
        """
        results = {}
        all_paths = list(self.paths.keys())

        for name in all_paths:
            print(f"\n  >>> Running {name.upper()} path only...")
            r = self.review(file_path, active_paths=[name])
            results[name] = {
                "n_issues": r["total_issues"],
                "issues": r["merged_issues"],
                "latency": r["total_latency_ms"],
            }

        # Also run all together
        print(f"\n  >>> Running ALL paths together...")
        r_all = self.review(file_path, active_paths=all_paths)
        results["all"] = {
            "n_issues": r_all["total_issues"],
            "issues": r_all["merged_issues"],
            "latency": r_all["total_latency_ms"],
        }

        # Print comparison
        print(f"\n{'='*60}")
        print(f"  PATH COMPARISON: {file_path}")
        print(f"{'='*60}")
        print(f"  {'Path':>14} | {'Issues':>7} | {'Latency(ms)':>11} | {'Cost':>8}")
        print(f"  {'-'*14}-+-{'-'*7}-+-{'-'*11}-+-{'-'*8}")
        for name, r in results.items():
            print(f"  {name:>14} | {r['n_issues']:>7} | {r['latency']:>11.0f} | "
                  f"{'$' if name=='all' else '$':>8}")

        # Analyze overlap
        print(f"\n  OVERLAP ANALYSIS:")
        all_issue_sigs = set()
        per_path_sigs = {}
        for name in all_paths:
            sigs = {(i.get("line", 0), i.get("description", "")[:30]) for i in results[name]["issues"]}
            per_path_sigs[name] = sigs
            all_issue_sigs.update(sigs)

        print(f"  Total unique issues across all paths: {len(all_issue_sigs)}")
        for name in all_paths:
            exclusive = len(per_path_sigs[name] - all_issue_sigs)
            shared = len(per_path_sigs[name] & all_issue_sigs)
            print(f"  [{name:>12}] exclusive={exclusive}, shared={shared}")

        return results


# Alias for convenience
nn = torch.nn
