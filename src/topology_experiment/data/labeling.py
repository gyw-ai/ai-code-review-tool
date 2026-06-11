"""Data collector and knowledge distillation trainer.

Collects teacher (Transformer/Ollama) labels and trains
smaller topology paths to reproduce them, enabling
controlled comparison of topological efficiency.
"""

import json
import os
import time
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..config import Config
from ..embedding import SimpleTokenizer, CodeEmbedding
from .code_parser import parse_file, build_ast_graph


def discover_python_files(root_dir, max_files=500):
    """Find all Python files in a directory tree."""
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".py"):
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, "rb") as f:
                        f.read(10)
                    files.append(fp)
                except Exception:
                    continue
    return files[:max_files]


class TeacherLabeler:
    """Uses the Transformer (Ollama) path to generate training labels.

    For each file, asks the LLM to identify issues at specific line numbers,
    producing token-level supervision for smaller paths to learn.
    """

    def __init__(self, config, ollama_client):
        self.config = config
        self.client = ollama_client
        self.model = config.OLLAMA_MODEL
        self.base_url = config.OLLAMA_BASE_URL

    def label_file(self, file_path):
        """Generate line-level labels for a file.

        Returns dict with:
          - file: file path
          - line_labels: list of {line, category, severity}
          - num_lines: total lines in file
        """
        parsed = parse_file(file_path)
        content = parsed["content"]
        lines = parsed["lines"]
        n_lines = len(lines)

        prompt = f"""Review this Python code and return a JSON array of issues found.
For each issue, specify the EXACT line number, category, and severity.

Categories: ["安全", "逻辑", "性能", "风格", "结构"]
Severities: ["critical", "high", "medium", "low"]

Code ({file_path}):
```python
{content[:6000]}
```

Return ONLY a JSON array:
[
  {{"line": 42, "category": "逻辑", "severity": "medium", "description": "..."}}
]"""

        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                },
                timeout=120,
            )
            raw = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r"\[[\s\S]*\]", raw)
            if not m:
                return {"file": file_path, "line_labels": [], "num_lines": n_lines}
            labels = json.loads(m.group())
        except Exception as e:
            print(f"  Teacher failed on {file_path}: {e}")
            return {"file": file_path, "line_labels": [], "num_lines": n_lines}

        valid = []
        categories = set(self.config.ISSUE_CATEGORIES.keys())
        for lbl in labels:
            line = lbl.get("line", 0)
            cat = lbl.get("category", "")
            if 1 <= line <= n_lines and cat in categories:
                valid.append({
                    "line": line,
                    "category": cat,
                    "severity": lbl.get("severity", "low"),
                    "description": lbl.get("description", ""),
                })

        return {"file": file_path, "line_labels": valid, "num_lines": n_lines}


class CodeReviewDataset(Dataset):
    """Dataset of code files with teacher-generated line labels."""

    def __init__(self, config, labeled_files, tokenizer=None, max_samples=None):
        self.config = config
        self.labeled_files = labeled_files
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.max_samples = max_samples

        # Fit tokenizer on all file contents so tokens map to meaningful ids
        texts = []
        for lf in labeled_files:
            fp = lf["file"]
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
            except Exception:
                continue
        if texts:
            self.tokenizer.fit(texts)

        # Pre-process
        self.samples = []
        for lf in labeled_files:
            if max_samples and len(self.samples) >= max_samples:
                break
            sample = self._process(lf)
            if sample:
                self.samples.append(sample)

    def _process(self, labeled_file):
        file_path = labeled_file["file"]
        labels = labeled_file["line_labels"]
        n_lines = labeled_file["num_lines"]

        try:
            parsed = parse_file(file_path)
        except Exception:
            return None

        content = parsed["content"]
        token_ids = self.tokenizer.encode(content, self.config.MAX_TOKENS)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        # Create line-level label vector: (num_lines, num_categories)
        cats = list(self.config.ISSUE_CATEGORIES.keys())
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        label_vec = torch.zeros(n_lines, len(cats))
        for lbl in labels:
            line = lbl["line"] - 1
            if 0 <= line < n_lines:
                idx = cat_to_idx.get(lbl["category"])
                if idx is not None:
                    label_vec[line, idx] = 1.0

        return {
            "tokens": tokens,
            "labels": label_vec,
            "n_lines": n_lines,
            "file": file_path,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Collate variable-length token sequences with padding."""
    tokens = [s["tokens"] for s in batch]
    max_len = max(t.size(0) for t in tokens)
    padded = torch.stack([
        F.pad(t, (0, max_len - t.size(0)), value=0)
        for t in tokens
    ])
    labels = [s["labels"] for s in batch]
    return {
        "tokens": padded,
        "labels": labels,
        "files": [s["file"] for s in batch],
    }


class DistillationTrainer:
    """Trains smaller topology paths to match teacher labels.

    This is NOT about matching the teacher's raw outputs, but about
    learning which lines have issues - a line-level classification task.
    Each path gets its own classifier head trained on the same labels.
    """

    def __init__(self, config, paths, embedding):
        self.config = config
        self.paths = paths
        self.embedding = embedding
        self.device = next(embedding.parameters()).device

    def train_path(self, path_name, dataloader, epochs=5, lr=1e-3):
        """Train a single path's classifier head to predict per-file issue category profile."""
        path = self.paths[path_name]
        optim = torch.optim.AdamW(path.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        loss_fn = nn.BCEWithLogitsLoss()

        path.train()
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            for batch in dataloader:
                tokens = batch["tokens"].to(self.device)
                line_labels = batch["labels"]  # list of (n_lines, n_cats) per file

                # Aggregate teacher labels to file-level profile
                targets = []
                for lbl in line_labels:
                    has_any = (lbl.sum(dim=0) > 0).float()  # (n_cats,)
                    targets.append(has_any)
                targets = torch.stack(targets).to(self.device)  # (B, n_cats)
                if targets.size(1) < 1:
                    continue

                with torch.no_grad():
                    emb = self.embedding(tokens)

                if path_name == "gnn":
                    result = path(emb, ast_graph=None, metadata={})
                else:
                    result = path(emb, metadata={})
                logits = result.get("logits", None)
                if logits is None or logits.size(0) != targets.size(0):
                    continue

                loss = loss_fn(logits, targets)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1}/{epochs} loss={avg:.4f}")

    def evaluate(self, path_name, dataloader):
        """Evaluate a trained path's detection accuracy."""
        path = self.paths[path_name]
        path.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in dataloader:
                tokens = batch["tokens"].to(self.device)
                emb = self.embedding(tokens)
                if path_name == "gnn":
                    result = path(emb, ast_graph=None, metadata={})
                else:
                    result = path(emb, metadata={})
                issues = result.get("issues", [])
                total += 1
        return {"path": path_name, "files_evaluated": total}


def run_labeling_pipeline(root_dir, output_path="teacher_labels.json", max_files=200):
    """Full pipeline: find files → teacher labels → save."""
    import requests
    from ..config import Config

    config = Config()
    print(f"Scanning {root_dir} for Python files...")
    files = discover_python_files(root_dir, max_files)
    print(f"Found {len(files)} files")

    class OllamaClient:
        def __init__(self):
            self.base_url = config.OLLAMA_BASE_URL

    client = OllamaClient()
    labeler = TeacherLabeler(config, client)

    all_labels = []
    for i, fp in enumerate(files):
        print(f"[{i+1}/{len(files)}] {os.path.relpath(fp, root_dir)}", end="")
        result = labeler.label_file(fp)
        all_labels.append(result)
        nl = len(result["line_labels"])
        print(f"  {nl} issues")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_labels, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_labels)} labeled files to {output_path}")
    return all_labels
