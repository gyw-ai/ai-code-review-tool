"""Fusion layer: cross-attention between topology path outputs.

Transforms the separate outputs of each topology path into a
unified, integrated review by learning cross-path interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    """Cross-attention fusion of multi-topology representations."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_paths = len(config.ROUTER_PATHS)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.FUSION_DIM,
            num_heads=config.FUSION_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.path_embeddings = nn.Embedding(self.n_paths, config.FUSION_DIM)
        self.output_proj = nn.Linear(config.FUSION_DIM, config.FUSION_DIM)
        self.issue_router = nn.Linear(config.FUSION_DIM, len(config.ISSUE_CATEGORIES))

    def forward(self, path_results):
        reps = []
        valid_indices = []

        for i, r in enumerate(path_results):
            rep = r.get("representation")
            if rep is not None and isinstance(rep, torch.Tensor) and rep.numel() > 0:
                rep = rep.view(1, -1)
                reps.append(rep)
                valid_indices.append(i)

        if len(reps) < 1:
            B, D = 1, self.config.FUSION_DIM
            return torch.zeros(B, D), self._merge_issues(path_results)

        reps = torch.stack(reps, dim=1)

        if reps.dim() == 4:
            reps = reps.squeeze(2)
        if reps.dim() == 2:
            reps = reps.unsqueeze(0)

        n_valid = reps.size(1)
        if n_valid < 2:
            return reps.squeeze(1), self._merge_issues(path_results)

        path_ids = torch.tensor(valid_indices[:n_valid], device=reps.device).unsqueeze(0)
        pe = self.path_embeddings(path_ids)
        reps = reps + pe

        attn_out, _ = self.cross_attn(reps, reps, reps)

        confs = []
        for i in valid_indices[:n_valid]:
            c = path_results[i].get("confidence")
            if c is not None and isinstance(c, torch.Tensor) and c.numel() > 0:
                confs.append(c.view(1, 1, 1))
            else:
                confs.append(torch.ones(1, 1, 1))

        confs = torch.cat(confs, dim=1)
        weights = F.softmax(confs, dim=1)
        fused = (attn_out * weights).sum(dim=1)

        fused = self.output_proj(fused)
        return fused, self._merge_issues(path_results)

    def _merge_issues(self, path_results):
        all_issues = []
        seen = set()
        for r in path_results:
            for issue in r.get("issues", []):
                key = (str(issue.get("file", "")), issue.get("line", 0), str(issue.get("description", ""))[:30])
                if key not in seen:
                    seen.add(key)
                    all_issues.append(issue)
        return all_issues
