"""Topology Router: dispatches code to appropriate topology paths.

The router analyzes code characteristics and determines which
topological lens is most appropriate. In Phase 1, it uses rule-based
heuristics. In Phase 2, it can learn routing policies from experiment data.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyRouter(nn.Module):
    """Routes code to topology paths based on structural analysis."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.path_names = config.ROUTER_PATHS

        self.encoder = nn.Sequential(
            nn.Linear(16, config.ROUTER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.ROUTER_HIDDEN, config.ROUTER_HIDDEN),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(config.ROUTER_HIDDEN, len(self.path_names))

    def forward(self, features):
        """features: (batch, 16) normalized code characteristics.
        Returns routing weights for each path.
        """
        h = self.encoder(features)
        logits = self.classifier(h)
        weights = torch.softmax(logits, dim=-1)
        return weights

    def route(self, metadata):
        """Route based on extracted code characteristics.

        Returns dict of path_name -> activation_weight.
        """
        feats = self._extract_features(metadata)
        feats_t = torch.tensor(feats, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            weights = self.forward(feats_t)

        return {name: weights[0, i].item() for i, name in enumerate(self.path_names)}

    def heuristic_fallback(self, metadata):
        """Rule-based routing when learned router is not yet trained.

        Assigns all paths equal weight initially.
        """
        content = metadata.get("content", "")
        lang = metadata.get("language", "")
        n_functions = len(metadata.get("functions", []))

        weights = {name: 0.70 for name in self.path_names}

        # Adjust based on heuristics
        if "sql" in content.lower() or "exec" in content.lower():
            weights["cnn"] = 0.95  # local pattern detection
        if "class" in content or n_functions > 5:
            weights["gnn"] = 0.90  # structural analysis
        if len(content) > 5000:
            weights["transformer"] = 0.85  # needs long context
        if any(kw in content for kw in ["async", "await", "yield", "callback"]):
            weights["rnn"] = 0.90  # execution flow
        if lang in ("html", "css", "json", "yaml"):
            weights["gnn"] = 0.30  # less structural benefit

        return weights

    @staticmethod
    def _extract_features(metadata):
        """Extract normalized code characteristics for routing."""
        content = metadata.get("content", "")
        lines = content.split("\n")
        n_functions = len(metadata.get("functions", []))

        return [
            min(len(lines) / 500, 1.0),                    # 0: file length
            min(len(content) / 50000, 1.0),                  # 1: content size
            min(n_functions / 20, 1.0),                     # 2: function count
            min(content.count(" class ") / 10, 1.0),         # 3: class density
            min(content.count("def ") / 20, 1.0),            # 4: def density
            min(content.count("import") / 30, 1.0),          # 5: import density
            min(content.count("(") / 200, 1.0),              # 6: call density
            min(content.count("=") / 200, 1.0),              # 7: assignment density
            min(content.count("if") / 50, 1.0),              # 8: branch density
            min((content.count("for") + content.count("while")) / 20, 1.0),  # 9: loop
            min(content.count("try") / 10, 1.0),             # 10: error handling
            min((content.count('"') + content.count("'")) / 500, 1.0),  # 11: string density
            min(max([len(l) for l in lines[:100]] + [0]) / 150, 1.0),   # 12: max line length
            min(content.count("#") / 100, 1.0),              # 13: comment density
            float(metadata.get("language", "") in ("py", "js", "ts")),  # 14: is scripting lang
            min((content.count("async") + content.count("await")) / 10, 1.0),  # 15: async
        ]
