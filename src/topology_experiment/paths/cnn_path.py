"""CNN path: captures local code patterns via 1D convolutions.

Topological bias: LOCAL — detects code smells, formatting issues,
security antipatterns that appear in small windows (3-7 tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from .base import TopologyPath


class CNNPath(TopologyPath):
    def __init__(self, config):
        super().__init__(config)
        self.path_name = "cnn"

        kernels = config.CNN_KERNEL_SIZES
        channels = config.CNN_CHANNELS

        self.convs = nn.ModuleList()
        for k, c in zip(kernels, channels):
            conv = nn.Conv1d(
                in_channels=config.EMBED_DIM,
                out_channels=c,
                kernel_size=k,
                padding=k // 2,
            )
            self.convs.append(conv)

        conv_out_dim = sum(channels)
        self.repr_proj = nn.Linear(conv_out_dim, config.FUSION_DIM)
        self.classifier = nn.Linear(conv_out_dim, len(config.ISSUE_CATEGORIES))
        self.dropout = nn.Dropout(config.CNN_DROPOUT)

        self._pattern_rules = _build_local_patterns()

    def forward(self, tokens, ast_graph=None, metadata=None):
        # tokens: (1, seq_len) -> embedding -> (1, seq_len, embed_dim)
        # Need (1, embed_dim, seq_len) for Conv1d
        x = tokens.transpose(1, 2)  # (1, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, h.size(2))  # global max pool
            conv_outs.append(h.squeeze(2))
        h = torch.cat(conv_outs, dim=1)  # (1, sum_channels)

        h = self.dropout(h)
        logits = self.classifier(h)  # (1, num_categories)
        probs = torch.sigmoid(logits)

        repr_vec = self.repr_proj(h)  # (1, fusion_dim)

        issues = self._rule_based_detection(tokens, metadata)

        return {
            "issues": issues,
            "representation": repr_vec,
            "confidence": probs.mean(dim=1),
            "logits": logits,
        }

    def _rule_based_detection(self, tokens, metadata):
        """Complement CNN with pattern matching for known local issues."""
        issues = []
        if metadata is None:
            return issues

        content = metadata.get("content", "")
        path = metadata.get("path", "")
        lines = content.split("\n")

        for pattern, severity, category, desc_template in self._pattern_rules:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for m in matches:
                line_num = content[: m.start()].count("\n") + 1
                issues.append({
                    "file": path,
                    "line": line_num,
                    "severity": severity,
                    "category": category,
                    "description": desc_template.format(match=m.group().strip()[:40]),
                    "source": "CNN_path",
                })

        return issues


def _build_local_patterns():
    """Patterns detectable within a small local window."""
    return [
        # Security: hardcoded secrets
        (r'(password|secret|api_key|token)\s*[=:]\s*["\'][^"\']+["\']', "high", "安全", "可能的硬编码凭证: {match}"),
        # Security: exec/eval usage
        (r'\b(exec|eval|compile)\s*\(', "high", "安全", "动态执行代码: {match}"),
        # Security: SQL concatenation
        (r'f["\'][^"\']*\{(?![^}]*\?)[^}]+\}[^"\']*["\'].*execute', "high", "安全", "SQL注入风险: {match}"),
        # Style: magic numbers
        (r'[^=!<>]=[ ]?\d{3,}(?!\.\d)', "low", "风格", "Magic number: {match}"),
        # Style: long lines
        (r'^.{120,}$', "low", "风格", "行过长 (>120字符)"),
        # Style: TODO/FIXME
        (r'(TODO|FIXME|HACK|XXX)', "low", "风格", "遗留标记: {match}"),
        # Logic: compare to None with ==
        (r'==\s*None|\!=\s*None', "medium", "逻辑", "应使用 'is None' 而非 '== None': {match}"),
        # Logic: bare except
        (r'except\s*:', "medium", "逻辑", "裸 except 会捕获所有异常"),
    ]
