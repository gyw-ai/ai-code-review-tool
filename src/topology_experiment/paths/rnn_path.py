"""RNN path: captures sequential execution flow via BiLSTM.

Topological bias: SEQUENTIAL — traces execution paths, variable
state evolution, control flow across function boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from .base import TopologyPath


class RNNPath(TopologyPath):
    def __init__(self, config):
        super().__init__(config)
        self.path_name = "rnn"

        self.bilstm = nn.LSTM(
            input_size=config.EMBED_DIM,
            hidden_size=config.RNN_HIDDEN,
            num_layers=config.RNN_LAYERS,
            dropout=config.RNN_DROPOUT if config.RNN_LAYERS > 1 else 0,
            bidirectional=config.RNN_BIDIRECTIONAL,
            batch_first=True,
        )

        lstm_out = config.RNN_HIDDEN * (2 if config.RNN_BIDIRECTIONAL else 1)
        self.repr_proj = nn.Linear(lstm_out, config.FUSION_DIM)
        self.classifier = nn.Linear(lstm_out, len(config.ISSUE_CATEGORIES))
        self.dropout = nn.Dropout(config.RNN_DROPOUT)

        self._sequence_rules = _build_sequence_patterns()

    def forward(self, tokens, ast_graph=None, metadata=None):
        # tokens: (1, seq_len, embed_dim)
        h, (hn, cn) = self.bilstm(tokens)
        h = torch.mean(h, dim=1)  # mean pool over sequence: (1, lstm_out)
        h = self.dropout(h)

        logits = self.classifier(h)
        probs = torch.sigmoid(logits)

        repr_vec = self.repr_proj(h)

        issues = self._execution_flow_analysis(tokens, metadata)

        return {
            "issues": issues,
            "representation": repr_vec,
            "confidence": probs.mean(dim=1),
            "logits": logits,
        }

    def _execution_flow_analysis(self, tokens, metadata):
        """Analyze execution flow issues."""
        issues = []
        if not metadata:
            return issues

        content = metadata.get("content", "")
        path = metadata.get("path", "")
        lines = content.split("\n")

        # Track stateful execution risks
        for pattern, severity, category, desc, check_fn_name in self._sequence_rules:
            for m in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[: m.start()].count("\n") + 1
                desc_text = desc.format(match=m.group().strip()[:40])
                issues.append({
                    "file": path,
                    "line": line_num,
                    "severity": severity,
                    "category": category,
                    "description": desc_text,
                    "source": "RNN_path",
                })

        return issues


def _build_sequence_patterns():
    """Patterns related to execution flow and state."""
    return [
        (r'(recursive|recursion)', "medium", "逻辑", "可能的递归调用: {match}", None),
        (r'while\s+True', "medium", "逻辑", "无限循环可能: {match}", None),
        (r'(mutex|lock)\.acquire.*(?!\.release)', "medium", "逻辑", "可能有未释放的锁", None),
        (r'global\s+', "low", "风格", "使用全局变量: {match}", None),
        (r'(sleep|time\.sleep)\s*\(', "low", "性能", "潜在阻塞调用: {match}", None),
        (r'for\s+\w+\s+in\s+range\(len\(', "low", "性能", "低效循环模式: {match}", None),
    ]
