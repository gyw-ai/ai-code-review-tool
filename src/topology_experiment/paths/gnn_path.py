"""GNN path: structural relationship analysis via Graph Neural Network.

Topological bias: STRUCTURAL — operates on the AST graph, capturing
inheritance hierarchies, call graphs, dependency structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os

from .base import TopologyPath


class GNNPath(TopologyPath):
    def __init__(self, config):
        super().__init__(config)
        self.path_name = "gnn"

        self.node_embed = nn.Linear(32, config.GNN_HIDDEN)
        self.convs = nn.ModuleList()
        for _ in range(config.GNN_LAYERS):
            self.convs.append(GCNLayer(config.GNN_HIDDEN, config.GNN_HIDDEN))

        self.repr_proj = nn.Linear(config.GNN_HIDDEN, config.FUSION_DIM)
        self.classifier = nn.Linear(config.GNN_HIDDEN, len(config.ISSUE_CATEGORIES))
        self.dropout = nn.Dropout(config.GNN_DROPOUT)
        self.structural_rules = _build_structural_patterns()

    def forward(self, tokens, ast_graph=None, metadata=None):
        if ast_graph is None or len(ast_graph.get("nodes", [])) < 2:
            return self._empty_result()

        x, edge_index = self._build_graph_tensor(ast_graph)
        if x is None:
            return self._empty_result()

        # GNN forward
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        h = torch.mean(x, dim=1, keepdim=False)  # graph-level readout over nodes

        repr_vec = self.repr_proj(h)

        logits = self.classifier(h)
        probs = torch.sigmoid(logits).mean(dim=-1, keepdim=True)

        issues = self._structural_analysis(ast_graph, metadata)

        return {
            "issues": issues,
            "representation": repr_vec,
            "confidence": probs.mean(dim=1),
            "logits": logits,
        }

    def _build_graph_tensor(self, ast_graph):
        nodes = ast_graph.get("nodes", [])
        edges = ast_graph.get("edges", [])

        if not nodes:
            return None, None

        # Node features: one-hot of type, normalized line number, depth
        type_set = sorted(set(n["type"] for n in nodes))
        type_map = {t: i for i, t in enumerate(type_set)}
        n = len(nodes)
        feat_dim = min(len(type_set) + 2, 32)

        x = torch.zeros(n, feat_dim)
        for i, node in enumerate(nodes):
            tid = type_map.get(node["type"], 0)
            x[i, tid % feat_dim] = 1.0
            x[i, -2] = node["line"] / 1000.0
            x[i, -1] = len(node.get("text", "")) / 100.0

        edge_index = torch.tensor([[p, c] for p, c in edges], dtype=torch.long).t()
        if edge_size := edge_index.size(1):
            edge_index = edge_index[:, : min(edge_size, 5000)]

        h = self.node_embed(x)
        return h.unsqueeze(0), edge_index

    def _structural_analysis(self, ast_graph, metadata):
        issues = []
        if not metadata:
            return issues

        content = metadata.get("content", "")
        path = metadata.get("path", "")
        functions = metadata.get("functions", [])

        nodes = ast_graph.get("nodes", [])
        edges = ast_graph.get("edges", [])

        # Detect god class / long function
        if functions:
            func_lines = [f["line"] for f in functions]
            if len(func_lines) > 20:
                issues.append({
                    "file": path,
                    "line": func_lines[0],
                    "severity": "medium",
                    "category": "结构",
                    "description": f"文件中有 {len(functions)} 个函数/类，考虑拆分",
                    "source": "GNN_path",
                })

        # Detect circular dependencies (within the module)
        for pattern, severity, category, desc in self.structural_rules:
            for m in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[: m.start()].count("\n") + 1
                issues.append({
                    "file": path,
                    "line": line_num,
                    "severity": severity,
                    "category": category,
                    "description": desc.format(match=m.group().strip()[:40]),
                    "source": "GNN_path",
                })

        return issues

    def _empty_result(self):
        return {
            "issues": [],
            "representation": torch.zeros(1, self.config.FUSION_DIM),
            "confidence": torch.zeros(1),
        }


class GCNLayer(nn.Module):
    """Simple Graph Convolution layer."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index):
        # x: (1, n, d)
        # edge_index: (2, e)
        x = x.squeeze(0)  # (n, d)
        out = self.weight(x)  # (n, d)

        if edge_index.size(1) > 0:
            src, dst = edge_index
            src = src.clamp(0, x.size(0) - 1)
            dst = dst.clamp(0, x.size(0) - 1)
            out.index_add_(0, dst, x[src])

        out = out + self.bias
        return out.unsqueeze(0)  # (1, n, d)


def _build_structural_patterns():
    return [
        (r'from\s+\.\.', "medium", "结构", "深层相对导入，考虑重构: {match}"),
        (r'class\s+\w+\(.*\bBase\b.*\)', "low", "结构", "可能的基类依赖: {match}"),
        (r'(singleton|factory|adapter|observer)\b', "low", "结构", "设计模式标记: {match}"),
        (r'import\s+\*', "medium", "结构", "通配符导入可能导致命名冲突"),
    ]
