from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class TopologyPath(ABC, nn.Module):
    """Base class for all topology paths.

    Each path encodes the input code using a different topological bias
    (CNN local, RNN sequential, GNN structural, Transformer global),
    then emits issues found through that lens.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.path_name = "base"

    @abstractmethod
    def forward(self, tokens, ast_graph=None, metadata=None):
        """Process code and return issues + representation.

        Args:
            tokens: LongTensor (batch, seq_len) of token ids
            ast_graph: optional graph data for GNN path
            metadata: dict with file path, language etc.

        Returns:
            dict with:
                - "issues": list of issue dicts
                - "representation": FloatTensor (batch, repr_dim) encoded code rep
                - "confidence": FloatTensor (batch,) path confidence
        """
        pass

    def extra_repr(self):
        return f"path={self.path_name}, params={sum(p.numel() for p in self.parameters()):,}"
