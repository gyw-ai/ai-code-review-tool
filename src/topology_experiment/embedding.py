import torch
import torch.nn as nn


class CodeEmbedding(nn.Module):
    """Shared token embedding layer for all topology paths."""

    def __init__(self, config, vocab_size=32000):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(vocab_size, config.EMBED_DIM, padding_idx=0)
        self.pos_embed = PositionalEncoding(config.EMBED_DIM, max_len=config.MAX_TOKENS)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens):
        """tokens: LongTensor (batch, seq_len)"""
        x = self.token_embed(tokens)
        x = self.pos_embed(x)
        x = self.dropout(x)
        return x  # (batch, seq_len, embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SimpleTokenizer:
    """Minimal BPE-free tokenizer for code files.
    Maps tokens to ids. For real use, replace with a proper tokenizer.
    """

    def __init__(self, max_vocab=32000):
        self.word2id = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.id2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self.max_vocab = max_vocab

    def fit(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        for word, _ in counter.most_common(self.max_vocab - 4):
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word

    def encode(self, text, max_len=2048):
        tokens = [2]  # <bos>
        for w in self._tokenize(text)[: max_len - 2]:
            tokens.append(self.word2id.get(w, 1))
        tokens.append(3)  # <eos>
        return tokens[:max_len]

    def _tokenize(self, text):
        import re
        return re.findall(r"[a-zA-Z_]\w*|[\d]+|.", text)
