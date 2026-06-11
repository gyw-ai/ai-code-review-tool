import os

class Config:
    # Paths
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    OLLAMA_API_KEY = "ollama"

    # Embedding
    EMBED_DIM = 768
    MAX_TOKENS = 2048

    # CNN path
    CNN_KERNEL_SIZES = [3, 5, 7]
    CNN_CHANNELS = [128, 128, 128]
    CNN_DROPOUT = 0.2

    # RNN path
    RNN_HIDDEN = 256
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.2
    RNN_BIDIRECTIONAL = True

    # GNN path
    GNN_HIDDEN = 128
    GNN_LAYERS = 3
    GNN_DROPOUT = 0.2

    # Transformer path
    TRANSFORMER_HEADS = 8
    TRANSFORMER_LAYERS = 4
    TRANSFORMER_DIM = 512

    # Router
    ROUTER_HIDDEN = 128
    ROUTER_PATHS = ["cnn", "rnn", "transformer", "gnn"]

    # Fusion
    FUSION_DIM = 512
    FUSION_HEADS = 4

    # Issue categories
    ISSUE_CATEGORIES = {
        "安全": ["sql_injection", "xss", "command_injection", "hardcoded_secret"],
        "逻辑": ["null_deref", "off_by_one", "race_condition", "infinite_loop"],
        "性能": ["nplus1_query", "unbounded_memory", "slow_algorithm"],
        "风格": ["naming_convention", "dead_code", "magic_number"],
        "结构": ["circular_dep", "god_class", "leaky_abstraction"],
    }

    # Experiment
    DEFAULT_OUTPUT = "experiment_results.json"
