"""Transformer path: global dependency analysis via Ollama LLM.

Topological bias: GLOBAL — captures long-range dependencies,
cross-function relationships, architectural issues.
"""

import json
import requests

from .base import TopologyPath


class TransformerPath(TopologyPath):
    def __init__(self, config):
        super().__init__(config)
        self.path_name = "transformer"
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL

    def forward(self, tokens, ast_graph=None, metadata=None):
        content = metadata.get("content", "") if metadata else ""
        path = metadata.get("path", "") if metadata else ""

        issues = self._call_ollama(content, path)

        return {
            "issues": issues,
            "representation": None,
            "confidence": None,
        }

    def _call_ollama(self, content, path):
        prompt = f"""你是一个代码审查专家。分析以下代码，以JSON数组格式返回发现的问题。
只关注需要全局上下文才能发现的深层问题：跨函数依赖、架构设计缺陷、数据流错误。
不要报告局部风格问题。

代码 ({path}):
```python
{content[:8000]}
```

JSON格式：
[
  {{
    "file": "{path}",
    "line": 行号,
    "severity": "critical/high/medium/low",
    "category": "安全/逻辑/性能/结构",
    "description": "问题描述",
    "suggestion": "修复建议"
  }}
]"""

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 2048,
                },
                timeout=120,
            )
            content = resp.json()["choices"][0]["message"]["content"]
            # Extract JSON array from response
            import re
            m = re.search(r'\[[\s\S]*\]', content)
            if m:
                return json.loads(m.group())
            return []
        except Exception as e:
            print(f"  [Transformer path] Ollama call failed: {e}")
            return []
