import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from openai import OpenAI
from src.core.git_integration import get_file_content

def main():
    parser = argparse.ArgumentParser(description="Claude Agent - Security Code Review (Local)")
    parser.add_argument("--api-key", default="ollama", help="API key (not needed for Ollama)")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama API base URL")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--changed-files", required=True)
    parser.add_argument("--security-level", default="strict")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    changed_files = args.changed_files.split(" ")

    files_context = []
    for file_path in changed_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            content = get_file_content(file_path)
            files_context.append({
                "path": file_path,
                "content": content,
                "language": file_path.split(".")[-1]
            })

    custom_rules = ""
    rules_path = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            custom_rules = f.read()

    prompt = f"""你是一个安全型代码审查专家。严格审查代码变更中的安全风险，重点发现：
1. SQL注入、XSS、命令注入等注入漏洞
2. 敏感信息泄露（密码、密钥硬编码）
3. 权限控制和认证缺陷
4. 不安全的文件操作和数据处理
5. 已知漏洞库引用（CVE）

安全等级：{args.security_level}
自定义审查规则：
{custom_rules}

变更文件：
{json.dumps(files_context, indent=2, ensure_ascii=False)}

请以严格的JSON格式返回结果，不要添加任何额外文本：
{{
  "issues": [
    {{
      "file": "文件路径",
      "line": 行号,
      "severity": "critical/high/medium/low",
      "category": "安全/逻辑/配置",
      "description": "问题详细描述",
      "suggestion": "修复建议",
      "fix_code": "完整的修复后文件内容（如有）"
    }}
  ],
  "summary": "安全审查总结（300字以内）",
  "has_critical_issues": true/false,
  "auto_fix_applied": true/false
}}"""

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    import re
    json_match = re.search(r'\{[\s\S]*"issues"[\s\S]*"summary"[\s\S]*\}', content)
    if json_match:
        content = json_match.group()
    content = content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"issues": [], "summary": f"JSON解析失败，原始输出：{content[:500]}", "has_critical_issues": False, "auto_fix_applied": False}

    result["auto_fixes"] = []
    result["auto_fix_applied"] = False

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
