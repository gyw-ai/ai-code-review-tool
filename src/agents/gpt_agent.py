import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from openai import OpenAI
from src.core.git_integration import get_file_content

def main():
    parser = argparse.ArgumentParser(description="GPT Agent - Exploratory Code Review (Local)")
    parser.add_argument("--api-key", default="ollama", help="API key (not needed for Ollama)")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama API base URL")
    parser.add_argument("--model", default="deepseek-r1:7b")
    parser.add_argument("--changed-files", required=True)
    parser.add_argument("--auto-fix-level", default="low,medium")
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

    prompt = f"""你是一个探索型代码审查专家。深度审查代码变更，重点发现：
1. 隐藏的逻辑漏洞和边缘情况
2. 代码设计缺陷和架构问题
3. 性能瓶颈和可优化点
4. 代码可读性和可维护性问题

自定义审查规则：
{custom_rules}

自动修复等级：{args.auto_fix_level}（仅修复该等级及以下问题）
修复要求：必须返回完整的修复后文件内容，不要只返回修改的部分

变更文件：
{json.dumps(files_context, indent=2, ensure_ascii=False)}

请以严格的JSON格式返回结果，不要添加任何额外文本：
{{
  "issues": [
    {{
      "file": "文件路径",
      "line": 行号,
      "severity": "critical/high/medium/low",
      "category": "逻辑/性能/风格/安全",
      "description": "问题详细描述",
      "suggestion": "修复建议",
      "fix_code": "完整的修复后文件内容（如有）"
    }}
  ],
  "summary": "审查总结（300字以内）",
  "has_critical_issues": true/false,
  "auto_fix_applied": true/false
}}"""

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
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

    auto_fix_levels = args.auto_fix_level.split(",")
    fixes = []
    for issue in result.get("issues", []):
        if issue.get("severity") in auto_fix_levels and issue.get("fix_code"):
            fixes.append({
                "file": issue["file"],
                "fix_code": issue["fix_code"]
            })

    result["auto_fixes"] = fixes
    result["auto_fix_applied"] = len(fixes) > 0

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()