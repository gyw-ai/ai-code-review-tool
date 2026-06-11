import argparse
import json
import os
import subprocess

def run_agent(agent_name, model, changed_files, output_file):
    agent_script = os.path.join("src", "agents", f"{agent_name}_agent.py")
    cmd = [
        "python", agent_script,
        "--model", model,
        "--changed-files", " ".join(changed_files),
        "--output", output_file,
    ]
    print(f"\n{'='*60}")
    print(f"  Running {agent_name.upper()} agent ({model})...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Local AI Code Review Duo Runner")
    parser.add_argument("--files", nargs="+", default=["demo.py"],
                        help="Files to review (default: demo.py)")
    parser.add_argument("--gpt-model", default="deepseek-r1:7b",
                        help="Ollama model for GPT agent")
    parser.add_argument("--claude-model", default="qwen2.5-coder:7b",
                        help="Ollama model for Claude agent")
    parser.add_argument("--mode", choices=["gpt", "claude", "both"], default="both",
                        help="Review mode")
    args = parser.parse_args()

    changed_files = [f for f in args.files if os.path.exists(f)]
    if not changed_files:
        print("No valid files to review")
        return

    gpt_result = None
    claude_result = None

    if args.mode in ("gpt", "both"):
        gpt_result = run_agent("gpt", args.gpt_model, changed_files, "gpt_result.json")

    if args.mode in ("claude", "both"):
        claude_result = run_agent("claude", args.claude_model, changed_files, "claude_result.json")

    print(f"\n{'='*60}")
    print("  MERGED REVIEW RESULT")
    print(f"{'='*60}")

    all_issues = []
    if claude_result:
        for i in claude_result.get("issues", []):
            i["source"] = "Claude (qwen2.5-coder:7b)"
            all_issues.append(i)
        print(f"\nClaude summary: {claude_result.get('summary', 'N/A')}")

    if gpt_result:
        for i in gpt_result.get("issues", []):
            i["source"] = "GPT (deepseek-r1:7b)"
            all_issues.append(i)
        print(f"GPT summary: {gpt_result.get('summary', 'N/A')}")

    seen = {}
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for issue in all_issues:
        key = f"{issue.get('file')}:{issue.get('line')}"
        if key not in seen or severity_order.get(issue["severity"], 0) > severity_order.get(seen[key]["severity"], 0):
            seen[key] = issue
    unique_issues = list(seen.values())
    unique_issues.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 4))

    print(f"\nTotal issues found: {len(unique_issues)}")
    for issue in unique_issues:
        print(f"\n  [{issue['severity'].upper()}] {issue['category']}")
        print(f"  File: {issue['file']}:{issue.get('line', '?')}")
        print(f"  Source: {issue.get('source', '?')}")
        print(f"  Desc: {issue.get('description', 'N/A')}")
        if issue.get("suggestion"):
            print(f"  Suggestion: {issue['suggestion']}")

    summary = {
        "total": len(unique_issues),
        "critical": len([i for i in unique_issues if i["severity"] == "critical"]),
        "high": len([i for i in unique_issues if i["severity"] == "high"]),
        "medium": len([i for i in unique_issues if i["severity"] == "medium"]),
        "low": len([i for i in unique_issues if i["severity"] == "low"]),
    }
    print(f"\n{'='*60}")
    print(f"  Summary: {json.dumps(summary, indent=2)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
