import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from github import Github
from src.core.git_integration import create_auto_fix_pr

def main():
    parser = argparse.ArgumentParser(description="Result Merger")
    parser.add_argument("--github-token", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--review-mode", required=True)
    parser.add_argument("--gpt-result", default="")
    parser.add_argument("--claude-result", default="")
    args = parser.parse_args()

    # 初始化GitHub客户端
    g = Github(args.github_token)
    repo = g.get_repo(args.repo)
    pr = repo.get_pull(args.pr_number)

    # 读取审查结果
    gpt_result = {"issues": [], "summary": "", "has_critical_issues": False, "auto_fixes": []}
    if args.review_mode in ["GPT_ONLY", "DUAL_AGENT"] and os.path.exists(args.gpt_result):
        with open(args.gpt_result, "r", encoding="utf-8") as f:
            gpt_result = json.load(f)
        for issue in gpt_result["issues"]:
            issue["source"] = "🤖 GPT-5.5 (探索型)"

    claude_result = {"issues": [], "summary": "", "has_critical_issues": False}
    if args.review_mode in ["CLAUDE_ONLY", "DUAL_AGENT"] and os.path.exists(args.claude_result):
        with open(args.claude_result, "r", encoding="utf-8") as f:
            claude_result = json.load(f)
        for issue in claude_result["issues"]:
            issue["source"] = "🛡️ Claude Mythos 5 (安全型)"

    # 合并问题（安全问题优先）
    all_issues = claude_result["issues"] + gpt_result["issues"]
    
    # 去重：同一文件同一行的问题只保留最严重的
    seen = {}
    for issue in all_issues:
        key = f"{issue['file']}:{issue['line']}"
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        if key not in seen or severity_order[issue["severity"]] > severity_order[seen[key]["severity"]]:
            seen[key] = issue
    unique_issues = list(seen.values())
    
    # 按严重程度排序
    unique_issues.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x["severity"]])
    
    # 统计问题数量
    critical_count = len([i for i in unique_issues if i["severity"] == "critical"])
    high_count = len([i for i in unique_issues if i["severity"] == "high"])
    medium_count = len([i for i in unique_issues if i["severity"] == "medium"])
    low_count = len([i for i in unique_issues if i["severity"] == "low"])
    has_critical_issues = critical_count > 0

    # 创建自动修复PR（仅开发分支）
    fix_pr_url = None
    if args.review_mode == "GPT_ONLY" and gpt_result.get("auto_fix_applied", False):
        fix_pr_url = create_auto_fix_pr(
            repo=repo,
            pr_number=args.pr_number,
            base_branch=pr.head.ref,
            fixes=gpt_result.get("auto_fixes", [])
        )

    # 生成PR评论
    comment = f"""# 🧠 AI Code Review Duo 审查报告

## 📊 审查概览
| 指标 | 数值 |
|------|------|
| 审查模式 | {args.review_mode} |
| 总问题数 | {len(unique_issues)} |
| 🔴 高危 | {critical_count} |
| 🟠 严重 | {high_count} |
| 🟡 中等 | {medium_count} |
| 🟢 轻微 | {low_count} |

"""

    if has_critical_issues:
        comment += "⚠️ **发现高危问题，本次审查不通过，请修复后重新提交**\n\n"
    else:
        comment += "✅ **审查通过，无高危问题**\n\n"

    if fix_pr_url:
        comment += f"🤖 **AI已自动生成修复PR**: [{fix_pr_url}]({fix_pr_url})\n\n"

    # 问题详情
    if unique_issues:
        comment += "## 🔍 问题详情\n\n"
        for issue in unique_issues:
            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }[issue["severity"]]
            
            comment += f"### {severity_emoji} {issue['severity'].upper()} - {issue['category']}\n"
            comment += f"- **文件**: [{issue['file']}:{issue['line']}]({pr.html_url}/files#diff-{issue['file'].replace('/', '-')}L{issue['line']})\n"
            comment += f"- **来源**: {issue['source']}\n"
            comment += f"- **描述**: {issue['description']}\n"
            if issue.get("suggestion"):
                comment += f"- **建议**: {issue['suggestion']}\n"
            comment += "\n"

    # 总结
    if gpt_result["summary"]:
        comment += f"## 📝 GPT-5.5 探索总结\n{gpt_result['summary']}\n\n"
    if claude_result["summary"]:
        comment += f"## 🛡️ Claude Mythos 5 安全总结\n{claude_result['summary']}\n\n"

    # 发布评论
    pr.create_issue_comment(comment)
    
    # 保存通知数据
    notification_data = {
        "project": args.repo.split("/")[-1],
        "pr_number": args.pr_number,
        "pr_url": pr.html_url,
        "status": "success" if not has_critical_issues else "failure",
        "critical_count": critical_count,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count
    }
    
    with open("notification_data.json", "w", encoding="utf-8") as f:
        json.dump(notification_data, f, indent=2)
    
    # 设置输出变量
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"has_critical_issues={str(has_critical_issues).lower()}\n")
    else:
        print(f"has_critical_issues={str(has_critical_issues).lower()}")

if __name__ == "__main__":
    main()