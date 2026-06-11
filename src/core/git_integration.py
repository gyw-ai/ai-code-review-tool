import os
import subprocess
from github import Github

def get_file_content(file_path):
    """获取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_auto_fix_pr(repo, pr_number, base_branch, fixes):
    """
    创建自动修复PR
    :param repo: GitHub仓库对象
    :param pr_number: 原PR编号
    :param base_branch: 原PR的源分支
    :param fixes: 修复列表，格式为[{"file": "path", "fix_code": "content"}]
    :return: 修复PR的URL
    """
    if not fixes:
        return None
    
    # 创建修复分支
    fix_branch = f"fix/ai-auto-fix-{pr_number}"
    try:
        # 获取最新的base分支引用
        base_ref = repo.get_git_ref(f"heads/{base_branch}")
        repo.create_git_ref(f"refs/heads/{fix_branch}", base_ref.object.sha)
        
        # 提交修复文件
        for fix in fixes:
            file_path = fix["file"]
            new_content = fix["fix_code"]
            file = repo.get_contents(file_path, ref=base_branch)
            repo.update_file(
                path=file_path,
                message=f"AI自动修复: {file_path}",
                content=new_content,
                sha=file.sha,
                branch=fix_branch
            )
        
        # 创建PR
        fix_pr = repo.create_pull(
            title=f"🤖 AI自动修复 PR #{pr_number}",
            body=f"这是AI自动生成的修复PR，用于修复PR #{pr_number} 中发现的问题。\n\n请审核后合并到 {base_branch} 分支。",
            head=fix_branch,
            base=base_branch
        )
        
        return fix_pr.html_url
    except Exception as e:
        print(f"创建自动修复PR失败: {e}")
        return None