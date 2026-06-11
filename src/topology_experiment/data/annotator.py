"""Interactive CLI annotator for reviewing topology experiment outputs.

Workflow:
  1. Run review on a file (all paths)
  2. Review each issue found
  3. Mark as correct/incorrect
  4. Add missed issues
  5. Save annotations
"""

import json
import os
import sys


class Annotator:
    def __init__(self, experiment_runner):
        self.runner = experiment_runner
        self.annotations = []

    def annotate_file(self, file_path):
        """Interactive annotation session for a single file."""
        result = self.runner.review(file_path)
        issues = result["merged_issues"]

        print(f"\n{'='*60}")
        print(f"  Annotating: {file_path}")
        print(f"  Issues found by all paths: {len(issues)}")
        print(f"{'='*60}")

        correct = []
        incorrect = []

        for i, issue in enumerate(issues):
            print(f"\n  [{i+1}/{len(issues)}] "
                  f"[{issue.get('severity','?').upper():>8}] "
                  f"{issue.get('category','?')}  "
                  f"L{issue.get('line','?')}")
            print(f"       Source: {issue.get('source','?')}")
            print(f"       {issue.get('description','')[:120]}")

            while True:
                ans = input("  Correct? [Y]es / [N]o / [S]kip: ").strip().lower()
                if ans in ("y", "yes", ""):
                    correct.append(issue)
                    break
                elif ans in ("n", "no"):
                    incorrect.append(issue)
                    break
                elif ans in ("s", "skip"):
                    break

        # Ask about missed issues
        print(f"\n  Any issues the system missed?")
        while True:
            line = input("  Line number (or empty to finish): ").strip()
            if not line:
                break
            try:
                line_num = int(line)
            except ValueError:
                continue
            cat = input("  Category (安全/逻辑/性能/风格/结构): ").strip()
            desc = input("  Description: ").strip()
            sev = input("  Severity (critical/high/medium/low): ").strip() or "medium"
            self.annotations.append({
                "file": file_path,
                "line": line_num,
                "category": cat,
                "severity": sev,
                "description": desc,
                "source": "human",
                "missed_by_system": True,
            })

        # Save per-file annotation
        annotation = {
            "file": file_path,
            "n_issues_total": len(issues),
            "correct": [self._summarize(i) for i in correct],
            "incorrect": [self._summarize(i) for i in incorrect],
            "human_added": [a for a in self.annotations if a.get("missed_by_system")],
        }
        return annotation

    def _summarize(self, issue):
        return {
            "line": issue.get("line"),
            "category": issue.get("category"),
            "severity": issue.get("severity"),
            "description": issue.get("description", "")[:80],
            "source": issue.get("source"),
        }
