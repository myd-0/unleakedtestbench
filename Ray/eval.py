import os
import re
import sys
import ast
import json
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

code_import = """
import os
import re
import math
import random
import string
import warnings
import datetime
import traceback
try:
    import numpy
    import numpy as np
except ImportError:
    pass
try:
    import pandas
    import pandas as pd
except ImportError:
    pass
try:
    import pytest
except ImportError:
    pass
from typing import List, Dict, Any, Optional, Union, Tuple, Set, FrozenSet, Sequence, Iterable, Generator, Callable
"""


def rename_test_functions(test_code):
    """Ensure unique test function names by prepending a counter."""
    counter = 0
    new_lines = []
    pattern = re.compile(r"(\s*def\s+)(test_)(.*)")
    for line in test_code.split('\n'):
        match = pattern.match(line)
        if match:
            counter += 1
            new_lines.append(f"{match.group(1)}test_{counter}_{match.group(3)}")
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


def sanitize_test_case(test_code):
    """Normalize common LLM formatting so tests can be executed as Python."""
    cleaned = test_code.strip()
    cleaned = re.sub(r"^\s*```(?:python)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)

    func_start = re.search(r"(?m)^\s*def\s+test_[A-Za-z0-9_]*\s*\(", cleaned)
    if func_start:
        cleaned = cleaned[func_start.start():].lstrip()

    lines = cleaned.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    return cleaned


def parse_coverage_json(coverage_json_path, module_name="mod.py"):
    """Parse coverage.json for line and branch coverage of the module."""
    lcov = 0.0
    bcov = 0.0

    try:
        with open(coverage_json_path) as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return lcov, bcov

    file_data = None
    for key in report.get("files", {}):
        if key.endswith(module_name):
            file_data = report["files"][key]
            break

    if file_data is None:
        return lcov, bcov

    summary = file_data.get("summary", {})

    num_statements = summary.get("num_statements", 0)
    covered_lines = summary.get("covered_lines", 0)
    if num_statements > 0:
        lcov = covered_lines / num_statements * 100

    num_branches = summary.get("num_branches", 0)
    covered_branches = summary.get("covered_branches", 0)
    if num_branches > 0:
        bcov = covered_branches / num_branches * 100

    return lcov, bcov


def parse_pytest_summary(output):
    """Extract passed/failed counts from pytest stdout."""
    passed = 0
    failed = 0
    # Match both verbose ("=== 2 passed in 0.5s ===") and quiet ("2 passed in 0.05s")
    p = re.search(r"(\d+)\s+passed", output)
    f = re.search(r"(\d+)\s+failed", output)
    if p:
        passed = int(p.group(1))
    if f:
        failed = int(f.group(1))
    return passed, failed


def evaluate_one(instance, num_tests):
    """Evaluate a single function's generated tests. Returns metrics dict."""
    func_name = instance.get("func_name", "unknown")
    code = instance["code"]
    tests = [sanitize_test_case(t) for t in instance["tests"][:num_tests]]

    result = {
        "task_id": instance.get("task_id"),
        "func_name": func_name,
        "passed": 0,
        "failed": 0,
        "total": 0,
        "lcov": 0.0,
        "bcov": 0.0,
        "status": "error",
    }

    with tempfile.TemporaryDirectory() as tmp:
        # Write mod.py
        mod_code = code_import + "\n\n" + code + "\n"
        mod_path = os.path.join(tmp, "mod.py")
        with open(mod_path, "w") as f:
            f.write(mod_code)

        # Write test.py
        test_code = code_import + "\n\nfrom mod import *\n\n"
        for t in tests:
            test_code += t + "\n\n"
        test_code = rename_test_functions(test_code)
        test_path = os.path.join(tmp, "test.py")
        with open(test_path, "w") as f:
            f.write(test_code)

        # Run pytest with coverage
        cov_json = os.path.join(tmp, "coverage.json")
        cmd = [
            sys.executable, "-m", "pytest", test_path,
            f"--cov={tmp}",
            "--cov-branch",
            f"--cov-report=json:{cov_json}",
            "--cov-report=",
            "--no-header",
            "-q",
        ]

        try:
            proc = subprocess.run(
                cmd, cwd=tmp, capture_output=True, text=True, timeout=60
            )
            passed, failed = parse_pytest_summary(proc.stdout)
            result["passed"] = passed
            result["failed"] = failed
            result["total"] = passed + failed

            lcov, bcov = parse_coverage_json(cov_json)
            result["lcov"] = round(lcov, 2)
            result["bcov"] = round(bcov, 2)
            result["status"] = "success"

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
        except Exception as e:
            result["status"] = f"error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated tests (Pass@k, LCov@k, BCov@k)")
    parser.add_argument("--input", type=str, required=True, help="Path to results JSONL file")
    parser.add_argument("--num_tests", type=int, default=5, help="Number of tests per function to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = json.load(f)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    print(f"Evaluating {len(data)} functions, using up to {args.num_tests} tests each")
    print(f"Input: {args.input}")
    print("-" * 60)

    results = []
    for instance in tqdm(data, desc="Evaluating"):
        r = evaluate_one(instance, args.num_tests)
        results.append(r)

    # Aggregate
    success = [r for r in results if r["status"] == "success"]
    n = len(data)
    n_success = len(success)

    total_passed = sum(r["passed"] for r in results)
    total_tests = sum(r["total"] for r in results)
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    avg_lcov = sum(r["lcov"] for r in success) / n_success if n_success > 0 else 0
    avg_bcov = sum(r["bcov"] for r in success) / n_success if n_success > 0 else 0

    print()
    print("=" * 60)
    print(f"Results: {n_success}/{n} functions evaluated successfully")
    print(f"  Pass@{args.num_tests}:  {pass_rate:.2f}%  ({total_passed}/{total_tests} tests passed)")
    print(f"  LCov@{args.num_tests}:  {avg_lcov:.2f}%")
    print(f"  BCov@{args.num_tests}:  {avg_bcov:.2f}%")
    print("=" * 60)

    # Save per-task results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    input_stem = Path(args.input).stem
    out_file = out_dir / f"eval_{input_stem}_k{args.num_tests}.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Per-task results saved to {out_file}")


if __name__ == "__main__":
    main()
