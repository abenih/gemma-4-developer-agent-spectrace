"""
Dual-Loop Verifier for SpecTrace.
Inner Loop: Deterministic AST linting and diff sanity checks.
Outer Loop: Speculative execution sandbox with differential test verification.
"""

import ast
import os
import subprocess
from typing import Tuple, Dict, Any

class InnerLoopVerifier:
    """Pre-execution validation: validates Python AST syntax and diff format."""
    
    @staticmethod
    def validate_patch_syntax(code_string: str) -> Tuple[bool, str]:
        """Ensures that modified code is valid Python bytecode."""
        try:
            ast.parse(code_string)
            return True, "AST syntax valid"
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    @staticmethod
    def extract_modified_blocks(patch_text: str) -> Dict[str, str]:
        """Parses unified diff format to extract target filenames and changed hunks."""
        files = {}
        current_file = None
        for line in patch_text.splitlines():
            if line.startswith('--- a/') or line.startswith('--- '):
                continue
            if line.startswith('+++ b/') or line.startswith('+++ '):
                current_file = line.split('/')[-1].strip()
                files[current_file] = []
            elif current_file and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                files[current_file].append(line)
        return {k: '\n'.join(v) for k, v in files.items()}

class OuterLoopVerifier:
    """Speculative execution loop: runs targeted test suites in an isolated sandbox."""

    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir

    def run_reproduction_test(self, test_command: str, timeout: int = 30) -> Tuple[bool, str]:
        """
        Runs the failing reproduction test.
        Returns (passed, stdout/stderr output).
        """
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=self.repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout
            )
            passed = (result.returncode == 0)
            return passed, result.stdout
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out (exceeded 30s limit)"
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
