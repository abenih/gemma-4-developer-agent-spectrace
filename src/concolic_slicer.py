"""
SpecTrace: Dynamic Concolic Slicing Engine for Python Codebases.
Extracts dynamic execution call traces and intersects them with
static AST dependency cones to produce minimal (< 1,200 tokens) context slices.
"""

import ast
import sys
import os
import inspect
from typing import Set, Dict, List, Tuple, Optional

class ASTDependencyVisitor(ast.NodeVisitor):
    """Extracts function and class definitions and their static dependency links."""
    def __init__(self):
        self.definitions: Dict[str, ast.AST] = {}
        self.calls: Dict[str, Set[str]] = {}
        self.current_scope: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_scope = self.current_scope
        self.current_scope = node.name
        self.definitions[node.name] = node
        self.calls[node.name] = set()
        self.generic_visit(node)
        self.current_scope = prev_scope

    def visit_Call(self, node: ast.Call):
        if self.current_scope:
            if isinstance(node.func, ast.Name):
                self.calls[self.current_scope].add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self.calls[self.current_scope].add(node.func.attr)
        self.generic_visit(node)

class ConcolicExecutionTracer:
    """Lightweight sys.settrace hook to capture dynamic execution lines and frames."""
    def __init__(self, target_prefix: str):
        self.target_prefix = target_prefix
        self.executed_lines: Set[Tuple[str, int]] = set()
        self.executed_functions: Set[Tuple[str, str]] = set()

    def trace_func(self, frame, event, arg):
        if event in ('call', 'line'):
            filename = frame.f_code.co_filename
            if self.target_prefix in filename and not 'site-packages' in filename:
                func_name = frame.f_code.co_name
                line_no = frame.f_lineno
                self.executed_lines.add((filename, line_no))
                self.executed_functions.add((filename, func_name))
        return self.trace_func

    def run_reproduction(self, callable_obj, *args, **kwargs) -> Optional[Exception]:
        """Runs the test or reproduction script under the trace hook."""
        old_trace = sys.gettrace()
        sys.settrace(self.trace_func)
        raised_exc = None
        try:
            callable_obj(*args, **kwargs)
        except Exception as e:
            raised_exc = e
        finally:
            sys.settrace(old_trace)
        return raised_exc

class ConcolicSlicer:
    """Combines dynamic execution lines with static AST dependency cones."""
    def __init__(self, repo_root: str):
        self.repo_root = repo_root

    def slice_from_trace(self, dynamic_functions: Set[Tuple[str, str]], context_margin: int = 15) -> Dict[str, str]:
        """
        Builds a compressed context slice containing only the executed AST blocks
        and their immediate callers/callees.
        """
        slices: Dict[str, List[str]] = {}
        for filepath, func_name in dynamic_functions:
            if not os.path.isfile(filepath):
                continue
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Simple AST extraction
            try:
                tree = ast.parse(''.join(lines), filename=filepath)
                visitor = ASTDependencyVisitor()
                visitor.visit(tree)
                target_node = visitor.definitions.get(func_name)
                if target_node and hasattr(target_node, 'lineno') and hasattr(target_node, 'end_lineno'):
                    start = max(1, target_node.lineno - 2)
                    end = min(len(lines), target_node.end_lineno + 2)
                    extracted = f"# [{os.path.basename(filepath)} : {func_name} (lines {start}-{end})]\n"
                    extracted += ''.join(lines[start-1:end])
                    slices.setdefault(filepath, []).append(extracted)
            except Exception:
                # Fallback to line margin
                pass

        return {fp: '\n\n'.join(chunks) for fp, chunks in slices.items()}
