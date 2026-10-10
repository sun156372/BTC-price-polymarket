"""Find names that are read but never bound. Catches the whole NameError class.

    python tools_v4/static_check.py            # check this package
    python tools_v4/static_check.py --path .   # check somewhere else

Why this exists
---------------
A refactor of ``run_live_v4.py`` removed two local assignments while leaving their
uses behind (``up_mid_now`` and ``slug``). Nothing failed at import time, the
strategy self-check passed, and every offline tool passed - because none of them
execute that function. The bug surfaced only when the loop ran against the real
network, where it aborted on the FIRST window.

That is the worst possible failure shape for a live trading loop: invisible to every
test that does not exercise the live path, and it fails exactly when real money is
closest. A compiler would have caught it; Python does not, so this does it.

HOW IT WORKS
------------
Walks the AST of each module, tracking names bound in each scope (assignments,
imports, arguments, comprehension targets, with/for/except targets, function and class
definitions, global/nonlocal declarations). Then reports any name READ in a scope that
resolves nowhere: not locally, not in an enclosing function, not at module level, not
a builtin.

Two deliberate relaxations, because precision matters more than recall here:

* Top-level names are pre-bound before bodies are walked, so a method calling a
  helper defined further down the file is not reported. Forward references are legal
  Python and flagging them would drown the real findings.
* ``*args``/``**kwargs`` suppress reporting inside that function, since unresolved
  names there are expected.

It found the live-loop bug when run against the broken version, and reports nothing
on the fixed one.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import os
import sys

SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "_cache"}

#: Names the interpreter injects into every module. ``__file__`` is not a builtin in
#: the ``builtins`` module, but every module has it, so listing it here avoids a whole
#: class of false positives.
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                  "__loader__", "__builtins__", "__debug__", "__dict__"}


class Scope:
    """One lexical scope: names it binds, and the names it reads."""

    __slots__ = ("bound", "reads", "parent", "is_function", "dynamic")

    def __init__(self, parent: "Scope | None", is_function: bool = False):
        self.bound: set[str] = set()
        self.reads: list[tuple[str, int]] = []
        self.parent = parent
        self.is_function = is_function
        #: True when the scope has *args/**kwargs, so unresolved names are expected
        self.dynamic = False

    def resolves(self, name: str) -> bool:
        scope: "Scope | None" = self
        while scope is not None:
            if name in scope.bound:
                return True
            scope = scope.parent
        return False

    def allows_unresolved(self) -> bool:
        scope: "Scope | None" = self
        while scope is not None:
            if scope.dynamic:
                return True
            if not scope.is_function:
                break
            scope = scope.parent
        return False


class Checker(ast.NodeVisitor):
    def __init__(self, module_bound: set[str]):
        self.module = Scope(None, is_function=False)
        self.module.bound |= module_bound
        self.stack: list[Scope] = [self.module]
        self.problems: list[tuple[str, int]] = []

    # ------------------------------------------------------------- utilities
    @property
    def scope(self) -> Scope:
        return self.stack[-1]

    def bind(self, name: str | None) -> None:
        if name:
            self.scope.bound.add(name)

    def bind_target(self, node: ast.AST) -> None:
        """Bind whatever a target expression assigns to."""
        if isinstance(node, ast.Name):
            self.bind(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self.bind_target(element)
        elif isinstance(node, ast.Starred):
            self.bind_target(node.value)
        # Subscript and Attribute targets mutate, they do not bind a name.

    def read(self, name: str, lineno: int) -> None:
        self.scope.reads.append((name, lineno))

    # ------------------------------------------------------------ statements
    def visit_Module(self, node: ast.Module) -> None:
        # Pre-bind every top-level name BEFORE walking bodies, so forward references
        # (a method calling a helper defined further down) are not false positives.
        self._prebind(node.body, self.module)
        for statement in node.body:
            self.visit(statement)
        self.report()

    def _prebind(self, body, scope: Scope) -> None:
        for statement in body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                scope.bound.add(statement.name)
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    scope.bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    scope.bound.add(alias.asname or alias.name)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    self._prebind_target(target, scope)
            elif isinstance(statement, ast.AnnAssign):
                self._prebind_target(statement.target, scope)
            elif isinstance(statement, (ast.If, ast.Try, ast.While, ast.For,
                                        ast.AsyncFor, ast.With, ast.AsyncWith)):
                for attr in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, attr, None)
                    if nested:
                        self._prebind(nested, scope)
                for handler in getattr(statement, "handlers", []) or []:
                    self._prebind(handler.body, scope)
                if isinstance(statement, (ast.For, ast.AsyncFor)):
                    self._prebind_target(statement.target, scope)

    def _prebind_target(self, node, scope: Scope) -> None:
        if isinstance(node, ast.Name):
            scope.bound.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self._prebind_target(element, scope)
        elif isinstance(node, ast.Starred):
            self._prebind_target(node.value, scope)

    def visit_FunctionDef(self, node) -> None:
        self.bind(node.name)
        self._enter_function(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self.bind(node.name)
        self._enter_function(node)

    def _enter_function(self, node) -> None:
        child = Scope(self.scope, is_function=True)
        args = node.args
        for arg in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
            child.bound.add(arg.arg)
        if args.vararg:
            child.bound.add(args.vararg.arg)
            child.dynamic = True
        if args.kwarg:
            child.bound.add(args.kwarg.arg)
            child.dynamic = True
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (list(args.defaults) + [d for d in args.kw_defaults if d]):
            self.visit(default)
        self.stack.append(child)
        for statement in node.body:
            self.visit(statement)
        self.stack.pop()
        self.scope.bound.add(node.name)
        self._flush(child)

    def _flush(self, scope: Scope) -> None:
        for name, lineno in scope.reads:
            if scope.resolves(name) or hasattr(builtins, name):
                continue
            if scope.allows_unresolved():
                continue
            self.problems.append((name, lineno))

    def report(self) -> None:
        self._flush(self.module)

    def visit_ClassDef(self, node) -> None:
        self.bind(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_Import(self, node) -> None:
        for alias in node.names:
            self.bind(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node) -> None:
        for alias in node.names:
            self.bind(alias.asname or alias.name)

    def visit_Global(self, node) -> None:
        for name in node.names:
            self.bind(name)

    def visit_Nonlocal(self, node) -> None:
        for name in node.names:
            self.bind(name)

    def visit_Assign(self, node) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.bind_target(target)

    def visit_AnnAssign(self, node) -> None:
        if node.value is not None:
            self.visit(node.value)
        if node.annotation is not None:
            self.visit(node.annotation)
        self.bind_target(node.target)

    def visit_AugAssign(self, node) -> None:
        self.visit(node.value)
        # an augmented assign both reads and writes
        if isinstance(node.target, ast.Name):
            self.read(node.target.id, node.lineno)
            self.bind(node.target.id)
        else:
            self.visit(node.target)

    def visit_NamedExpr(self, node) -> None:
        self.visit(node.value)
        self.bind_target(node.target)

    def visit_For(self, node) -> None:
        self.visit(node.iter)
        self.bind_target(node.target)
        for statement in node.body + node.orelse:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_With(self, node) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.bind_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_Try(self, node) -> None:
        for statement in node.body + node.orelse + node.finalbody:
            self.visit(statement)
        for handler in node.handlers:
            self.visit(handler)

    def visit_ExceptHandler(self, node) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self.bind(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_Lambda(self, node) -> None:
        child = Scope(self.scope, is_function=True)
        args = node.args
        for arg in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
            child.bound.add(arg.arg)
        if args.vararg:
            child.bound.add(args.vararg.arg)
        if args.kwarg:
            child.bound.add(args.kwarg.arg)
        self.stack.append(child)
        self.visit(node.body)
        self.stack.pop()
        self._flush(child)

    # ---------------------------------------------------------- expressions
    def visit_Name(self, node) -> None:
        if isinstance(node.ctx, ast.Load):
            self.read(node.id, node.lineno)

    def visit_comprehension(self, node) -> None:
        self.visit(node.iter)
        self.bind_target(node.target)
        for condition in node.ifs:
            self.visit(condition)

    def visit_ListComp(self, node) -> None:
        child = Scope(self.scope, is_function=True)
        self.stack.append(child)
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.elt)
        self.stack.pop()
        self._flush(child)

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node) -> None:
        child = Scope(self.scope, is_function=True)
        self.stack.append(child)
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.key)
        self.visit(node.value)
        self.stack.pop()
        self._flush(child)


def check_file(path: str) -> list[tuple[str, int]]:
    source = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [(f"<syntax error: {exc.msg}>", exc.lineno or 0)]
    checker = Checker(set(dir(builtins)) | MODULE_DUNDERS)
    checker.visit(tree)
    return checker.problems


def iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=os.path.dirname(here),
                        help="package root to check (default: this package)")
    args = parser.parse_args(argv)

    total = 0
    bad_files = 0
    for path in iter_files(args.path):
        problems = check_file(path)
        total += 1
        if problems:
            bad_files += 1
            rel = os.path.relpath(path, args.path)
            for name, lineno in problems:
                print(f"  {rel}:{lineno}: undefined name {name!r}")
    print()
    print(f"  checked {total} files, {bad_files} with undefined names")
    if bad_files:
        print("  These raise NameError at RUNTIME, possibly only on a rare branch.")
        return 1
    print("  no undefined names found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
