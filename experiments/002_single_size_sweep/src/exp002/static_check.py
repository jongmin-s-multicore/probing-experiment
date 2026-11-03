"""Static attribute check (AST): every `alias.attr` where alias is an exp002 module imported in the same file, and every `from exp002.x import name`
(/ `from .x import name`) must be defined at top level of the target module. Catches calls to functions that do not exist (e.g. data.count_train_slices
was missing and neither selftest nor --dry-run reached it). Aliases are collected from imports at any scope in the file; heuristic, no execution."""
from __future__ import annotations

import ast
from pathlib import Path

PKG = "exp002"


def top_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.If, ast.Try)):                       # names defined under if/try at top level
            for x in ast.walk(n):
                if isinstance(x, (ast.FunctionDef, ast.ClassDef)):
                    names.add(x.name)
                elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    names.add(x.id)
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    for a in x.names:
                        names.add((a.asname or a.name).split(".")[0])
    return names


def check(src_dir, extra_files=()):
    """Returns (list of problems, number of checked references)."""
    pkg_dir = Path(src_dir) / PKG
    mods = {p.stem: p for p in pkg_dir.glob("*.py")}
    defs = {m: top_names(ast.parse(p.read_text())) for m, p in mods.items()}
    problems, n = [], 0
    for f in sorted(list(mods.values()) + [Path(x) for x in extra_files]):
        tree = ast.parse(f.read_text())
        alias = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = (node.module or "")
                if node.level == 1 or base == PKG:                 # from . import x / from .x import y / from exp002 import x / from exp002.x import y
                    sub = base if node.level == 1 else ""
                    if node.level == 1 and base == "":
                        for a in node.names:
                            if a.name in mods:
                                alias.setdefault(a.asname or a.name, set()).add(a.name)
                            else:
                                problems.append("%s:%d: 'from . import %s' - no such module" % (f.name, node.lineno, a.name))
                    elif base == PKG:
                        for a in node.names:
                            if a.name in mods:
                                alias.setdefault(a.asname or a.name, set()).add(a.name)
                            else:
                                problems.append("%s:%d: 'from exp002 import %s' - no such module" % (f.name, node.lineno, a.name))
                    else:
                        tgt = sub if node.level == 1 else base[len(PKG) + 1:]
                        if tgt not in mods:
                            problems.append("%s:%d: unknown module %s" % (f.name, node.lineno, tgt))
                            continue
                        for a in node.names:
                            n += 1
                            if a.name != "*" and a.name not in defs[tgt]:
                                problems.append("%s:%d: 'from %s import %s' - %s not defined in %s.py" % (f.name, node.lineno, base, a.name, a.name, tgt))
                elif base.startswith(PKG + "."):
                    tgt = base[len(PKG) + 1:]
                    for a in node.names:
                        n += 1
                        if tgt in mods and a.name not in defs[tgt]:
                            problems.append("%s:%d: 'from %s import %s' - not defined in %s.py" % (f.name, node.lineno, base, a.name, tgt))
        rebound = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)} | {x.arg for x in ast.walk(tree) if isinstance(x, ast.arg)}
        for node in ast.walk(tree):                                  # an alias also used as a variable/parameter name (e.g. matrix M, basis B) is skipped: limitation
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in alias and node.value.id not in rebound:
                n += 1
                targets = alias[node.value.id]
                if not any(node.attr in defs[t] for t in targets):
                    problems.append("%s:%d: %s.%s - not defined in %s" % (f.name, node.lineno, node.value.id, node.attr, "/".join(sorted(t + ".py" for t in targets))))
    return problems, n
