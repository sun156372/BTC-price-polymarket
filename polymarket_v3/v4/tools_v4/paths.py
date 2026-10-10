"""Locate the offline dataset, once, for every tool in this package.

    from paths import resolve_data_root
    root = resolve_data_root(args.root)

The research tools each computed the dataset location from their own file depth
(``dirname`` three times). That works while they sit beside the dataset and breaks
the moment they are copied anywhere else, which is exactly what packaging does. The
symptom is quiet: the tool starts, finds nothing, and reports "no data".

Resolution order, first hit wins:

1. an explicit ``--root`` argument
2. the ``V4_DATA_ROOT`` environment variable
3. ``<package>/../<name>``   - dataset beside the package
4. ``<package>/../../<name>``  - dataset at the workspace root
5. ``<package>/<name>``     - dataset inside the package

``<name>`` defaults to ``micro_v3``, the extraction directory produced by the archive
tool. Nothing is cached, so relocating the data mid-session is picked up immediately.
"""
from __future__ import annotations

import os

PKG = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(PKG)

DEFAULT_NAMES = ("micro_v3", "data")


def candidates(name: str = "micro_v3") -> list[str]:
    """Every location that will be checked, in order."""
    out = []
    env = os.environ.get("V4_DATA_ROOT")
    if env:
        out.append(env)
    out.extend([
        os.path.join(os.path.dirname(V4), name),
        os.path.join(os.path.dirname(os.path.dirname(V4)), name),
        os.path.join(V4, name),
    ])
    return out


def resolve_data_root(explicit: str | None = None, name: str = "micro_v3",
                      required: bool = True) -> str | None:
    """Return the dataset root, or None when ``required`` is False and none exists."""
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        if required:
            raise SystemExit(
                f"--root {explicit} is not a directory.\n"
                f"Point it at the extracted dataset, e.g. --root <workspace>/{name}")
        return None
    for path in candidates(name):
        if path and os.path.isdir(path):
            return path
    if not required:
        return None
    tried = "\n  ".join(candidates(name))
    raise SystemExit(
        "Cannot find the offline dataset. Looked in:\n  "
        f"{tried}\n\n"
        "Fix it with either:\n"
        "  --root <path to the extracted dataset>\n"
        "  set V4_DATA_ROOT=<path>\n\n"
        "If you have never built it, this package cannot be verified offline; the "
        "live paper loop (run_live_v4.py) needs no local dataset.")


def describe() -> str:
    """A short human-readable summary, for tool headers."""
    found = resolve_data_root(required=False)
    if found:
        return f"data root: {found}"
    return ("data root: NOT FOUND - offline tools need --root or V4_DATA_ROOT; "
            "the live loop does not")
