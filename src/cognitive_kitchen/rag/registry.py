"""Plugin registry with package auto-discovery.

A strategy file declares itself with @register; dropping the file into the
package is the whole of installation. Nothing keeps a list of strategies.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections import defaultdict
from typing import Any, Callable

_REG: dict[str, dict[str, Callable[..., Any]]] = defaultdict(dict)
_DISCOVERED: set[str] = set()


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        _REG[kind][name] = factory
        return factory
    return deco


def discover(*packages: str) -> None:
    """Import every module in each package so its @register calls execute."""
    for pkg in packages:
        if pkg in _DISCOVERED:
            continue
        module = importlib.import_module(pkg)
        for _, modname, _ in pkgutil.iter_modules(module.__path__):
            if modname.startswith("_"):
                continue
            importlib.import_module(f"{pkg}.{modname}")
        _DISCOVERED.add(pkg)


def available(kind: str) -> list[str]:
    return sorted(_REG[kind])


def build(kind: str, name: str, **params: Any) -> Any:
    if name not in _REG[kind]:
        raise KeyError(f"unknown {kind} {name!r}; have {available(kind)}")
    return _REG[kind][name](**params)


def signature(kind: str, name: str) -> dict[str, Any]:
    """Default parameters of a factory, so the UI can render controls."""
    import inspect

    sig = inspect.signature(_REG[kind][name])
    return {p.name: p.default for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty}