"""Ingredient vocabulary.

Built once before the graph, by `uv run ck-vocab`.
"""
from .ingredients import (all_canonical, canonical, category, reload, resolve,
                          same_ingredient)

__all__ = ["all_canonical", "canonical", "category", "reload", "resolve",
           "same_ingredient"]
