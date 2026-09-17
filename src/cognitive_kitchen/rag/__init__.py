"""Retrieval-augmented pipeline: loaders, chunkers, retrievers, evaluation.

The build side (everything outside `eval`) must never read the golden dataset.
It is the answer key, not training material.
"""