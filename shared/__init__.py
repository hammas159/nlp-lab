"""Shared benchmark construction.

Projects in this lab compare methods *against each other*, which only works if they are
scored on the identical corpus and queries. The benchmark therefore lives here rather
than being copied into each project - a duplicate would silently drift, and a comparison
across two slightly different corpora is not a comparison.
"""
