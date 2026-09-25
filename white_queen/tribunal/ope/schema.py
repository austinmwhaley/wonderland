"""Declared source-schema adapters.

Agnostic ingest must not guess roles from a generic column list with ad-hoc
heuristics — that is how the colony schema leaked next-state columns into the
context. A schema adapter DECLARES the role→columns mapping for a known log
layout, and `to_canonical` consults the adapters first.

Adapters are pure functions `matches(cols) -> bool` plus `roles(cols) ->
dict(context=[...], next_context=[...], exclude=[...])`. Register a new source
with `register_schema(...)`.
"""

from __future__ import annotations

import re

_CTX_PREFIX_PATTERNS = (
    re.compile(r"^o(\d+)$"),
    re.compile(r"^obs[_]?(\d+)$"),
    re.compile(r"^state[_]?(\d+)$"),
    re.compile(r"^x[_]?(\d+)$"),
    re.compile(r"^feature[_]?(\d+)$"),
    re.compile(r"^context[_]?(\d+)$"),
)
_NEXT_PREFIX_PATTERNS = (
    re.compile(r"^n(\d+)$"),
    re.compile(r"^next[_]?(\d+)$"),
    re.compile(r"^next_obs[_]?(\d+)$"),
    re.compile(r"^obs2[_]?(\d+)$"),
    re.compile(r"^next_context[_]?(\d+)$"),
)

# Behavior-metadata columns that are NOT context features.
_METADATA = ("greedy_action", "greedy", "eps", "epsilon", "prob_taken")


def _prefixed_group(cols, patterns):
    for pat in patterns:
        idx = {}
        for c in cols:
            m = pat.match(str(c))
            if m:
                idx[int(m.group(1))] = c
        if idx:
            return [idx[i] for i in range(len(idx)) if i in idx]
    return None


class SchemaAdapter:
    """A declared column-role mapping for one known log layout."""

    def __init__(self, name, match, context, next_context=None, exclude=()):
        self.name = name
        self._match = match
        self._context = context
        self._next = next_context
        self._exclude = tuple(exclude)

    def matches(self, cols):
        return bool(self._match(cols))

    def roles(self, cols):
        ctx = self._context(cols) if callable(self._context) else self._context
        nxt = (
            (self._next(cols) if callable(self._next) else self._next)
            if self._next is not None
            else None
        )
        return {"context": ctx, "next_context": nxt, "exclude": list(self._exclude)}


SCHEMA_ADAPTERS = []


def register_schema(adapter, overwrite=False):
    for i, a in enumerate(SCHEMA_ADAPTERS):
        if a.name == adapter.name:
            if not overwrite:
                raise ValueError(f"schema {adapter.name!r} already registered")
            SCHEMA_ADAPTERS[i] = adapter
            return adapter
    SCHEMA_ADAPTERS.append(adapter)
    return adapter


def _colony_match(cols):
    return (
        _prefixed_group(cols, _CTX_PREFIX_PATTERNS) is not None
        and _prefixed_group(cols, _NEXT_PREFIX_PATTERNS) is not None
    )


def _colony_context(cols):
    return _prefixed_group(cols, _CTX_PREFIX_PATTERNS)


def _colony_next(cols):
    ctx = _prefixed_group(cols, _CTX_PREFIX_PATTERNS)
    nxt = _prefixed_group(cols, _NEXT_PREFIX_PATTERNS)
    if nxt is None or (ctx is not None and len(nxt) != len(ctx)):
        return None
    return nxt


register_schema(
    SchemaAdapter(
        "colony_prefixed", _colony_match, _colony_context, _colony_next, exclude=_METADATA
    )
)


def detect_schema(cols, explicit=None):
    """Return the first matching adapter's roles, or None. `explicit` (a
    columns override from the caller) always wins and suppresses detection."""
    if explicit is not None:
        return None
    for a in SCHEMA_ADAPTERS:
        if a.matches(cols):
            return a.roles(cols)
    return None
