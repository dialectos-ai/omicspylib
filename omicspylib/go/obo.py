"""
Minimal reader for Gene Ontology files in OBO 1.2 format.

Only what is needed to walk the GO graph offline is kept: names, namespaces,
alternative ids, obsolete terms with their ``replaced_by`` and every
``is_a`` / ``relationship`` edge. No dependency beyond the standard library.
"""

from __future__ import annotations

import gzip
import os
import re
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_STANZA_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")


@dataclass(frozen=True)
class GeneOntology:
    """
    A parsed Gene Ontology, as read by :func:`load_obo`.

    Ids are always GO primary ids unless stated otherwise. Every term found in
    the file, obsolete or not, has an entry in ``names`` (an empty string when
    the term has no ``name`` line).

    Attributes
    ----------
    data_version: str or None
        The header ``data-version``, e.g. ``"releases/2026-07-26"``.
    format_version: str or None
        The header ``format-version``, e.g. ``"1.2"``.
    names: dict
        Primary id to term name.
    namespaces: dict
        Primary id to ``biological_process``, ``molecular_function`` or
        ``cellular_component``.
    alt_ids: dict
        Alternative id to primary id.
    obsolete: frozenset
        Primary ids of obsolete terms.
    replaced_by: dict
        Obsolete id to its replacement. Only terms with exactly one
        ``replaced_by`` line are kept; more than one is ambiguous.
    edges: dict
        Relation name (``is_a`` or a relationship type such as ``part_of``)
        to a mapping of child id to its parent ids.
    """

    data_version: str | None
    format_version: str | None
    names: dict[str, str]
    namespaces: dict[str, str]
    alt_ids: dict[str, str]
    obsolete: frozenset[str]
    replaced_by: dict[str, str]
    edges: dict[str, dict[str, frozenset[str]]]

    def __contains__(self, go_id: object) -> bool:
        """True for a primary id or an alternative id present in the file."""
        return isinstance(go_id, str) and (go_id in self.names or go_id in self.alt_ids)

    def primary(self, go_id: str) -> str:
        """Return the primary id of ``go_id`` (itself when it is not an alternative id)."""
        return self.alt_ids.get(go_id, go_id)

    def ancestors(self, go_id: str, relations: Iterable[str]) -> set[str]:
        """
        Return all terms reachable upwards from ``go_id``, including the term itself.

        Plain graph reachability over the union of ``relations``, taken
        literally (no relation is expanded into its sub-relations, and no
        composition rules are applied). ``go_id`` is first resolved to its
        primary id; an id not in the ontology gives an empty set.
        """
        start = self.primary(go_id)
        if start not in self.names:
            return set()
        graphs = [self.edges[rel] for rel in set(relations) if rel in self.edges]
        seen = {start}
        stack = [start]
        while stack:
            term = stack.pop()
            for graph in graphs:
                for parent in graph.get(term, ()):
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
        return seen


def _read_lines(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    else:
        with path.open(encoding="utf-8") as fh:
            yield from fh


def _first_token(value: str) -> str:
    """First whitespace-delimited token, i.e. without trailing qualifiers or ``! comment``."""
    return value.split(maxsplit=1)[0] if value.strip() else ""


def load_obo(path: str | os.PathLike[str]) -> GeneOntology:
    """
    Parse a Gene Ontology OBO file (plain ``.obo`` or gzipped ``.obo.gz``).

    Only ``[Term]`` stanzas with a ``GO:`` id are read; ``[Typedef]`` and other
    stanzas are skipped. Every relationship type found is stored under its own
    name; the parser does not decide which relations matter.
    ``intersection_of``, ``consider``, ``xref``, ``synonym`` and ``def`` lines
    are ignored. A term with more than one ``replaced_by`` line is treated as
    having none, and those terms are counted in a single warning.

    Parse once and pass the result to :func:`omicspylib.go_to_goslim` (argument
    ``obo``) when mapping more than once.

    Parameters
    ----------
    path: str or os.PathLike
        Path to the OBO file. A ``.gz`` suffix means gzip.

    Returns
    -------
    GeneOntology
        The parsed ontology.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file has no GO ``[Term]`` stanza.

    Notes
    -----
    GO files are distributed under CC BY 4.0. The caller is responsible for
    keeping the file, and for recording its release (``data_version``) and
    source.

    Examples
    --------
    >>> from omicspylib import load_obo
    >>> go = load_obo("data/go/go-basic.obo")  # doctest: +SKIP
    >>> go.data_version  # doctest: +SKIP
    'releases/2026-07-26'
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OBO file not found: {path}")

    header: dict[str, str] = {}
    names: dict[str, str] = {}
    namespaces: dict[str, str] = {}
    alt_ids: dict[str, str] = {}
    obsolete: set[str] = set()
    replaced_by: dict[str, list[str]] = {}
    edges: dict[str, dict[str, set[str]]] = {}

    stanza: str | None = None
    term: dict[str, list[str]] = {}

    def flush() -> None:
        if stanza != "Term" or not term.get("id"):
            return
        go_id = term["id"][0]
        if not go_id.startswith("GO:") or go_id in names:
            return
        names[go_id] = term.get("name", [""])[0]
        if "namespace" in term:
            namespaces[go_id] = term["namespace"][0]
        for alt in term.get("alt_id", []):
            alt_ids[alt] = go_id
        if term.get("is_obsolete", ["false"])[0] == "true":
            obsolete.add(go_id)
        if "replaced_by" in term:
            replaced_by[go_id] = term["replaced_by"]
        for parent in term.get("is_a", []):
            edges.setdefault("is_a", {}).setdefault(go_id, set()).add(parent)
        for rel_value in term.get("relationship", []):
            parts = rel_value.split()
            if len(parts) > 1:
                edges.setdefault(parts[0], {}).setdefault(go_id, set()).add(parts[1])

    for raw in _read_lines(path):
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        match = _STANZA_RE.match(line)
        if match:
            flush()
            stanza = match.group("name")
            term = {}
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.strip()
        if stanza is None:
            header.setdefault(key.strip(), value)
            continue
        key = key.strip()
        if key in ("id", "alt_id", "is_a", "replaced_by", "is_obsolete"):
            value = _first_token(value)
        elif key == "relationship":
            value = value.split("!", 1)[0].strip()
        elif key not in ("name", "namespace"):
            continue
        term.setdefault(key, []).append(value)
    flush()

    if not names:
        raise ValueError(f"No GO [Term] stanza found in {path}")

    ambiguous = sorted(go_id for go_id, targets in replaced_by.items() if len(set(targets)) > 1)
    if ambiguous:
        warnings.warn(
            f"{len(ambiguous)} term(s) with more than one replaced_by are treated as having none "
            f"(first: {', '.join(ambiguous[:5])}).",
            UserWarning,
            stacklevel=2,
        )

    return GeneOntology(
        data_version=header.get("data-version"),
        format_version=header.get("format-version"),
        names=names,
        namespaces=namespaces,
        alt_ids=alt_ids,
        obsolete=frozenset(obsolete),
        replaced_by={go_id: targets[0] for go_id, targets in replaced_by.items() if len(set(targets)) == 1},
        edges={rel: {child: frozenset(parents) for child, parents in graph.items()} for rel, graph in edges.items()},
    )
