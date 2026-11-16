from __future__ import annotations

import os
import re
import time
import warnings

import requests

from omicspylib.go.obo import GeneOntology, load_obo

_RELATION_RE = re.compile(r"^[a-z_]+$")

# Sub-relations included when ``regulates`` is requested in offline mode.
_REGULATES = ("regulates", "positively_regulates", "negatively_regulates")


def go_to_goslim(
    from_ids: list[str],
    to_ids: list[str],
    relations: str = "is_a,part_of,occurs_in,regulates",
    batch_size: int = 100,
    delay_time: float = 0.1,
    *,
    obo: str | os.PathLike[str] | GeneOntology | None = None,
    minimal: bool = False,
    replace_obsolete: bool = True,
) -> dict[str, list[str]]:
    """
    Map a list of GO term ids, to the corresponding GO-slim terms.

    You might use this function to provide a lookup table
    of the raw GO terms, e.g., as extracted from Uniprot, to a fixed
    "vocabulary" of GO-slim terms, of your interest.

    Once you have this mapping, you can rename the GO terms of your
    original dataset.

    There is no guarantee for 1-to-1 mapping.
    Both ``from_ids`` and ``to_ids`` will be reduced to their unique
    values internally.

    The function has two modes:

    * **online** (``obo=None``, the default): the mapping is computed by the
      EBI QuickGO web service, in batches. The server decides how alternative
      ids and obsolete terms are handled, and the order of the slim ids.
    * **offline** (``obo`` given): the mapping is computed locally from an OBO
      file, with no network call of any kind. ``batch_size`` and ``delay_time``
      are ignored. The semantics are described in the notes below.

    Parameters
    ----------
    from_ids: list
        A list of GO ids, e.g. from your proteins as loaded from Uniprot.
    to_ids: list
        A list of GO ids from your slim set. E.g. a generic slim set
        of your interest.
    relations: str
        The set of relationships over which the slimming information is
        computed, joined with comma.
        Defaults to ``is_a,part_of,occurs_in,regulates`` (see API doc in notes).
    batch_size: int
        Online mode only. Requests to EMBL API are made with batches of the
        specified size. Then, results are merged into one response. Too large
        batches can cause 400 errors.
    delay_time: float
        Online mode only. Delay time in seconds between consecutive requests.
    obo: str, os.PathLike, GeneOntology or None
        Path to a GO OBO file (``.obo`` or ``.obo.gz``), or an ontology
        already parsed with :func:`omicspylib.load_obo`. When given, the
        mapping runs offline. Pass a parsed ontology to avoid re-reading the
        file on every call.
    minimal: bool
        Offline mode only. If True, drop every slim hit that is an ancestor
        (under the same relations) of another slim hit of the same term,
        as in map2slim's minimal mode. If False (default), keep all hits.
        ``minimal=True`` in online mode raises ``ValueError``.
    replace_obsolete: bool
        Offline mode only (ignored online, where the server decides). If True
        (default), an obsolete term with a single ``replaced_by`` is replaced
        by its replacement, following chains. If False, obsolete terms map to
        nothing. Applies to ``from_ids`` and ``to_ids`` alike.

    Returns
    -------
    dict
        Input GO id to a list of slim GO ids. Only ids with at least one slim
        term are keys. Offline, keys are the input ids as passed (stripped),
        alternative ids included, and values are primary slim ids, sorted.

    Raises
    ------
    ValueError
        If ``minimal=True`` without ``obo``; offline, if ``relations`` is
        empty or contains a name with characters other than ``[a-z_]``, or if
        the OBO file has no GO term.
    FileNotFoundError
        Offline, if the OBO path does not exist.

    Warns
    -----
    UserWarning
        Offline, at most once per call for each of: ids in ``from_ids`` not
        in the ontology; a requested relation with no edge in the file;
        ``to_ids`` entries that are unknown or obsolete without replacement.

    Notes
    -----
    For more information about the online endpoint, visit the API
    documentation at:
    https://www.ebi.ac.uk/QuickGO/api/index.html#!/gene_ontology/findSlimsUsingGET

    Offline semantics, for each distinct, stripped id in ``from_ids``:

    1. *Resolve*: an alternative id becomes its primary id. An obsolete term
       is followed through its single ``replaced_by`` (chains followed, cycles
       guarded) when ``replace_obsolete=True``; otherwise, or when it has only
       ``consider`` or nothing, it is unmapped. ``to_ids`` are resolved the
       same way.
    2. *Ancestors*: transitive closure upwards over the union of the requested
       relations, including the term itself. This is plain graph reachability:
       no relation-composition rules are applied (``regulates`` followed by
       ``regulates`` is traversed). Agreement with QuickGO is not claimed.
    3. *Intersect* the ancestors with the resolved slim set.
    4. *Minimal* (``minimal=True``): drop slim hits that are ancestors of
       another slim hit.

    Relation names: ``is_a`` plus any relationship type in the file
    (``part_of``, ``occurs_in``, ``regulates``, ``has_part`` ...).
    ``regulates`` expands to ``regulates``, ``positively_regulates`` and
    ``negatively_regulates``, the latter two being sub-relations of it in GO.

    ``go-basic.obo`` keeps only relations that stay within one GO aspect, so it
    has **no** ``occurs_in`` edges: the default relation set run against it
    silently loses ``occurs_in``. A warning is raised in that case. Use the
    full ``go.obo`` if cross-aspect relations matter.

    GO files are distributed under CC BY 4.0. This library does not ship or
    download them: the caller keeps the file and records its release
    (``GeneOntology.data_version``) and source.

    The defaults of ``minimal`` and ``replace_obsolete`` are part of the
    contract and will not change without a major version bump.

    Examples
    --------
    >>> from omicspylib import go_to_goslim
    >>> from_ids = ['GO:0000922','GO:0035253']
    >>> to_ids = ['GO:0005929', 'GO:0005856']
    >>> go_mapping = go_to_goslim(from_ids, to_ids)  # doctest: +SKIP
    >>> print(go_mapping)
    # Expected output:
    {'GO:0035253': ['GO:0005929', 'GO:0005856'],
    'GO:0000922': ['GO:0005856']}

    Offline, with a pinned file parsed once:

    >>> from omicspylib import load_obo
    >>> go = load_obo("data/go/go-basic.obo")  # doctest: +SKIP
    >>> go_to_goslim(from_ids, to_ids, relations="is_a,part_of",
    ...              obo=go, minimal=False, replace_obsolete=False)  # doctest: +SKIP
    """
    if obo is not None:
        return _go_to_goslim_offline(
            from_ids, to_ids, relations, obo=obo, minimal=minimal, replace_obsolete=replace_obsolete
        )
    if minimal:
        raise ValueError("minimal=True is only supported offline (pass `obo`); QuickGO has no such option.")
    return _go_to_goslim_online(from_ids, to_ids, relations, batch_size, delay_time)


def _go_to_goslim_online(
    from_ids: list[str], to_ids: list[str], relations: str, batch_size: int, delay_time: float
) -> dict[str, list[str]]:
    url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/slim"
    from_ids = list(set(from_ids))
    to_ids = list(set(to_ids))
    from_ids = [go_id.strip() for go_id in from_ids]
    to_ids = [go_id.strip() for go_id in to_ids]

    n_items = len(from_ids)
    n_steps = len(from_ids) // batch_size + 1

    all_mappings = []
    for i in range(n_steps):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_items)

        params = {
            "slimsToIds": ",".join(to_ids),
            "slimsFromIds": ",".join(from_ids[start_idx:end_idx]),
            "relations": relations,
        }
        response = requests.get(url, params=params, timeout=60)
        time.sleep(delay_time)
        if response.ok:
            batched_mappings = response.json()["results"]
            all_mappings.extend(batched_mappings)
        else:
            response.raise_for_status()

    # reformat outputs
    return {r["slimsFromId"]: r["slimsToIds"] for r in all_mappings}


def _parse_relations(relations: str) -> tuple[list[str], set[str]]:
    """Return the requested relation names and their expanded set."""
    requested = list(dict.fromkeys(r.strip() for r in relations.split(",") if r.strip()))
    if not requested:
        raise ValueError("`relations` must name at least one relation.")
    bad = [r for r in requested if not _RELATION_RE.match(r)]
    if bad:
        raise ValueError(f"Invalid relation name(s): {bad}. Use names such as 'is_a', 'part_of'.")
    expanded = set(requested)
    if "regulates" in expanded:
        expanded.update(_REGULATES)
    return requested, expanded


def _resolve(go: GeneOntology, go_id: str, replace_obsolete: bool) -> str | None:
    """Primary, non-obsolete id for ``go_id``, or None if it cannot be resolved."""
    current = go.primary(go_id)
    seen: set[str] = set()
    while current in go.names:
        if current not in go.obsolete:
            return current
        if not replace_obsolete or current in seen or current not in go.replaced_by:
            return None
        seen.add(current)
        current = go.primary(go.replaced_by[current])
    return None


def _first(ids: list[str], n: int = 5) -> str:
    return ", ".join(ids[:n])


def _go_to_goslim_offline(
    from_ids: list[str],
    to_ids: list[str],
    relations: str,
    *,
    obo: str | os.PathLike[str] | GeneOntology,
    minimal: bool,
    replace_obsolete: bool,
) -> dict[str, list[str]]:
    requested, expanded = _parse_relations(relations)
    go = obo if isinstance(obo, GeneOntology) else load_obo(obo)

    def n_edges(rel: str) -> int:
        return len(go.edges.get(rel, {}))

    empty = [r for r in requested if sum(n_edges(e) for e in (_REGULATES if r == "regulates" else (r,))) == 0]
    if empty:
        warnings.warn(
            f"Requested relation(s) with no edges in this OBO file (data-version {go.data_version}): "
            f"{', '.join(empty)}. They have no effect on the mapping.",
            UserWarning,
            stacklevel=3,
        )

    # slim set
    slim: set[str] = set()
    bad_slims: list[str] = []
    for raw in dict.fromkeys(t.strip() for t in to_ids):
        if not raw:
            continue
        resolved = _resolve(go, raw, replace_obsolete)
        if resolved is None:
            bad_slims.append(raw)
        else:
            slim.add(resolved)
    if bad_slims:
        warnings.warn(
            f"{len(bad_slims)} slim id(s) unknown or obsolete without replacement, ignored "
            f"(first: {_first(bad_slims)}).",
            UserWarning,
            stacklevel=3,
        )

    ancestor_cache: dict[str, set[str]] = {}

    def ancestors(term: str) -> set[str]:
        if term not in ancestor_cache:
            ancestor_cache[term] = go.ancestors(term, expanded)
        return ancestor_cache[term]

    result: dict[str, list[str]] = {}
    unknown: list[str] = []
    for raw in dict.fromkeys(f.strip() for f in from_ids):
        if not raw:
            continue
        if raw not in go:
            unknown.append(raw)
            continue
        resolved = _resolve(go, raw, replace_obsolete)
        if resolved is None:
            continue
        hits = ancestors(resolved) & slim
        if minimal and len(hits) > 1:
            hits = {h for h in hits if not any(h != other and h in ancestors(other) for other in hits)}
        if hits:
            result[raw] = sorted(hits)

    if unknown:
        warnings.warn(
            f"{len(unknown)} GO id(s) not in the ontology, left unmapped (first: {_first(unknown)}).",
            UserWarning,
            stacklevel=3,
        )
    return result
