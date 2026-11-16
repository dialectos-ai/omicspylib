"""Offline GO -> GO slim mapping against a hand-written synthetic OBO file."""

import gzip
import shutil
import socket
import warnings
from pathlib import Path
from typing import Any

import pytest
import requests

from omicspylib import GeneOntology, go_to_goslim, load_obo

MINI_OBO = Path("tests/data/mini_go.obo")

ROOT = "GO:9999001"
SLIM_BP = "GO:9999002"
CHILD = "GO:9999003"
GRANDCHILD = "GO:9999004"
PART_OF = "GO:9999005"
REG_TARGET = "GO:9999006"
REGULATES = "GO:9999007"
POS_REGULATES = "GO:9999008"
SLIM_CC = "GO:9999011"
SLIM_CC_ALT = "GO:9999091"
OCCURS_IN = "GO:9999012"
HAS_PART = "GO:9999013"
WITH_ALT = "GO:9999014"
WITH_ALT_ALT = "GO:9999094"
OBS_REPLACED = "GO:9999020"
OBS_CONSIDER = "GO:9999021"
OBS_CHAIN = "GO:9999022"
OBS_CYCLE = "GO:9999023"
OBS_SLIM = "GO:9999025"
OBS_AMBIGUOUS = "GO:9999026"

SLIMS = [SLIM_BP, REG_TARGET, SLIM_CC]
ALL_RELATIONS = "is_a,part_of,occurs_in,regulates"


def _load_quiet(path: Path | str) -> GeneOntology:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_obo(path)


@pytest.fixture(scope="module")
def mini_go() -> GeneOntology:
    return _load_quiet(MINI_OBO)


@pytest.fixture
def mini_go_gz(tmp_path: Path) -> Path:
    target = tmp_path / "mini_go.obo.gz"
    with MINI_OBO.open("rb") as src, gzip.open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def _map(
    mini_go: GeneOntology, from_ids: list[str], to_ids: list[str] | None = None, **kwargs: Any
) -> dict[str, list[str]]:
    """Offline mapping that fails on any warning."""
    kwargs.setdefault("relations", ALL_RELATIONS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return go_to_goslim(from_ids, SLIMS if to_ids is None else to_ids, obo=mini_go, **kwargs)


# --- parsing -----------------------------------------------------------------


def test_header_parsed(mini_go: GeneOntology) -> None:
    assert mini_go.data_version == "releases/2099-01-01"
    assert mini_go.format_version == "1.2"


def test_missing_header_is_none(tmp_path: Path) -> None:
    path = tmp_path / "no_header.obo"
    path.write_text("[Term]\nid: GO:9999001\nname: root\n", encoding="utf-8")
    go = load_obo(path)
    assert go.data_version is None
    assert go.format_version is None
    assert go.names == {"GO:9999001": "root"}


def test_terms_alt_ids_obsolete(mini_go: GeneOntology) -> None:
    assert mini_go.names[CHILD] == "child process"
    assert mini_go.namespaces[SLIM_CC] == "cellular_component"
    assert mini_go.alt_ids == {SLIM_CC_ALT: SLIM_CC, WITH_ALT_ALT: WITH_ALT}
    assert OBS_CONSIDER in mini_go.obsolete
    assert mini_go.replaced_by[OBS_REPLACED] == CHILD
    assert "part_of" not in mini_go.names  # [Typedef] skipped
    assert mini_go.edges["positively_regulates"][POS_REGULATES] == frozenset({REG_TARGET})
    assert mini_go.edges["has_part"][HAS_PART] == frozenset({SLIM_CC})


def test_ambiguous_replaced_by_warns_and_is_dropped() -> None:
    with pytest.warns(UserWarning, match="more than one replaced_by"):
        go = load_obo(MINI_OBO)
    assert OBS_AMBIGUOUS in go.obsolete
    assert OBS_AMBIGUOUS not in go.replaced_by


def test_ancestors_include_self(mini_go: GeneOntology) -> None:
    assert mini_go.ancestors(GRANDCHILD, ["is_a"]) == {GRANDCHILD, CHILD, SLIM_BP, ROOT}
    assert mini_go.ancestors("GO:0000000", ["is_a"]) == set()


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_obo("tests/data/does_not_exist.obo")
    with pytest.raises(FileNotFoundError):
        go_to_goslim([CHILD], SLIMS, obo="tests/data/does_not_exist.obo")


def test_file_without_terms(tmp_path: Path) -> None:
    path = tmp_path / "empty.obo"
    path.write_text("format-version: 1.2\n\n[Typedef]\nid: part_of\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No GO"):
        load_obo(path)


# --- relations ----------------------------------------------------------------


def test_is_a_chain(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [GRANDCHILD], relations="is_a") == {GRANDCHILD: [SLIM_BP]}


def test_slim_term_maps_to_itself(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [SLIM_BP]) == {SLIM_BP: [SLIM_BP]}


def test_part_of(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [PART_OF], relations="is_a,part_of") == {PART_OF: [SLIM_BP]}
    assert _map(mini_go, [PART_OF], relations="is_a,occurs_in") == {}


def test_regulates_expands_to_sub_relations(mini_go: GeneOntology) -> None:
    result = _map(mini_go, [REGULATES, POS_REGULATES], relations="is_a,regulates")
    assert result == {REGULATES: [REG_TARGET], POS_REGULATES: [REG_TARGET]}


def test_positively_regulates_alone(mini_go: GeneOntology) -> None:
    result = _map(mini_go, [REGULATES, POS_REGULATES], relations="is_a,positively_regulates")
    assert result == {POS_REGULATES: [REG_TARGET]}


def test_occurs_in(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [OCCURS_IN]) == {OCCURS_IN: [SLIM_CC]}


def test_has_part_not_followed_by_default(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [HAS_PART]) == {HAS_PART: [SLIM_BP]}
    assert _map(mini_go, [HAS_PART], relations="is_a,has_part") == {HAS_PART: [SLIM_BP, SLIM_CC]}


def test_relation_without_edges_warns(mini_go: GeneOntology) -> None:
    with pytest.warns(UserWarning, match=r"no edges.*releases/2099-01-01.*negatively_regulates"):
        go_to_goslim([CHILD], SLIMS, relations="is_a,negatively_regulates", obo=mini_go)


@pytest.mark.parametrize("relations", ["", " , ", "is_a,part-of", "is_a,Part_of", "is_a;part_of"])
def test_bad_relations(mini_go: GeneOntology, relations: str) -> None:
    with pytest.raises(ValueError, match="relation"):
        go_to_goslim([CHILD], SLIMS, relations=relations, obo=mini_go)


# --- alt ids and obsolete terms -------------------------------------------------


def test_alt_id_in_from_ids_keeps_input_key(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [WITH_ALT_ALT]) == {WITH_ALT_ALT: [SLIM_BP]}


def test_alt_id_in_to_ids(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [OCCURS_IN], to_ids=[SLIM_CC_ALT]) == {OCCURS_IN: [SLIM_CC]}


def test_obsolete_replaced_by(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [OBS_REPLACED], replace_obsolete=True) == {OBS_REPLACED: [SLIM_BP]}
    assert _map(mini_go, [OBS_REPLACED], replace_obsolete=False) == {}


def test_obsolete_replaced_by_chain(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [OBS_CHAIN]) == {OBS_CHAIN: [SLIM_BP]}


def test_obsolete_consider_only_cycle_and_ambiguous_unmapped(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [OBS_CONSIDER, OBS_CYCLE, OBS_AMBIGUOUS]) == {}


def test_obsolete_slim_in_to_ids(mini_go: GeneOntology) -> None:
    assert _map(mini_go, [CHILD], to_ids=[OBS_SLIM], replace_obsolete=True) == {CHILD: [SLIM_BP]}
    with pytest.warns(UserWarning, match="slim id"):
        result = go_to_goslim([CHILD], [OBS_SLIM, "GO:0000000"], obo=mini_go, replace_obsolete=False)
    assert result == {}


def test_unknown_id_warns_once(mini_go: GeneOntology) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = go_to_goslim([CHILD, "GO:0000001", "GO:0000002"], SLIMS, relations="is_a", obo=mini_go)
    assert result == {CHILD: [SLIM_BP]}
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert "2 GO id(s) not in the ontology" in messages[0]
    assert "GO:0000001" in messages[0]


# --- minimal and output shape ----------------------------------------------------


def test_minimal(mini_go: GeneOntology) -> None:
    to_ids = [ROOT, SLIM_BP]
    assert _map(mini_go, [CHILD], to_ids=to_ids, minimal=False) == {CHILD: [ROOT, SLIM_BP]}
    assert _map(mini_go, [CHILD], to_ids=to_ids, minimal=True) == {CHILD: [SLIM_BP]}


def test_minimal_online_raises() -> None:
    with pytest.raises(ValueError, match="minimal"):
        go_to_goslim([CHILD], SLIMS, minimal=True)


def test_whitespace_duplicates_and_sorted(mini_go: GeneOntology) -> None:
    result = _map(mini_go, [" " + HAS_PART, HAS_PART + "\n", HAS_PART, ""], relations="is_a,has_part")
    assert result == {HAS_PART: [SLIM_BP, SLIM_CC]}


def test_input_forms_identical(mini_go: GeneOntology, mini_go_gz: Path) -> None:
    from_ids = [GRANDCHILD, PART_OF, REGULATES, OCCURS_IN, WITH_ALT_ALT, OBS_CHAIN]
    expected = go_to_goslim(from_ids, SLIMS, obo=mini_go)
    assert len(expected) == len(from_ids)
    for obo in (str(MINI_OBO), MINI_OBO, mini_go_gz, str(mini_go_gz)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert go_to_goslim(from_ids, SLIMS, obo=obo) == expected
    assert _load_quiet(mini_go_gz) == mini_go


def test_no_network(monkeypatch: pytest.MonkeyPatch, mini_go: GeneOntology) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(socket, "socket", blocked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert go_to_goslim([CHILD], SLIMS, obo=MINI_OBO) == {CHILD: [SLIM_BP]}
    assert go_to_goslim([CHILD], SLIMS, obo=mini_go) == {CHILD: [SLIM_BP]}
