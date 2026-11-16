"""Gene Ontology utilities: GO -> GO slim mapping, online or from a local OBO file."""

from omicspylib.go.goslim import go_to_goslim
from omicspylib.go.obo import GeneOntology, load_obo

__all__ = ["GeneOntology", "go_to_goslim", "load_obo"]
