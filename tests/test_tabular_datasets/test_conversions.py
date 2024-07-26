from omicspylib import ProteinsDataset


def test_peptides_to_proteins_dataset_conversion(peptides_dataset):
    """
    Validate that from a PeptidesDataset you can jump to
    a ProteinsDataset.
    """
    # action
    proteins_dataset = peptides_dataset.to_proteins()

    # assertion
    assert isinstance(proteins_dataset, ProteinsDataset)