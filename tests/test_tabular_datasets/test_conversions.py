from omicspylib import ProteinsDataset


def test_peptides_to_proteins_dataset_conversion(peptides_dataset):
    # action
    proteins_dataset = peptides_dataset.to_proteins()

    a=1
    # assertion
    assert isinstance(proteins_dataset, ProteinsDataset)