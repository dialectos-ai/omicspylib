from omicspylib import ProteinsDataset
import numpy as np


def test_peptides_to_proteins_dataset_conversion(peptides_dataset):
    """
    Validate that from a PeptidesDataset you can jump to
    a ProteinsDataset.
    """
    # action
    proteins_dataset = peptides_dataset.to_proteins()

    # assertion
    assert isinstance(proteins_dataset, ProteinsDataset)
    row = proteins_dataset.to_table().iloc[0].values
    assert np.all(row > 40)


def test_calc_peptide_counts(peptides_dataset):
    """
    When converting peptides to proteins,
    you should have the option to calculate also peptide counts,
    in addition to total protein intensity. Test that.
    """
    # action
    pept_counts = peptides_dataset.to_proteins(agg_method='counts')

    # assertion
    data_df = pept_counts.to_table()
    raw_row_vals = data_df.iloc[0].values
    int_row_vals = raw_row_vals.astype(int)
    assert np.all(np.isclose(raw_row_vals - int_row_vals, 0, atol=0.001))


def test_column_renaming_while_converting_peptides_to_proteins(peptides_dataset):
    """
    You can convert a peptides' dataset to a proteins' dataset. However, the type
    of the values might change (e.g., you calculate peptide counts from a peptide
    dataset of normalized intensities). For this, you should be able to
    rename the columns after the conversion, avoid naming conflicts during
    dataset joining later on.
    """
    # setup
    original_cols = peptides_dataset.to_table().columns.tolist()
    prefix = 'Counts: '
    new_names = {c: f'{prefix}{c}' for c in original_cols}

    # action
    pept_counts = peptides_dataset.to_proteins(agg_method='counts', names_lookup=new_names)

    # assertion
    data_df = pept_counts.to_table()
    for col in data_df.columns:
        assert col.startswith(prefix)
