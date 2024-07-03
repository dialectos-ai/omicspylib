import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from omicspylib import PeptidesDataset
from omicspylib.plots import plot_density


def test_normalization_method():
    # setup
    data_df = pd.read_csv('tests/data/peptides_dataset.tsv', sep='\t')
    config = {
        'id_col': 'peptide_id',
        'conditions': {
            'c1': ['c1_rep1', 'c1_rep2', 'c1_rep3', 'c1_rep4', 'c1_rep5'],
            'c2': ['c2_rep1', 'c2_rep2', 'c2_rep3', 'c2_rep4', 'c2_rep5'],
            'c3': ['c3_rep1', 'c3_rep2', 'c3_rep3', 'c3_rep4', 'c3_rep5'],
        },
        'protein_id_col': 'protein_id',
    }
    float_columns = data_df.select_dtypes(include=['float64']).columns
    shift = [i/4 for i in range(len(float_columns))]
    mask = data_df[float_columns] > 0
    data_df[float_columns] += shift * mask
    dataset = PeptidesDataset.from_df(data_df, **config)

    col_means_before = dataset.mean(axis=0)

    # action
    norm_dataset = dataset.normalize(method='mean')
    col_means_after = norm_dataset.mean(axis=0)

    # assertion
    global_mean_before = np.mean(col_means_before)
    assert not np.any(np.isclose(global_mean_before, col_means_before, atol=0.1))
    global_mean_after = np.mean(col_means_after)
    assert np.all(np.isclose(global_mean_after, col_means_after, atol=0.1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_density(dataset, ax=ax[0])
    plot_density(norm_dataset, ax=ax[1])
    plt.show()
