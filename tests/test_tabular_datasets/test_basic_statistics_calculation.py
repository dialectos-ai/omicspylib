def test_mean_calculation_on_proteins_dataset(proteins_dataset, peptides_dataset):
    """
    Given a dataset, calculate the average quantitative value for each condition.
    Output should be a data frame with each condition's name as column
    """
    # setup
    exp_cols = ['c1', 'c2', 'c3']
    datasets = [proteins_dataset, peptides_dataset]

    # action
    for dataset in datasets:
        result = dataset.mean()

        # assertion
        for exp_col in exp_cols:
            assert f'mean_{exp_col}' in result.columns
        assert result.shape[0] == dataset.n_records
        assert result.shape[1] == dataset.n_conditions


def test_frequency_calculation_on_tabular_datasets(proteins_dataset, peptides_dataset):
    """
    Given a dataset, calculate the frequency of experiments with value
    above the given threshold for each condition.
    Output should be a data frame with each condition's name as column
    """
    # setup
    exp_cols = ['c1', 'c2', 'c3']
    datasets = [proteins_dataset, peptides_dataset]

    # action
    for dataset in datasets:
        result = dataset.frequency()

        # assertion
        for exp_col in exp_cols:
            assert f'frequency_{exp_col}' in result.columns
        assert result.shape[0] == dataset.n_records
        assert result.shape[1] == dataset.n_conditions
