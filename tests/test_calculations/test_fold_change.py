import pandas as pd

from omicspylib.calculations import calc_fold_change


def test_fold_change_calculation():
    """
    Given a data frame and two column names, calculate the fold
    change and return the expected output.
    """
    # setup
    data = {'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10]}
    df = pd.DataFrame(data)
    column1 = 'A'
    column2 = 'B'
    exp_fc_col = f'fold change {column1} over {column2}'
    exp_log2_fc_col = f'log2 fold change {column1} over {column2}'

    # action
    fc_out = calc_fold_change(df, column1, column2)

    # assertion
    assert exp_fc_col in fc_out.columns
    assert exp_log2_fc_col in fc_out.columns
    assert fc_out.shape[0] == df.shape[0]
