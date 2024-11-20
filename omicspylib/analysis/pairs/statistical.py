from typing import Optional

from omicspylib import ProteinsDataset
from omicspylib.calculations.fold_change import calc_fold_change
from omicspylib.calculations.ttest import calc_ttest_adj, MULTITEST_METHOD


class PairwiseComparisonTTestFC:
    """
    Perform pairwise comparison between two groups,
    using a t-test and a fold change rule.
    """
    def __init__(self,
                 dataset: ProteinsDataset,
                 condition_a: str,
                 condition_b: str):
        self._raw_dataset = dataset
        self._condition_a = condition_a
        self._condition_b = condition_b

    def eval(self,
             min_frequency: int = 3,
             na_threshold: float = 0.0,
             pval_adj_method: Optional[MULTITEST_METHOD] = 'fdr_bh',
             use_log_transformed: bool = True):
        """
        Perform the pairwise comparison between the two groups, using
        a t-test and a fold change rule. By default, quantitative values
        are log2 transformed, prior to t-test calculation. For the fold
        change calculation, the original values are used.

        Parameters
        ----------
        min_frequency: int, optional
            Records identified in less than this number of biological
            repeats, in a given experimental condition, will be excluded
            from the analysis.
        na_threshold: float, optional
            Values equal or below this threshold are considered missing.
        pval_adj_method: str or None, optional
            Method to adjust p-values for multiple hypothesis testing error.
            If not provided, no adjustment will be performed.
        use_log_transformed: bool, optional
            By default, quantitative values are log2 transformed prior to
            t-test. If set to ``False`` this transformation will be omitted.

        Returns
        -------
        pd.DataFrame
            A Pandas data frame with the results of the t-test and fold
            change calculations. Use the data frame index to join
            back the results with the dataset.
        """
        dataset = self._raw_dataset.filter(
            cond=[self._condition_a, self._condition_b],
            min_frequency=min_frequency,
            na_threshold=na_threshold)

        mean_abundance = dataset.mean(na_threshold=na_threshold, join_method='inner')
        mean_abundance = mean_abundance.rename(columns={
            f'mean_{self._condition_a}': self._condition_a,
            f'mean_{self._condition_b}': self._condition_b,
        })

        fc_out = calc_fold_change(
            data=mean_abundance,
            condition_a=self._condition_a,
            condition_b=self._condition_b)

        if use_log_transformed:
            ttest_input_dataset = dataset.log2_transform()
        else:
            ttest_input_dataset = dataset

        ttest_out = calc_ttest_adj(
            data=ttest_input_dataset,
            condition_a=self._condition_a,
            condition_b=self._condition_b,
            na_threshold=na_threshold,
            pval_adj_method=pval_adj_method)

        out_df = ttest_out.merge(fc_out, left_index=True, right_index=True, how='outer')

        return out_df
