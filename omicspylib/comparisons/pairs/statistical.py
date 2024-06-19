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

    def compare(self,
                min_frequency: int = 3,
                na_threshold: float = 0.0,
                pval_adj_method: Optional[MULTITEST_METHOD] = 'fdr_bh'):
        """

        Parameters
        ----------
        min_frequency: int, optional
            Minimum number of biological repeats, a given property is
            quantified, within an experimental condition. Records with lower
            frequency are excluded from the analysis.
        na_threshold: float, optional
            Values equal or below this threshold are considered missing.
        pval_adj_method: str or None, optional
            Method to adjust p-values for multiple hypothesis testing error.
            If not provided, no adjustment will be performed.

        Returns
        -------
        pd.DataFrame
            A pandas data frame with the measured values and the results of
            the t-test and fold change calculations.
        """
        dataset = self._raw_dataset.filter(
            conditions=[self._condition_a, self._condition_b],
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

        log2_dset = dataset.log2_transform()

        ttest_out = calc_ttest_adj(
            data=log2_dset,
            condition_a=self._condition_a,
            condition_b=self._condition_b,
            na_threshold=na_threshold,
            pval_adj_method=pval_adj_method)

        df = dataset.to_table(join_method='outer')
        df = df.merge(ttest_out, left_index=True, right_index=True, how='left')
        df = df.merge(fc_out, left_index=True, right_index=True, how='left')

        return df
