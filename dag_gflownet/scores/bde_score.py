import math
import numpy as np
import pandas as pd

from scipy.special import gammaln
from collections import namedtuple

from dag_gflownet.scores.base import BaseScore, LocalScore


StateCounts = namedtuple('StateCounts', ['key', 'counts'])


class BDeScore(BaseScore):
    """BDe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (discrete) dataset D. Each column
        corresponds to one variable. If there is interventional data, the
        interventional targets must be specified in the "INT" column (the
        indices of interventional targets are assumed to be 1-based).
    
    prior : `BasePrior` instance
        The prior over graphs p(G).

    equivalent_sample_size : float (default: 1.)
        The equivalent sample size (of uniform pseudo samples) for the
        Dirichlet hyperparameters. The score is sensitive to this value,
        runs with different values might be useful.
    """
    def __init__(self, data, prior, equivalent_sample_size=1.):
        if 'INT' in data.columns:  # Interventional data
            # Indices should start at 0, instead of 1;
            # observational data will have INT == -1.
            self._interventions = data.INT.map(lambda x: int(x) - 1)
            data = data.drop(['INT'], axis=1)
        else:
            self._interventions = np.full(data.shape[0], -1)
        self.equivalent_sample_size = equivalent_sample_size
        super().__init__(data, prior)

        self.state_names = {
            column: sorted(self.data[column].cat.categories.tolist())
            for column in self.data.columns
        }

    def get_local_scores(self, target, indices, indices_after=None):
        # Get all the state counts
        state_counts_before, state_counts_after = self.state_counts(
            target, indices, indices_after=indices_after)

        local_score_after = self.local_score(*state_counts_after)
        if state_counts_before is not None:
            local_score_before = self.local_score(*state_counts_before)
        else:
            local_score_before = None

        return (local_score_before, local_score_after)

    def state_counts(self, target, indices, indices_after=None):
        # Source: pgmpy.estimators.BaseEstimator.state_counts()
        all_indices = indices if (indices_after is None) else indices_after
        parents = [self.column_names[index] for index in all_indices]
        variable = self.column_names[target]

        data = self.data[self._interventions != target]
        data = data[[variable] + parents].dropna()

        state_count_data = (data.groupby([variable] + parents)
                                .size()
                                .unstack(parents))

        if not isinstance(state_count_data.columns, pd.MultiIndex):
            state_count_data.columns = pd.MultiIndex.from_arrays(
                [state_count_data.columns]
            )

        parent_states = [self.state_names[parent] for parent in parents]
        columns_index = pd.MultiIndex.from_product(parent_states, names=parents)

        state_counts_after = StateCounts(
            key=(target, tuple(all_indices)),
            counts=(state_count_data
                .reindex(index=self.state_names[variable], columns=columns_index)
                .fillna(0))
        )

        if indices_after is not None:
            subset_parents = [self.column_names[index] for index in indices]
            if subset_parents:
                data = (state_counts_after.counts
                    .groupby(axis=1, level=subset_parents)
                    .sum())
            else:
                data = state_counts_after.counts.sum(axis=1).to_frame()

            state_counts_before = StateCounts(
                key=(target, tuple(indices)),
                counts=data
            )
        else:
            state_counts_before = None

        return (state_counts_before, state_counts_after)

    def local_score(self, key, counts):
        counts = np.asarray(counts)
        num_parents_states = counts.shape[1]
        num_parents = len(key[1])

        log_gamma_counts = np.zeros_like(counts, dtype=np.float_)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts.size

        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=np.float_)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        local_score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * math.lgamma(alpha)
            - counts.size * math.lgamma(beta)
        )

        return LocalScore(
            key=key,
            score=local_score,
            prior=self.prior(num_parents)
        )
