import sys

from collections import namedtuple
from abc import ABC, abstractmethod

LocalScore = namedtuple('LocalScore', ['key', 'score', 'prior'])

class BaseScore(ABC):
    """Base class for the scorer.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.column_names = list(data.columns)
        self.num_variables = len(self.column_names)
        self.prior.num_variables = self.num_variables

    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data
                local_score_before, local_score_after = self.get_local_scores(
                    target, indices, indices_after=indices_after)

                out_queue.put((True, *local_score_after))
                if local_score_before is not None:
                    out_queue.put((True, *local_score_before))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None, None))

    @abstractmethod
    def get_local_scores(self, target, indices, indices_after=None):
        pass


class BasePrior(ABC):
    """Base class for the prior over graphs p(G).
    
    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.
    
    Parameters
    ----------
    num_variables : int (optional)
        The number of variables in the graph. If not specified, this gets
        populated inside the scorer class.
    """
    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError('The number of variables is not defined.')
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value
