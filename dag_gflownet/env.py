import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gym.spaces import Dict, Box, Discrete

from dag_gflownet.utils.cache import LRUCache


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(
            self,
            num_envs,
            scorer,
            max_parents=None,
            num_workers=4,
            context=None,
            cache_max_size=10_000
        ):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments, or equivalently the number of
            parallel trajectories to sample.
        
        scorer : BaseScore instance
            The score to use. Note that this contains the data.

        max_parents : int, optional
            Maximum number of parents for each node in the DAG. If None, then
            there is no constraint on the maximum number of parents.

        num_workers : int (default: 4)
            The number of multiprocessing workers to compute the local scores.
            Use `num_workers=0` to disable multiprocessing.

        context : str, optional
            The multiprocessing context.

        cache_max_size : int (default: 10_000)
            The maximum size of the LRU cache for the local scores.
        """
        self.scorer = scorer
        self.num_workers = num_workers

        self.num_variables = scorer.num_variables
        self.local_scores = LRUCache(max_size=cache_max_size)
        self._state = None
        self.max_parents = max_parents or self.num_variables

        if num_workers > 0:
            ctx = get_context(context)

            self.in_queue = ctx.Queue()
            self.out_queue = ctx.Queue()
            self.error_queue = ctx.Queue()

            self.processes = []
            for index in range(num_workers):
                process = ctx.Process(
                    target=self.scorer,
                    args=(index, self.in_queue, self.out_queue, self.error_queue),
                    daemon=True
                )
                process.start()

        shape = (self.num_variables, self.num_variables)
        max_edges = self.num_variables * (self.num_variables - 1) // 2
        observation_space = Dict({
            'adjacency': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'mask': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'num_edges': Discrete(max_edges),
            'score': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float_),
            'order': Box(low=-1, high=max_edges, shape=shape, dtype=np.int_)
        })
        action_space = Discrete(self.num_variables ** 2 + 1)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self):
        shape = (self.num_envs, self.num_variables, self.num_variables)
        closure_T = np.eye(self.num_variables, dtype=np.bool_)
        self._closure_T = np.tile(closure_T, (self.num_envs, 1, 1))
        self._state = {
            'adjacency': np.zeros(shape, dtype=np.int_),
            'mask': 1 - self._closure_T,
            'num_edges': np.zeros((self.num_envs,), dtype=np.int_),
            'score': np.zeros((self.num_envs,), dtype=np.float_),
            'order': np.full(shape, -1, dtype=np.int_)
        }
        return deepcopy(self._state)

    def step(self, actions):
        sources, targets = divmod(actions, self.num_variables)
        keys, local_cache, data = self.local_scores_async(sources, targets)
        dones = (sources == self.num_variables)
        sources, targets = sources[~dones], targets[~dones]

        # Make sure that all the actions are valid
        if not np.all(self._state['mask'][~dones, sources, targets]):
            raise ValueError('Some actions are invalid: either the edge to be '
                             'added is already in the DAG, or adding this edge '
                             'would lead to a cycle.')

        # Update the adjacency matrices
        self._state['adjacency'][~dones, sources, targets] = 1
        self._state['adjacency'][dones] = 0

        # Update transitive closure of transpose
        source_rows = np.expand_dims(self._closure_T[~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= np.logical_and(source_rows, target_cols)  # Outer product
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Update the masks
        self._state['mask'] = 1 - (self._state['adjacency'] + self._closure_T)

        # Update the masks (maximum number of parents)
        num_parents = np.sum(self._state['adjacency'], axis=1, keepdims=True)
        self._state['mask'] *= (num_parents < self.max_parents)

        # Update the order
        self._state['order'][~dones, sources, targets] = self._state['num_edges'][~dones]
        self._state['order'][dones] = -1

        # Update the number of edges
        self._state['num_edges'] += 1
        self._state['num_edges'][dones] = 0

        # Get the difference of log-rewards. The environment returns the
        # delta-scores log R(G_t) - log R(G_{t-1}), corresponding to a local
        # change in the scores. This quantity can be used directly in the loss
        # function derived from the trajectory detailed loss.
        delta_scores = self.local_scores_wait(keys, local_cache, data)

        # Update the scores. The scores returned by the environments are scores
        # relative to the empty graph: score(G) - score(G_0).
        self._state['score'] += delta_scores
        self._state['score'][dones] = 0

        return (deepcopy(self._state), delta_scores, dones, {})

    def local_scores_async(self, sources, targets):
        keys, local_cache, queued_data = [], set(), []
        for i, (source, target) in enumerate(zip(sources, targets)):
            if source == self.num_variables:
                key = (None, None, None)

            else:
                adjacency = self._state['adjacency'][i]

                # Key before adding the new source node
                indices = tuple(index for index, is_parent
                    in enumerate(adjacency[:, target]) if is_parent)

                # Key after adding the new source node
                indices_after = list(indices)
                bisect.insort(indices_after, source)
                indices_after = tuple(indices_after)

                if not self._is_in_cache((target, indices_after), local_cache):
                    if not self._is_in_cache((target, indices), local_cache):
                        data = (indices, indices_after)
                        local_cache.update({
                            (target, indices),
                            (target, indices_after)
                        })
                    else:
                        data = (indices_after, None)
                        local_cache.add((target, indices_after))
                elif not self._is_in_cache((target, indices), local_cache):
                    data = (indices, None)
                    local_cache.add((target, indices))
                else:
                    data = None

                if data is not None:
                    queued_data.append((target,) + data)
                    if self.num_workers > 0:
                        self.in_queue.put((target,) + data)

                key = (target, indices, indices_after)

            keys.append(key)
        return keys, local_cache, queued_data

    def local_scores_wait(self, keys, local_cache, data):
        # Collect the values returned by the workers
        if self.num_workers > 0:
            for _ in local_cache:
                is_success, key, value, prior = self.out_queue.get()

                if is_success:
                    self.local_scores[key] = value + prior
                else:
                    _, exctype, value = self.error_queue.get()
                    raise exctype(value)
        else:
            for target, indices, indices_after in data:
                local_score_before, local_score_after = self.scorer.get_local_scores(
                    target, indices, indices_after=indices_after)

                self.local_scores[local_score_after.key] = (
                    local_score_after.score + local_score_after.prior)
                if local_score_before is not None:
                    self.local_scores[local_score_before.key] = (
                        local_score_before.score + local_score_before.prior)

        delta_scores = []
        for target, key_tm1, key_t in keys:
            if target is None:
                delta_scores.append(0.)

            else:
                delta_scores.append(
                    self.local_scores[(target, key_t)]
                    - self.local_scores[(target, key_tm1)]
                )

        return np.array(delta_scores, dtype=np.float_)

    def _is_in_cache(self, key, local_cache):
        if key in self.local_scores:
            _ = self.local_scores[key]  # Refresh LRU cache
            return True
        else:
            return (key in local_cache)

    def close_extras(self, **kwargs):
        if self.num_workers > 0:
            for _ in range(self.num_workers):
                self.in_queue.put(None)

            for process in self.processes:
                process.join()
