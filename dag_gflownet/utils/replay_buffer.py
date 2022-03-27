import numpy as np
import math

from numpy.random import default_rng


class ReplayBuffer:
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables

        nbytes = math.ceil((num_variables ** 2) / 8)
        dtype = np.dtype([
            ('adjacency', np.uint8, (nbytes,)),
            ('num_edges', np.int_, (1,)),
            ('actions', np.int_, (1,)),
            ('is_exploration', np.bool_, (1,)),
            ('delta_scores', np.float_, (1,)),
            ('scores', np.float_, (1,)),
            ('mask', np.uint8, (nbytes,)),
            ('next_adjacency', np.uint8, (nbytes,)),
            ('next_mask', np.uint8, (nbytes,))
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False
        self._prev = np.full((capacity,), -1, dtype=np.int_)

    def add(
            self,
            observations,
            actions,
            is_exploration,
            next_observations,
            delta_scores,
            dones,
            prev_indices=None
        ):
        indices = np.full((dones.shape[0],), -1, dtype=np.int_)
        if np.all(dones):
            return indices

        num_samples = np.sum(~dones)
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity
        indices[~dones] = add_idx

        data = {
            'adjacency': self.encode(observations['adjacency'][~dones]),
            'num_edges': observations['num_edges'][~dones],
            'actions': actions[~dones],
            'delta_scores': delta_scores[~dones],
            'mask': self.encode(observations['mask'][~dones]),
            'next_adjacency': self.encode(next_observations['adjacency'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones]),

            # Extra keys for monitoring
            'is_exploration': is_exploration[~dones],
            'scores': observations['score'][~dones],
        }

        for name in data:
            shape = self._replay.dtype[name].shape
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))
        
        if prev_indices is not None:
            self._prev[add_idx] = prev_indices[~dones]

        return indices

    def sample(self, batch_size, rng=default_rng()):
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        # Convert structured array into dictionary
        return {
            'adjacency': self.decode(samples['adjacency']),
            'num_edges': samples['num_edges'],
            'actions': samples['actions'],
            'delta_scores': samples['delta_scores'],
            'mask': self.decode(samples['mask']),
            'next_adjacency': self.decode(samples['next_adjacency']),
            'next_mask': self.decode(samples['next_mask'])
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index

    @property
    def transitions(self):
        return self._replay[:len(self)]

    def save(self, filename):
        data = {
            'version': 2,
            'replay': self.transitions,
            'index': self._index,
            'is_full': self._is_full,
            'prev': self._prev,
            'capacity': self.capacity,
            'num_variables': self.num_variables,
        }
        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            if data['version'] != 2:
                raise IOError(f'Unknown version: {data["version"]}')
            replay = cls(
                capacity=data['capacity'],
                num_variables=data['num_variables']
            )
            replay._index = data['index']
            replay._is_full = data['is_full']
            replay._prev = data['prev']
            replay._replay[:len(replay)] = data['replay']
        return replay

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables ** 2)
        return np.packbits(encoded, axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.num_variables, self.num_variables)
        return decoded.astype(dtype)

    @property
    def dummy(self):
        shape = (self.num_variables, self.num_variables)
        return {
            'adjacency': np.zeros(shape, dtype=np.float32),
            'num_edges': np.zeros((1,), dtype=np.int_),
            'actions': np.zeros((1,), dtype=np.int_),
            'delta_scores': np.zeros((1,), dtype=np.float_),
            'mask': np.zeros(shape, dtype=np.float32),
            'next_adjacency': np.zeros(shape, dtype=np.float32),
            'next_mask': np.zeros(shape, dtype=np.float32)
        }
