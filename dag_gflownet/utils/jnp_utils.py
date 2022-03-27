import jax.numpy as jnp

from jax import random


def batch_random_choice(key, probas, masks):
    # Sample from the distribution
    uniform = random.uniform(key, shape=(probas.shape[0], 1))
    cum_probas = jnp.cumsum(probas, axis=1)
    samples = jnp.sum(cum_probas < uniform, axis=1, keepdims=True)

    # In rare cases, the sampled actions may be invalid, despite having
    # probability 0. In those cases, we select the stop action by default.
    stop_mask = jnp.ones((masks.shape[0], 1), dtype=masks.dtype)  # Stop action is always valid
    masks = masks.reshape(masks.shape[0], -1)
    masks = jnp.concatenate((masks, stop_mask), axis=1)

    is_valid = jnp.take_along_axis(masks, samples, axis=1)
    stop_action = masks.shape[1]
    samples = jnp.where(is_valid, samples, stop_action)

    return jnp.squeeze(samples, axis=1)
