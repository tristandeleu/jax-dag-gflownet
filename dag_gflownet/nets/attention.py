import jax.numpy as jnp
import haiku as hk

from jax import nn


class LinearMultiHeadAttention(hk.MultiHeadAttention):
    def __call__(self, query, key, value, mask=None):
        feature_map = lambda x: nn.elu(x) + 1.
        eps = 1e-6

        query_heads = self._linear_projection(query, self.key_size, 'query')
        key_heads = self._linear_projection(key, self.key_size, 'key')
        value_heads = self._linear_projection(value, self.value_size, 'value')

        # Map the query & key with a feature map
        query_heads = feature_map(query_heads)
        key_heads = feature_map(key_heads)

        key_values = jnp.einsum('...thd,...thk->...hkd', key_heads, value_heads)
        normalizer = 1. / (jnp.einsum('...thd,...hd->...th',
            query_heads, jnp.sum(key_heads, axis=-3)) + eps)
        attn = jnp.einsum('...thd,...hkd,...th->...thk',
            query_heads, key_values, normalizer)

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)
