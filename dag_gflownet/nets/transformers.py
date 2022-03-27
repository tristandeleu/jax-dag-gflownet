import jax.numpy as jnp
import haiku as hk

from jax import nn

from dag_gflownet.nets.attention import LinearMultiHeadAttention


class DenseBlock(hk.Module):
    def __init__(self, output_size, init_scale, widening_factor=4, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.init_scale = init_scale
        self.widening_factor = widening_factor

    def __call__(self, inputs):
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        hiddens = hk.Linear(
            self.widening_factor * self.output_size,
            w_init=w_init
        )(inputs)
        hiddens = nn.gelu(hiddens)
        return hk.Linear(self.output_size, w_init=w_init)(hiddens)


class TransformerBlock(hk.Module):
    def __init__(
            self,
            num_heads,
            key_size,
            embedding_size,
            init_scale,
            widening_factor=4,
            name=None
        ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.embedding_size = embedding_size
        self.init_scale = init_scale
        self.widening_factor = widening_factor

    def __call__(self, hiddens, inputs):
        w_init = hk.initializers.VarianceScaling(self.init_scale)

        inputs_embedding = hk.Linear(
            self.embedding_size,
            w_init=w_init,
            name='linear_1'
        )(inputs)
        h_norm = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name='layernorm_1'
        )(jnp.concatenate((inputs_embedding, hiddens), axis=-1))
        h_attn = LinearMultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=self.init_scale
        )(h_norm, h_norm, h_norm, mask=None)
        hiddens = hiddens + h_attn

        inputs_embedding = hk.Linear(
            self.embedding_size,
            w_init=w_init,
            name='linear_2'
        )(inputs)
        h_norm = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name='layernorm_2'
        )(jnp.concatenate((inputs_embedding, hiddens), axis=-1))
        h_dense = DenseBlock(
            init_scale=self.init_scale,
            widening_factor=self.widening_factor,
            output_size=self.num_heads * self.key_size
        )(h_norm)
        hiddens = hiddens + h_dense

        return hiddens
