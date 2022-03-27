import jax.numpy as jnp
import haiku as hk

from dag_gflownet.nets.transformers import TransformerBlock
from dag_gflownet.utils.gflownet import log_policy


def gflownet(adjacency, mask):
    """GFlowNet used in DAG-GFlowNet.

    This GFlowNet uses a neural network architecture based on Linear
    Transformers. It is composed of a common backbone of 3 Linear Transformers
    layers, followed by two heads: one to compute the probability to stop the
    sampling process, and another to compute the logits of transitioning to a
    new graph, given that we didn't stop. Each head is composed of an additional
    2 Linear Transformers layers, followed by a 3-layer MLP.

    Note that each Linear Transformers layer takes an embedding obtained at the
    previous layer of the network, as well as an embedding of the input adjacency
    matrix (with a different embedding at each layer). This ensures that the
    information about which edges are present in the graph is propagated as much
    as possible.

    The GFlowNet takes as an input a *single* adjacency matrix; this model is
    later vmapped inside the `DAGGFlowNet` class.

    Parameters
    ----------
    adjacency : jnp.DeviceArray
        The adjacency matrix of a graph G. This array must have size `(N, N)`,
        where `N` is the number of variables in G.

    mask : jnp.DeviceArray
        The mask for the valid actions that can be taken. This array must have
        size `(N, N)`, where `N` is the number of variables in G.

    Returns
    -------
    logits : jnp.DeviceArray
        The logits to compute P(G' | G) the probability of transitioning to a
        new graph G' given G (including terminating, via the terminal state s_f).
        This array has size `(N ** 2 + 1,)`, where `N` is the number of variables.
    """
    # Create the edges as pairs of indices (source, target)
    num_variables = adjacency.shape[0]
    indices = jnp.arange(num_variables ** 2)
    sources, targets = jnp.divmod(indices, num_variables)
    edges = jnp.stack((sources, num_variables + targets), axis=1)

    # Embedding of the edges
    embeddings = hk.Embed(2 * num_variables, embed_dim=128)(edges)
    embeddings = embeddings.reshape(num_variables ** 2, -1)

    # Reshape the adjacency matrix
    adjacency = adjacency.reshape(num_variables ** 2, 1)

    # Apply common body
    num_layers = 5
    for i in range(3):
        embeddings = TransformerBlock(
            num_heads=4,
            key_size=64,
            embedding_size=128,
            init_scale=2. / num_layers,
            widening_factor=2,
            name=f'body_{i+1}'
        )(embeddings, adjacency)

    # Apply individual heads
    logits = logits_head(embeddings, adjacency)
    stop = stop_head(embeddings, adjacency)

    # Return the logits
    return log_policy(logits, stop, mask)


def logits_head(embeddings, adjacency):
    num_layers = 5

    for i in range(2):
        embeddings = TransformerBlock(
            num_heads=4,
            key_size=64,
            embedding_size=128,
            init_scale=2. / num_layers,
            widening_factor=2,
            name=f'head_logits_{i+1}'
        )(embeddings, adjacency)

    logits = hk.nets.MLP([256, 128, 1])(embeddings)
    return jnp.squeeze(logits, axis=-1)


def stop_head(embeddings, adjacency):
    num_layers = 5

    for i in range(2):
        embeddings = TransformerBlock(
            num_heads=4,
            key_size=64,
            embedding_size=128,
            init_scale=2. / num_layers,
            widening_factor=2,
            name=f'head_stop_{i+1}'
        )(embeddings, adjacency)

    mean = jnp.mean(embeddings, axis=-2)  # Average over edges
    return hk.nets.MLP([256, 128, 1])(mean)
