import numpy as np
import jax.numpy as jnp
import optax

from tqdm.auto import trange
from jax import nn, lax, random


MASKED_VALUE = -1e5


def mask_logits(logits, masks):
    return masks * logits + (1. - masks) * MASKED_VALUE


def detailed_balance_loss(
        log_pi_t,
        log_pi_tp1,
        actions,
        delta_scores,
        num_edges,
        delta=1.
    ):
    r"""Detailed balance loss.

    This function computes the detailed balance loss, in the specific case
    where all the states are complete. This loss function is given by:

    $$ L(\theta; s_{t}, s_{t+1}) = \left[\log\frac{
        R(s_{t+1})P_{B}(s_{t} \mid s_{t+1})P_{\theta}(s_{f} \mid s_{t})}{
        R(s_{t})P_{\theta}(s_{t+1} \mid s_{t})P_{\theta}(s_{f} \mid s_{t+1})
    }\right]^{2} $$

    In practice, to avoid gradient explosion, we use the Huber loss instead
    of the L2-loss (the L2-loss can be emulated with a large value of delta).
    Moreover, we do not backpropagate the error through $P_{\theta}(s_{f} \mid s_{t+1})$,
    which is computed using a target network.

    Parameters
    ----------
    log_pi_t : jnp.DeviceArray
        The log-probabilities $\log P_{\theta}(s' \mid s_{t})$, for all the
        next states $s'$, including the terminal state $s_{f}$. This array
        has size `(B, N ** 2 + 1)`, where `B` is the batch-size, and `N` is
        the number of variables in a graph.

    log_pi_tp1 : jnp.DeviceArray
        The log-probabilities $\log P_{\theta}(s' \mid s_{t+1})$, for all the
        next states $s'$, including the terminal state $s_{f}$. This array
        has size `(B, N ** 2 + 1)`, where `B` is the batch-size, and `N` is
        the number of variables in a graph. In practice, `log_pi_tp1` is
        computed using a target network with parameters $\theta'$.

    actions : jnp.DeviceArray
        The actions taken to go from state $s_{t}$ to state $s_{t+1}$. This
        array has size `(B, 1)`, where `B` is the batch-size.

    delta_scores : jnp.DeviceArray
        The delta-scores between state $s_{t}$ and state $s_{t+1}$, given by
        $\log R(s_{t+1}) - \log R(s_{t})$. This array has size `(B, 1)`, where
        `B` is the batch-size.

    num_edges : jnp.DeviceArray
        The number of edges in $s_{t}$. This array has size `(B, 1)`, where `B`
        is the batch-size.

    delta : float (default: 1.)
        The value of delta for the Huber loss.

    Returns
    -------
    loss : jnp.DeviceArray
        The detailed balance loss averaged over a batch of samples.

    logs : dict
        Additional information for logging purposes.
    """
    # Compute the forward log-probabilities
    log_pF = jnp.take_along_axis(log_pi_t, actions, axis=-1)

    # Compute the backward log-probabilities
    log_pB = -jnp.log1p(num_edges)

    error = (jnp.squeeze(delta_scores + log_pB - log_pF, axis=-1)
        + log_pi_t[:, -1] - lax.stop_gradient(log_pi_tp1[:, -1]))
    loss = jnp.mean(optax.huber_loss(error, delta=delta))

    logs = {
        'error': error,
        'loss': loss,
    }
    return (loss, logs)


def log_policy(logits, stop, masks):
    masks = masks.reshape(logits.shape)
    masked_logits = mask_logits(logits, masks)
    can_continue = jnp.any(masks, axis=-1, keepdims=True)

    logp_continue = (nn.log_sigmoid(-stop)
        + nn.log_softmax(masked_logits, axis=-1))
    logp_stop = nn.log_sigmoid(stop)

    # In case there is no valid action other than stop
    logp_continue = jnp.where(can_continue, logp_continue, MASKED_VALUE)
    logp_stop = logp_stop * can_continue

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def uniform_log_policy(masks):
    masks = masks.reshape(masks.shape[0], -1)
    num_edges = jnp.sum(masks, axis=-1, keepdims=True)

    logp_stop = -jnp.log1p(num_edges)
    logp_continue = mask_logits(logp_stop, masks)

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def posterior_estimate(
        gflownet,
        params,
        env,
        key,
        num_samples=1000,
        verbose=True,
        **kwargs
    ):
    """Get the posterior estimate of DAG-GFlowNet as a collection of graphs
    sampled from the GFlowNet.

    Parameters
    ----------
    gflownet : `DAGGFlowNet` instance
        Instance of a DAG-GFlowNet.

    params : dict
        Parameters of the neural network for DAG-GFlowNet. This must be a dict
        that can be accepted by the Haiku model in the `DAGGFlowNet` instance.

    env : `GFlowNetDAGEnv` instance
        Instance of the environment.

    key : jax.random.PRNGKey
        Random key for sampling from DAG-GFlowNet.

    num_samples : int (default: 1000)
        The number of samples in the posterior approximation.

    verbose : bool
        If True, display a progress bar for the sampling process.

    Returns
    -------
    posterior : np.ndarray instance
        The posterior approximation, given as a collection of adjacency matrices
        from graphs sampled with the posterior approximation. This array has
        size `(B, N, N)`, where `B` is the number of sample graphs in the
        posterior approximation, and `N` is the number of variables in a graph.

    logs : dict
        Additional information for logging purposes.
    """
    samples = []
    observations = env.reset()
    with trange(num_samples, disable=(not verbose), **kwargs) as pbar:
        while len(samples) < num_samples:
            order = observations['order']
            actions, key, _ = gflownet.act(params, key, observations, 1.)
            observations, _, dones, _ = env.step(np.asarray(actions))

            samples.extend([order[i] for i, done in enumerate(dones) if done])
            pbar.update(min(num_samples - pbar.n, np.sum(dones).item()))
    orders = np.stack(samples[:num_samples], axis=0)
    logs = {
        'orders': orders,
    }
    return ((orders >= 0).astype(np.int_), logs)
