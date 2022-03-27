import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
import pickle
import jax

from tqdm import trange
from numpy.random import default_rng

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics
from dag_gflownet.utils import io


def main(args):
    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    # Create the environment
    scorer, data, graph = get_scorer(args, rng=rng)
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables
    )

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(
        delta=args.delta,
        update_target_every=args.update_target_every
    )
    optimizer = optax.adam(args.lr)
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['adjacency'],
        replay.dummy['mask']
    )
    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - args.min_exploration),
        transition_steps=args.num_iterations // 2,
        transition_begin=args.prefill,
    ))

    # Training loop
    indices = None
    observations = env.reset()
    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            actions, key, logs = gflownet.act(params.online, key, observations, epsilon)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs = gflownet.step(params, state, samples)

                pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

    # Evaluate the posterior estimate
    posterior, _ = posterior_estimate(
        gflownet,
        params.online,
        env,
        key,
        num_samples=args.num_samples_posterior,
        desc='Sampling from posterior'
    )

    # Compute the metrics
    ground_truth = nx.to_numpy_array(graph, weight=None)
    results = {
        'expected_shd': expected_shd(posterior, ground_truth),
        'expected_edges': expected_edges(posterior),
        **threshold_metrics(posterior, ground_truth)
    }

    # Save model, data & results
    args.output_folder.mkdir(exist_ok=True)
    with open(args.output_folder / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, default=str)
    data.to_csv(args.output_folder / 'data.csv')
    with open(args.output_folder / 'graph.pkl', 'wb') as f:
        pickle.dump(graph, f)
    io.save(args.output_folder / 'model.npz', params=params.online)
    replay.save(args.output_folder / 'replay_buffer.npz')
    np.save(args.output_folder / 'posterior.npy', posterior)
    with open(args.output_folder / 'results.json', 'w') as f:
        json.dump(results, f, default=list)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    import json

    parser = ArgumentParser(description='DAG-GFlowNet for Strucure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--scorer_kwargs', type=json.loads, default='{}',
        help='Arguments of the scorer.')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
        help='Arguments of the prior over graphs.')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
        help='Number of iterations (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')
    
    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    exploration.add_argument('--update_epsilon_every', type=int, default=10,
        help='Frequency of update for epsilon (default: %(default)s)')
    
    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--update_target_every', type=int, default=1000,
        help='Frequency of update for the target network (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    misc.add_argument('--num_workers', type=int, default=4,
        help='Number of workers (default: %(default)s)')
    misc.add_argument('--mp_context', type=str, default='spawn',
        help='Multiprocessing context (default: %(default)s)')
    misc.add_argument('--output_folder', type=Path, default='output',
        help='Output folder (default: %(default)s)')

    subparsers = parser.add_subparsers(help='Type of graph', dest='graph')

    # Erdos-Renyi Linear-Gaussian graphs
    er_lingauss = subparsers.add_parser('erdos_renyi_lingauss')
    er_lingauss.add_argument('--num_variables', type=int, required=True,
        help='Number of variables')
    er_lingauss.add_argument('--num_edges', type=int, required=True,
        help='Average number of edges')
    er_lingauss.add_argument('--num_samples', type=int, required=True,
        help='Number of samples')

    # Flow cytometry data (Sachs) with observational data
    sachs_continuous = subparsers.add_parser('sachs_continuous')

    # Flow cytometry data (Sachs) with interventional data
    sachs_intervention = subparsers.add_parser('sachs_interventional')

    args = parser.parse_args()

    main(args)
