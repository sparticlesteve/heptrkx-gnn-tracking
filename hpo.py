import os
import argparse
import time

from crayai import hpo

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/agnn.yaml')
    # Allocation arguments
    parser.add_argument('--alloc-args',
                        default='-C gpu --gpus-per-task 1 -c 10 --exclusive',
                        help='Extra alloc arguments to apply')
    parser.add_argument('-t', '--time', help='SLURM job allocation time')
    parser.add_argument('--nodes', type=int,
                        help='Number of nodes to run optimization over, total')
    parser.add_argument('--nodes-per-eval', type=int, default=1,
                        help='Number of nodes per individual evaluation')
    parser.add_argument('--ntasks-per-node', type=int,
                        help='Number of tasks per node')
    parser.add_argument('--alg', default='random', choices=['random', 'genetic'],
                        help='Specify the HPO algorithm to use')
    # Training configuration
    parser.add_argument('--epochs', type=int, default=16,
                        help='Number of epochs to train in each evaluation')
    # Random search arguments
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of random search iterations')
    # Genetic search arguments
    parser.add_argument('--demes', type=int, default=4,
                        help='Number of populations')
    parser.add_argument('--pop-size', type=int, default=4,
                        help='Size of the genetic population')
    parser.add_argument('--generations', type=int, default=4,
                        help='Number of generations to run')
    parser.add_argument('--mutation-rate', type=float, default=0.05,
                        help='Mutation rate between generations of genetic optimization')
    parser.add_argument('--crossover-rate', type=float, default=0.33,
                        help='Crossover rate between generations of genetic optimization')
    return parser.parse_args()

def main():

    args = parse_args()

    # Hyperparameters
    params = hpo.Params([
        ['--hidden-dim', 32, [16, 32, 64, 128, 256]],
        ['--n-edge-layers', 4, [1, 2, 4, 8]],
        ['--n-node-layers', 4, [1, 2, 4, 8]],
        ['--weight-decay', 1.e-4, (0., 1e-3)],
        ['--n-graph-iters', 8, (4, 16)],
        ['--real-weight', 3., (1., 6.)],
        ['--lr', 0.001, [1e-5, 1e-4, 1e-3, 1e-2]],
    ])

    # Define the command to be run by the evaluator
    output_dir = "'${SCRATCH}/heptrkx/results/hpo_%s_${SLURM_JOB_ID}_${SLURM_STEP_ID}'" % (
        os.path.basename(args.config).split('.')[0])
    cmd = ('python -u train.py %s' % args.config +
           ' --rank-gpu -d ddp-file --fom best' +
           ' --n-epochs %i' % args.epochs +
           ' --output-dir %s' % output_dir)

    # SLURM options
    n_nodes = (args.nodes if args.nodes is not None
               else int(os.environ['SLURM_JOB_NUM_NODES']))
    n_tasks_per_node = (args.ntasks_per_node if args.ntasks_per_node is not None
                        else int(os.environ['SLURM_NTASKS_PER_NODE']))
    n_tasks_per_eval = args.nodes_per_eval * n_tasks_per_node
    alloc_args = ('-J hpo %s --time %s' % (args.alloc_args, args.time) +
                  ' --ntasks-per-node %i' % n_tasks_per_node)
    launch_args = '-n %i -u' % n_tasks_per_eval

    # Define the evaluator
    evaluator = hpo.Evaluator(cmd,
                              nodes=n_nodes,
                              workload_manager='slurm',
                              alloc_args=alloc_args,
                              launch_args=launch_args,
                              nodes_per_eval=args.nodes_per_eval,
                              verbose=True)

    # Random search optimizer
    if args.alg == 'random':
        optimizer = hpo.RandomOptimizer(evaluator, num_iters=args.iters)

    # Genetic search optimizer
    else:
        results_file = 'hpo.log'
        optimizer = hpo.GeneticOptimizer(evaluator,
                                         pop_size=args.pop_size,
                                         num_demes=args.demes,
                                         generations=args.generations,
                                         mutation_rate=args.mutation_rate,
                                         crossover_rate=args.crossover_rate,
                                         verbose=True,
                                         log_fn=results_file)

    # Run the Optimizer
    optimizer.optimize(params)

if __name__ == '__main__':
    main()
