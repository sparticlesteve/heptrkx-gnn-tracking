import os
import argparse
import time

from crayai import hpo

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/agnn_pbt.yaml')
    parser.add_argument('--nodes', type=int, default=16,
                        help='Number of nodes to run optimization over, total')
    parser.add_argument('--nodes-per-eval', type=int, default=1,
                        help='Number of nodes per individual evaluation')
    parser.add_argument('--demes', type=int, default=2,
                        help='Number of populations')
    parser.add_argument('--pop-size', type=int, default=8,
                        help='Size of the genetic population')
    parser.add_argument('--generations', type=int, default=4,
                        help='Number of generations to run')
    parser.add_argument('--mutation-rate', type=float, default=0.05,
                        help='Mutation rate between generations of genetic optimization')
    parser.add_argument('--crossover-rate', type=float, default=0.33,
                        help='Crossover rate between generations of genetic optimization')
    parser.add_argument('--output-dir', default='./run',
                        help='Directory to store all outputs and checkpoints')
    parser.add_argument('--alloc-args', default='-J hpo -C haswell -q interactive -t 4:00:00')
    return parser.parse_args()

def main():

    args = parse_args()

    # Hardcode some config
    #n_nodes = 4 #32
    #config_file = 'configs/test.yaml'
    #pop_size = 2 #16
    #n_demes = 2 #4
    #n_generations = 4
    #mutation_rate = 0.05
    #crossover_rate = 0.33
    #alloc_args='-J hpo -C haswell -q interactive -t 4:00:00'
    #checkpoint_dir = 'checkpoints'
    
    # Hyperparameters
    params = hpo.Params([
        ['--lr', 0.001, (1e-6, 0.1)],
        ['--n-graph-iters', 4, (1, 16)],
        ['--real-weight', 3., (1., 6.)]
    ])
    
    # Define the command to be run by the evaluator
    cmd = 'python train.py %s' % args.config
    cmd += ' --fom last --n-epochs 1 --resume --output-dir @checkpoint'
    
    # Define the evaluator
    result_dir = os.path.expandvars('$SCRATCH/heptrkx/results/pbt_%s' %
                                    time.strftime('%Y%m%d_%H%M%S'))
    evaluator = hpo.Evaluator(cmd,
                              run_path=result_dir,
                              nodes=args.nodes,
                              launcher='wlm',
                              verbose=True, 
                              nodes_per_eval=args.nodes_per_eval,
                              checkpoint='checkpoints',
                              alloc_args=args.alloc_args)
    
    # Define the Optimizer
    optimizer = hpo.GeneticOptimizer(evaluator,
                                     pop_size=args.pop_size,
                                     num_demes=args.demes,
                                     generations=args.generations,
                                     mutation_rate=args.mutation_rate,
                                     crossover_rate=args.crossover_rate,
                                     verbose=True)
    
    # Run the Optimizer
    optimizer.optimize(params)

if __name__ == '__main__':
    main()
