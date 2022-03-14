import argparse
import os

from dataManipulation import *
from utils import summary, summary_raw, get_support_from_mcmc
from vbpi import VBPI
import time
import numpy as np
import datetime

parser = argparse.ArgumentParser()

######### Data arguments
parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')
parser.add_argument('--empFreq', default=False, action='store_true', help='emprical frequence for KL computation')

######### Model arguments
parser.add_argument('--flow_type', type=str, default='identity', help=' identity | planar | realnvp ')
parser.add_argument('--psp', type=bool, default=True, help=' turn on psp branch length feature, default=True')
parser.add_argument('--nf', type=int, default=2, help=' branch length feature embedding dimension ')
parser.add_argument('--sh', type=list, default=[100],
                    help=' list of the hidden layer sizes for permutation invariant flow ')
parser.add_argument('--Lnf', type=int, default=5, help=' number of layers for permutation invariant flow ')

######### Optimizer arguments
parser.add_argument('--stepszTree', type=float, default=0.001, help=' step size for tree topology parameters ')
parser.add_argument('--stepszBranch', type=float, default=0.001, help=' stepsz for branch length parameters ')
parser.add_argument('--maxIter', type=int, default=400000, help=' number of iterations for training, default=400000')
parser.add_argument('--invT0', type=float, default=0.001, help=' initial inverse temperature for annealing schedule, default=0.001')
parser.add_argument('--nwarmStart', type=float, default=100000, help=' number of warm start iterations, default=100000')
parser.add_argument('--nParticle', type=int, default=10, help='number of particles for variational objectives, default=10')
parser.add_argument('--ar', type=float, default=0.75, help='step size anneal rate, default=0.75')
parser.add_argument('--af', type=int, default=20000, help='step size anneal frequency, default=20000')
parser.add_argument('--tf', type=int, default=1000, help='monitor frequency during training, default=1000')
parser.add_argument('--lbf', type=int, default=5000, help='lower bound test frequency, default=5000')

parser.add_argument('--vbpi_seed', type=int, default=1, help='seed for vbpi')

args = parser.parse_args()

args.result_folder = '../results/' + args.dataset + '_vbpi_seed_' + str(args.vbpi_seed)
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

args.save_to_path = args.result_folder + '/'
if args.flow_type != 'identity':
    args.save_to_path = args.save_to_path + args.flow_type + '_' + str(args.Lnf)
else:
    args.save_to_path = args.save_to_path + 'base'
args.save_to_path = args.save_to_path + '_' + str(datetime.date.today()) + '.pt'

print('Training with the following settings: {}'.format(args))

np.random.seed(args.vbpi_seed)

###### Load Data
print('\nLoading Data set: {} ......'.format(args.dataset))
run_time = -time.time()

tree_dict_ufboot, tree_names_ufboot = summary_raw(args.dataset, '../data/' + args.dataset + '/ufboot/')
data, taxa = loadData('../data/' + args.dataset + '/' + args.dataset + '.fasta', 'fasta')

run_time += time.time()
print('Support loaded in {:.1f} seconds'.format(run_time))

if args.empFreq:
    print('\nLoading empirical posterior estimates ......')
    run_time = -time.time()
    tree_dict_total, tree_names_total, tree_wts_total = summary(args.dataset, '../data/' + args.dataset + '/raw_emp/')
    emp_tree_freq = {tree_dict_total[tree_name]: tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}
    run_time += time.time()
    print('Empirical estimates from MrBayes loaded in {:.1f} seconds'.format(run_time))
else:
    emp_tree_freq = None

rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_ufboot, tree_names_ufboot)
del tree_dict_ufboot, tree_names_ufboot

model = VBPI(taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden=np.ones(4)/4., subModel=('JC', 1.0),
                 emp_tree_freq=emp_tree_freq, feature_dim=args.nf, hidden_sizes=args.sh, num_of_layers_nf=args.Lnf,
                 flow_type=args.flow_type)

print('Parameter Info:')
for param in model.parameters():
    print(param.dtype, param.size())
print('feature dim: {}, hidden_sizes: {}, num_of_layers_nf: {}'.format(args.nf, args.sh, args.Lnf))

print('\nVBPI running, Flow type: {}'.format(args.flow_type))
print('Results will be saved to: {}\n'.format(args.save_to_path))
model.learn({'tree':args.stepszTree,'branch':args.stepszBranch}, args.maxIter, test_freq=args.tf, n_particles=args.nParticle, anneal_freq=args.af, init_inverse_temp=args.invT0,
             warm_start_interval=args.nwarmStart, save_to_path=args.save_to_path)
