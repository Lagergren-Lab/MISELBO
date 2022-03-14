import argparse
import os

from dataManipulation import *
from utils import summary, summary_raw, get_support_from_mcmc
from vbpi import VBPI
from utils import namenum
import time
import torch
import numpy as np
import datetime

import math

parser = argparse.ArgumentParser()

######### Load arguments
parser.add_argument('--date', default=None, help='Specify the experiment date. i.e. 2021-05-17')

######### Data arguments
parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')
parser.add_argument('--empFreq', default=False, action='store_true', help='emprical frequence for KL computation')


######### Model arguments
parser.add_argument('--flow_type', type=str, default='identity', help=' identity | planar | realnvp ')
parser.add_argument('--psp', type=bool, default=True, help=' turn on psp branch length feature, default=True')
parser.add_argument('--nf', type=int, default=2, help=' branch length feature embedding dimension ')
parser.add_argument('--sh', type=list, default=[100], help=' list of the hidden layer sizes for permutation invariant flow ')
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

parser.add_argument('--seed_val', type=int, default=11, help='seed value')

args = parser.parse_args()

args.result_folder = 'results/' + args.dataset
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

args.save_to_path = args.result_folder + '/'
if args.flow_type != 'identity':
    args.save_to_path = args.save_to_path + args.flow_type + '_' + str(args.Lnf)
else:
    args.save_to_path += 'base'
args.save_to_path = args.save_to_path + '_' + str(datetime.date.today()) + '.pt'
 
print('Model with the following settings: {}'.format(args))

###### Load Data
print('\nLoading Data set: {} ......'.format(args.dataset))
run_time = -time.time()

tree_dict_ufboot, tree_names_ufboot = summary_raw(args.dataset, 'data/' + args.dataset + '/ufboot/')
data, taxa = loadData('data/' + args.dataset + '/' + args.dataset + '.fasta', 'fasta')

run_time += time.time()
print('Support loaded in {:.1f} seconds'.format(run_time))

rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_ufboot, tree_names_ufboot)
del tree_dict_ufboot, tree_names_ufboot

L = 1000  # n_particles

model_list = []
for model_seed in [4, 15, 23, 42, 108]:

    model = VBPI(taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden=np.ones(4)/4., subModel=('JC', 1.0),
                     emp_tree_freq=None, feature_dim=args.nf, hidden_sizes=args.sh, num_of_layers_nf=args.Lnf,
                     flow_type=args.flow_type)

    # Load Model State and Sample Trees
    model_fname = args.result_folder + "/" + "m" + str(model_seed) + "_" + args.flow_type + '_' + str(args.Lnf) + ".pt"
    model.load_from(model_fname)
    model_list.append(model)

S = len(model_list)
print('Models are loaded. \tNumber of models (S): ', len(model_list), "\tNumber of particles (L): ", L)

########
t_start = time.time()
print("\nSampling trees and branch lengths...")
torch.manual_seed(args.seed_val)
np.random.seed(args.seed_val)

########
t_start = time.time()
print("\nCalculating IWELBO Original...")
torch.manual_seed(args.seed_val)
np.random.seed(args.seed_val)

lower_bounds = []
for s in range(S):
    cur_model = model_list[s]
    torch.manual_seed(args.seed_val)
    np.random.seed(args.seed_val)
    lb = cur_model.lower_bound(n_particles=L, n_runs=1)
    lower_bounds.append(lb)

iwelbo_orig = np.mean(lower_bounds)
print("Time taken by lower bound computation: ", str(time.time() - t_start))
print("IWELBO: ", iwelbo_orig)
print("Mean LB: ", np.mean(lower_bounds), "\tStd LB: ", np.std(lower_bounds),
      "\tMin LB: ", np.min(lower_bounds), "\tMax LB: ", np.max(lower_bounds))
print("LBs: ", lower_bounds)

########
t_start = time.time()
print("\nSampling trees and branch lengths...")
torch.manual_seed(args.seed_val)
np.random.seed(args.seed_val)

lower_bounds = []
for s in range(S):
    cur_model = model_list[s]
    torch.manual_seed(args.seed_val)
    np.random.seed(args.seed_val)

    samp_trees, samp_log_branch, log_q_branch, logq_tree, log_q_joint, log_ll, log_p_prior, log_p_joint \
        = cur_model.sample_trees_and_branch_lengths(n_particles=L)
    lower_bounds.append(torch.logsumexp(log_ll + log_p_prior + cur_model.log_p_tau - log_q_branch - logq_tree - math.log(L), 0).item())
iwelbo = np.mean(lower_bounds)
print("Time taken by lower bound computation: ", str(time.time() - t_start))
print("IWELBO: ", iwelbo)
print("Mean LB: ", np.mean(lower_bounds), "\tStd LB: ", np.std(lower_bounds),
      "\tMin LB: ", np.min(lower_bounds), "\tMax LB: ", np.max(lower_bounds))
print("LBs: ", lower_bounds)

########
t_start = time.time()
print("\nCalculating IWELBO Presampled...")
torch.manual_seed(args.seed_val)
np.random.seed(args.seed_val)

# Sample trees and get corresponding branch length parameters
tree_list, params_list, samp_log_branch_base_list = [], [], []
final_samp_log_branch_list, final_logq_branch_list = [], []
log_p_joint_list = []
log_ll_list, log_p_prior_list = [], []
for s in range(S):
    cur_model = model_list[s]
    torch.manual_seed(args.seed_val)
    np.random.seed(args.seed_val)

    # Sample Trees
    samp_trees = [cur_model.tree_model.sample_tree() for particle in range(L)]
    [namenum(tree, cur_model.taxa) for tree in samp_trees]

    torch.manual_seed(args.seed_val)
    np.random.seed(args.seed_val)

    samp_log_branch, logq_branch, neigh_ss_idxes, mean, std \
        = cur_model.branch_model.mean_std_encoder(samp_trees, return_details=True)

    tree_list.append(samp_trees)
    params_list.append([mean, std, neigh_ss_idxes])
    samp_log_branch_base_list.append(samp_log_branch)

    samp_log_branch, logq_branch = cur_model.branch_model.invariant_flow(samp_log_branch, logq_branch, neigh_ss_idxes)
    final_samp_log_branch_list.append(samp_log_branch)
    final_logq_branch_list.append(logq_branch)

    log_ll = torch.stack([cur_model.phylo_model.loglikelihood(log_branch, tree)
                          for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
    logp_prior = cur_model.phylo_model.logprior(samp_log_branch)
    log_p_joint_list.append(log_ll + logp_prior + cur_model.log_p_tau)
    log_ll_list.append(log_ll)
    log_p_prior_list.append(logp_prior)

print("Time taken by sampling: ", str(time.time() - t_start))

# Evaluate the lower bound V1 (using final values)
t_start = time.time()
lower_bounds = []
for s in range(S):
    cur_model = model_list[s]
    samp_trees = tree_list[s]
    samp_log_branch = final_samp_log_branch_list[s]
    logq_branch = final_logq_branch_list[s]

    logq_tree = torch.stack([cur_model.logq_tree(tree) for tree in samp_trees])

    log_ll = torch.stack([cur_model.phylo_model.loglikelihood(log_branch, tree)
                          for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
    log_p_prior = cur_model.phylo_model.logprior(samp_log_branch)

    lower_bounds.append(torch.logsumexp(log_ll + log_p_prior + cur_model.log_p_tau - logq_tree - logq_branch - math.log(L), 0).item())
iwelbo = np.mean(lower_bounds)
print("Time taken by lower bound computation: ", str(time.time() - t_start))
print("IWELBO: ", iwelbo)
print("Mean LB: ", np.mean(lower_bounds), "\tStd LB: ", np.std(lower_bounds),
      "\tMin LB: ", np.min(lower_bounds), "\tMax LB: ", np.max(lower_bounds))
print("LBs: ", lower_bounds)

# Evaluate the lower bound V2 (use base values to reach final values)
lower_bounds = []
with torch.no_grad():
    for s in range(S):
        cur_model = model_list[s]
        samp_trees = tree_list[s]
        mean, std, neigh_ss_idxes = params_list[s]
        samp_log_branch_base = samp_log_branch_base_list[s]
        log_p_joint = log_p_joint_list[s]

        logq_branch = torch.sum(-0.5 * math.log(2 * math.pi) - std - (0.5 * (samp_log_branch_base - mean) ** 2) / (std.exp() ** 2), -1)
        samp_log_branch, logq_branch = cur_model.branch_model.invariant_flow(samp_log_branch_base, logq_branch, neigh_ss_idxes)
        logq_tree = torch.stack([cur_model.logq_tree(tree) for tree in samp_trees])
        log_q_joint = logq_tree + logq_branch

        lower_bounds.append(torch.logsumexp(log_p_joint - log_q_joint - math.log(L), 0).item())
iwelbo = np.mean(lower_bounds)
print("Time taken by lower bound computation: ", str(time.time() - t_start))
print("IWELBO: ", iwelbo)
print("Mean LB: ", np.mean(lower_bounds), "\tStd LB: ", np.std(lower_bounds),
      "\tMin LB: ", np.min(lower_bounds), "\tMax LB: ", np.max(lower_bounds))
print("LBs: ", lower_bounds)

########
t_start = time.time()
print("\nCalculating MISELBO Presampled...")
torch.manual_seed(args.seed_val)
np.random.seed(args.seed_val)

lower_bounds = []
with torch.no_grad():
    for s in range(S):
        cur_model = model_list[s]
        samp_trees = tree_list[s]
        samp_log_branch_base = samp_log_branch_base_list[s]
        log_p_joint = log_p_joint_list[s]

        log_q_part = []
        for s_prime in range(S):
            cur_model_s_prime = model_list[s_prime]
            mean, std, neigh_ss_idxes = params_list[s_prime]

            logq_branch = torch.sum(-0.5 * math.log(2 * math.pi) - std - (0.5 * (samp_log_branch_base - mean) ** 2) / (std.exp() ** 2), -1)
            samp_log_branch, logq_branch = cur_model_s_prime.branch_model.invariant_flow(samp_log_branch_base, logq_branch, neigh_ss_idxes)

            logq_tree = torch.stack([cur_model_s_prime.logq_tree(tree) for tree in samp_trees])
            log_q_joint = logq_tree + logq_branch
            log_q_part.append(log_q_joint)

        denominator = torch.logsumexp(torch.stack(log_q_part) - math.log(S), 0)
        lower_bounds.append(torch.logsumexp(log_p_joint - denominator - math.log(L), 0).item())
miselbo = np.mean(lower_bounds)
print("Time taken by lower bound computation: ", str(time.time() - t_start))
print("MISELBO: ", miselbo)
print("Mean LB: ", np.mean(lower_bounds), "\tStd LB: ", np.std(lower_bounds),
      "\tMin LB: ", np.min(lower_bounds), "\tMax LB: ", np.max(lower_bounds))
print("LBs: ", lower_bounds)

#########
print("\nSummary:")
print("\tFinal IWELBO: ", iwelbo)
print("\tFinal MISELBO: ", miselbo)
print("\tDifference (MISELBO - IWELBO): ", miselbo - iwelbo)
if miselbo - iwelbo < 0:
    print("WARNING! The difference is negative! ", miselbo - iwelbo)

print("\n\tIWELBO_orig: ", iwelbo_orig)
print("\tDifference (IWELBO_orig - IWELBO): ", iwelbo_orig - iwelbo)
if iwelbo_orig - iwelbo != 0:
    print("WARNING! The difference is non-zero! ", iwelbo_orig - iwelbo)


########
t_start = time.time()
print("\nCalculating JSD Presampled...")
torch.manual_seed(args.seed_val)
np.random.seed(args.seed_val)

lower_bounds = []
with torch.no_grad():
    for s in range(S):
        cur_model = model_list[s]
        samp_trees = tree_list[s]
        samp_log_branch_base = samp_log_branch_base_list[s]

        log_qs = 0
        log_q_part = []
        for s_prime in range(S):
            cur_model_s_prime = model_list[s_prime]
            mean, std, neigh_ss_idxes = params_list[s_prime]

            logq_branch = torch.sum(-0.5 * math.log(2 * math.pi) - std - (0.5 * (samp_log_branch_base - mean) ** 2) / (std.exp() ** 2), -1)
            samp_log_branch, logq_branch = cur_model_s_prime.branch_model.invariant_flow(samp_log_branch_base, logq_branch, neigh_ss_idxes)

            logq_tree = torch.stack([cur_model_s_prime.logq_tree(tree) for tree in samp_trees])
            log_q_joint = logq_tree + logq_branch
            log_q_part.append(log_q_joint)

            if s_prime == s:
                log_qs = log_q_joint

        log_q_mixture = torch.logsumexp(torch.stack(log_q_part), 0)
        lower_bounds.append(torch.mean(log_qs - log_q_mixture).item())
jsd = np.log(S) + np.mean(lower_bounds)
print("Time taken by lower bound computation: ", str(time.time() - t_start))
print("JSD: ", jsd)
print("Range: [0, ", np.log(S), "]")
if jsd < 0 or jsd > np.log(S):
    print("WARNING! The JSD is out of range! ", jsd, ". [0, ", np.log(S), "]")
print("LBs: ", lower_bounds)
