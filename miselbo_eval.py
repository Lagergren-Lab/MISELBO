# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os

import torch
import numpy as np
from time import time
import pandas as pd

from torch.multiprocessing import Process
import torch.distributed as dist

from model import AutoEncoder
import utils
import datasets
from train import test, init_processes, find_free_port


def compute_miselbo(logP, logQ):
    # B is the size of the mini-batch (perhaps remove for vbpi)
    B, S, L = logP.size()
    # logsumexp over the S mixture components in the denominator of the log ratio
    logQ_mixtures = torch.logsumexp(logQ, dim=-1) - np.log(S)
    # logsumexp over the L importance samples, results in S log ratios
    log_ratios = torch.logsumexp(logP - logQ_mixtures, dim=-1) - np.log(L)
    # sum over both batch and S weights
    miselbo = torch.sum(1/S * log_ratios, dim=-1)
    return miselbo


def compute_average_iwelbo(logP, logQ):
    B, S, L = logP.size()
    iwelbos = torch.zeros((B, S)).cuda()
    average_iwelbo = 0
    for s in range(S):
        iwelbo = (torch.logsumexp(logP[:, s] - logQ[:, s, :, s], dim=-1) - np.log(L))
        iwelbos[:, s] = iwelbo
        average_iwelbo += 1 / S * iwelbo
    # average_iwelbo = np.sum(iwelbos * 1 / S, axis=-1)
    return average_iwelbo, iwelbos


def compute_jsd(logQ):
    B, S, L, S = logQ.size()
    logQ_mixtures = torch.logsumexp(logQ, dim=-1).cuda()
    jsd = torch.log(torch.ones(B)*S).cuda()
    for s in range(S):
        jsd += (1 / S) * torch.sum((logQ[:, s, :, s] - logQ_mixtures[:, s]) * (1 / L), dim=-1)
    return jsd


def calc_elbos_and_jsds(valid_queue, L, args, logging, models, only_miselbo=False, only_iwelbo=False):
    S = len(models)
    if args.distributed:
        dist.barrier()
    miselbo = 0
    iwelbo = torch.zeros((1, S)).cuda()
    avg_iwelbo = 0
    jsd = 0
    for model in models:
        model.eval()
    for step, x in enumerate(valid_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()
        logQ = torch.zeros((x.size(0), S, L, S)).cuda()
        logP = torch.zeros((x.size(0), S, L)).cuda()
        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            for s, model in enumerate(models):
                for k in range(L):
                    logits, log_q, log_p, kl_all, kl_diag, z_s, s_s = model.forward_miselbo(x)
                    output = model.decoder_output(logits)
                    recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                    logP[:, s, k] = -recon_loss + log_p
                    for j, model_j in enumerate(models):
                        if j == s:
                            logQ[:, s, k, j] = log_q
                        else:
                            logits, log_q_j, log_p_j, kl_all_j, kl_diag_j, _, _ = model_j.forward_miselbo(x, z_s, s_s)
                            logQ[:, s, k, j] = log_q_j

        if only_iwelbo:
            miselbo_batch = torch.Tensor(0)
            avg_iwelbo_batch, iwelbo_batch = compute_average_iwelbo(logP, logQ)
            jsd_batch = torch.Tensor(0)
        elif only_miselbo:
            miselbo_batch = compute_miselbo(logP, logQ)
            avg_iwelbo_batch = torch.Tensor(0)
            iwelbo_batch = torch.Tensor(0)
            jsd_batch = torch.Tensor(0)
        else:
            miselbo_batch = compute_miselbo(logP, logQ)
            avg_iwelbo_batch, iwelbo_batch = compute_average_iwelbo(logP, logQ)
            jsd_batch = compute_jsd(logQ)

        step += 1
        miselbo = (step - 1) / step * miselbo + (1 / step) * torch.mean(miselbo_batch)
        avg_iwelbo = (step - 1) / step * avg_iwelbo + (1 / step) * torch.mean(avg_iwelbo_batch)
        iwelbo = (step - 1) / step * iwelbo + (1 / step) * torch.mean(iwelbo_batch, dim=0)
        jsd = (step - 1) / step * jsd + (1 / step) * torch.mean(jsd_batch)
        step -= 1
        if step % 100 == 0 and not (only_iwelbo or only_miselbo):
            print(f"Batch number: {step}")
            print(f"miselbo: {miselbo}")
            print(f"avg_iwelbo: {avg_iwelbo}")
            print(f"iwelbo: {iwelbo}")
            print(f"JSD: {jsd}")

    logging.info(f"miselbo: {miselbo}")
    logging.info(f"avg_iwelbo: {avg_iwelbo}")
    logging.info(f"iwelbo: {iwelbo}")
    logging.info(f"JSD: {jsd}")
    print("------ Final results -----")
    print(f"miselbo: {miselbo}")
    print(f"avg_iwelbo: {avg_iwelbo}")
    print(f"diff elbo: {miselbo - avg_iwelbo}")
    print(f"iwelbo: {iwelbo}")
    print(f"JSD: {jsd}")
    return miselbo, avg_iwelbo, iwelbo, jsd


def main(eval_args):
    logging = utils.Logger(eval_args.local_rank, eval_args.save, miselbo_eval=True)

    # ensures that weight initializations are all the same

    # load a checkpoint
    logging.info('loading q0:')
    logging.info(eval_args.q0)
    checkpoint = torch.load(eval_args.q0, map_location='cpu')
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        logging.info('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    logging.info('loaded the q0 at epoch %d', checkpoint['epoch'])
    arch_instance = utils.get_arch_cells(args.arch_instance)
    model_0 = AutoEncoder(args, None, arch_instance)
    # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
    # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
    # did not have this variable.
    model_0.load_state_dict(checkpoint['state_dict'], strict=False)
    model_0 = model_0.cuda()
    models = [model_0]

    models = add_models(eval_args, logging, models)
    M = len(models)
    validate_diff_models = True
    if validate_diff_models:
        for m in range(1, M):
            print(f"diff models 0 and {m}:")
            logging.info(f"diff models 0 and {m}:")
            compare_parameters(model_0, models[m], logging)

    logging.info('args = %s', args)
    logging.info('num conv layers: %d', len(model_0.all_conv_layers))
    logging.info('param size = %fM ', utils.count_parameters_in_M(model_0))

    # load train valid queue
    args.data = eval_args.data
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)

    if eval_args.eval_on_train:
        logging.info('Using the training data for eval.')
        valid_queue = train_queue

    # get number of bits
    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output
    S = len(models)

    if eval_args.ablation:
        df = pd.DataFrame(columns=['S', 'L', 'MISELBO', 'JSD', 'avg IWELBO', 'iwelbos'])
        i = 0
        for s in range(2, S):
            print(f"-------- S: {s} ---------------")
            models_s = models[0:s]
            logging.info('--------- S %f -----------', s)
            for L in [1, 2, 3, 5, 10, 50]:
                torch.manual_seed(0)
                print(f"L: {L}")
                logging.info('----S %f, L %f -----------', s, L)
                miselbo, avg_iwelbo, iwelbo, jsd = calc_elbos_and_jsds(valid_queue, models=models_s, L=L, args=args, logging=logging)
                df.loc[i] = [s, L, -miselbo.cpu().numpy(), jsd.cpu().numpy(), -avg_iwelbo.cpu().numpy(), -iwelbo.cpu().numpy()]
                i += 1
    else:
        torch.manual_seed(0)
        L = eval_args.num_iw_samples
        print(f"---- S: {int(S)} ---- L: {int(L)} -----")
        logging.info(f"----S {int(S)}, L {int(L)} -----------")
        time_eval = eval_args.time_eval
        if time_eval:
            start_time = time()
        miselbo, avg_iwelbo, iwelbo, jsd = calc_elbos_and_jsds(valid_queue, models=models, L=L, args=args,
                                                               logging=logging, only_miselbo=eval_args.only_miselbo,
                                                               only_iwelbo=eval_args.only_iwelbo)
        if time_eval:
            time_elapsed = time() - start_time
        logging.info('final valid miselbo %f', miselbo)
        logging.info('final valid avg_iwelbo %f', avg_iwelbo)
        logging.info(f"final valid iwelbo {iwelbo}")
        logging.info(f"final best iwelbo {torch.max(iwelbo)}")
        logging.info('final valid JSD %f', jsd)
        logging.info('final valid delta L %f', avg_iwelbo - miselbo)
        if time_eval:
            logging.info('Eval times %f', time_elapsed)

    if eval_args.ablation:
        df.to_excel(os.path.join(eval_args.save, eval_args.ablation_file_name))
    save = False
    if save:
        np.save(os.path.join(eval_args.save, 'miselbo'), miselbo.cpu().numpy())
        np.savetxt(os.path.join(eval_args.save, 'miselbo.txt'), miselbo.cpu().numpy().reshape(1,1))
        np.save(os.path.join(eval_args.save, 'avg_iwelbo'), avg_iwelbo.cpu().numpy())
        np.savetxt(os.path.join(eval_args.save, 'avg_iwelbo.txt'), avg_iwelbo.cpu().numpy().reshape(1,1))
        np.save(os.path.join(eval_args.save, 'iwelbo'), iwelbo.cpu().numpy())
        np.savetxt(os.path.join(eval_args.save, 'iwelbo.txt'), iwelbo.cpu().numpy().reshape(S ,1))
        np.save(os.path.join(eval_args.save, 'jsd'), jsd.cpu().numpy())
        np.savetxt(os.path.join(eval_args.save, 'jsd.txt'), jsd.cpu().numpy().reshape(1,1))


def add_models(eval_args, logging, models):
    if eval_args.q1 != 'None':
        logging.info('loading q1:')
        logging.info(eval_args.q1)
        checkpoint_1 = torch.load(eval_args.q1, map_location='cpu')
        args_1 = checkpoint_1['args']
        args_1.num_mixture_dec = 10
        logging.info('loaded the q1 at epoch %d', checkpoint_1['epoch'])
        arch_instance_1 = utils.get_arch_cells(args_1.arch_instance)
        model_1 = AutoEncoder(args_1, None, arch_instance_1)
        model_1.load_state_dict(checkpoint_1['state_dict'], strict=False)
        model_1.cuda()
        models.append(model_1)

    if eval_args.q2 != 'None':
        logging.info('loading q2:')
        logging.info(eval_args.q2)
        checkpoint_2 = torch.load(eval_args.q2, map_location='cpu')
        args_2 = checkpoint_2['args']
        args_2.num_mixture_dec = 10
        logging.info('loaded the q2 at epoch %d', checkpoint_2['epoch'])
        arch_instance_2 = utils.get_arch_cells(args_2.arch_instance)
        model_2 = AutoEncoder(args_2, None, arch_instance_2)
        model_2.load_state_dict(checkpoint_2['state_dict'], strict=False)
        model_2.cuda()
        models.append(model_2)

    if eval_args.q3 != 'None':
        logging.info('loading q3:')
        logging.info(eval_args.q3)
        checkpoint_3 = torch.load(eval_args.q3, map_location='cpu')
        args_3 = checkpoint_3['args']
        args_3.num_mixture_dec = 10
        logging.info('loaded the q3 at epoch %d', checkpoint_3['epoch'])
        arch_instance_3 = utils.get_arch_cells(args_3.arch_instance)
        model_3 = AutoEncoder(args_3, None, arch_instance_3)
        model_3.load_state_dict(checkpoint_3['state_dict'], strict=False)
        model_3.cuda()
        models.append(model_3)

    if eval_args.q4 != 'None':
        logging.info('loading q4:')
        logging.info(eval_args.q4)
        checkpoint_4 = torch.load(eval_args.q4, map_location='cpu')
        args_4 = checkpoint_4['args']
        args_4.num_mixture_dec = 10
        logging.info('loaded the q4 at epoch %d', checkpoint_4['epoch'])
        arch_instance_4 = utils.get_arch_cells(args_4.arch_instance)
        model_4 = AutoEncoder(args_4, None, arch_instance_4)
        model_4.load_state_dict(checkpoint_4['state_dict'], strict=False)
        model_4.cuda()
        models.append(model_4)

    if eval_args.q5 != 'None':
        logging.info('loading q5:')
        logging.info(eval_args.q5)
        checkpoint_5 = torch.load(eval_args.q5, map_location='cpu')
        args_5 = checkpoint_5['args']
        args_5.num_mixture_dec = 10
        logging.info('loaded the q5 at epoch %d', checkpoint_5['epoch'])
        arch_instance_5 = utils.get_arch_cells(args_5.arch_instance)
        model_5 = AutoEncoder(args_5, None, arch_instance_5)
        model_5.load_state_dict(checkpoint_5['state_dict'], strict=False)
        model_5.cuda()
        models.append(model_5)
    return models


def compare_parameters(model_0, model_1, logging):
    diff_enc = 0
    for (e0, e1) in zip(model_0.enc_tower.parameters(), model_1.enc_tower.parameters()):
        diff_enc += torch.sum(torch.abs(e0 - e1))
    print(f"diff enc tower: {diff_enc}")
    logging.info(f"diff enc tower: {diff_enc}")

    diff_dec_tower = 0
    for (d0, d1) in zip(model_0.dec_tower.parameters(), model_1.dec_tower.parameters()):
        diff_dec_tower += torch.sum(torch.abs(d0 - d1))
    print(f"diff dec tower: {diff_dec_tower}")
    logging.info(f"diff dec tower: {diff_dec_tower}")
    diff_samp = 0
    for (d0, d1) in zip(model_0.dec_sampler.parameters(), model_1.dec_sampler.parameters()):
        diff_samp += torch.sum(torch.abs(d0 - d1))
    print(f"diff sampler: {diff_samp}")
    logging.info(f"diff sampler: {diff_samp}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--q0', type=str, default='results/eval-exp_test/checkpoint_0.pt',
                        help='location of the checkpoint of q0')
    parser.add_argument('--q1', type=str, default='None',
                        help='location of the checkpoint of q1')
    parser.add_argument('--q2', type=str, default='None',
                        help='location of the checkpoint of q2')
    parser.add_argument('--q3', type=str, default='None',
                        help='location of the checkpoint of q3')
    parser.add_argument('--q4', type=str, default='None',
                        help='location of the checkpoint of q4')
    parser.add_argument('--q5', type=str, default='None',
                        help='location of the checkpoint of q5')
    parser.add_argument('--ablation', action='store_true', default=False,
                        help='Run ablation study.')
    parser.add_argument('--ablation_file_name', type=str, default=None,
                        help='location of the checkpoint')
    parser.add_argument('--save', type=str, default='/results/miselbo',
                        help='location of the checkpoint')
    parser.add_argument('--data', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--num_iw_samples', type=int, default=1000,
                        help='The number of IW samples used in test_ll mode.')
    parser.add_argument('--L_list', type=str, default=None,
                        help='Used for experiments involving different L.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--time_eval', action='store_true', default=False,
                        help='Time and log evaluation time.')
    parser.add_argument('--only_miselbo', action='store_true', default=False,
                        help='Only calculate the MISELBO.')
    parser.add_argument('--only_iwelbo', action='store_true', default=False,
                        help='Only calculate IWELBO and average IWELBO.')
    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')

    args = parser.parse_args()
    utils.create_exp_dir(args.save)

    size = args.world_size

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            p = Process(target=init_processes, args=(rank, size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = False
        init_processes(0, size, main, args, master_port=find_free_port())