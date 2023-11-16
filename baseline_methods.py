import copy

import torch
import logging
from utils import epoch_global, get_time


def fedavg(global_net, clients, round, testloader, args, total_data_points=None):
    logging.info("="*50 + " Round: " + str(round) + " " + "="*50)
    for each_worker in range(args.nworkers):
        clients[each_worker].train_net(local_round=args.Iteration, batch_size=args.batch_train, device=args.device, args=args)
        loss_avg, acc_avg = epoch_global('test', testloader, clients[each_worker].model, clients[each_worker].optimizer,
                                         clients[each_worker].criterion, args, aug=False)
        clients[each_worker].model = clients[each_worker].model.cpu()
        logging.info('%s Evaluate_%02d_%02d: val loss = %.6f val acc = %.4f' % (get_time(), round, each_worker, loss_avg, acc_avg))

    fed_avg_freqs = [len(clients[each_worker].each_worker_data) / total_data_points for each_worker in range(args.nworkers)]
    global_para = global_net.cpu().state_dict()

    for each_worker in range(args.nworkers):
        net_para = clients[each_worker].model.cpu().state_dict()
        if each_worker == 0:
            for key in net_para:
                global_para[key] = net_para[key] * fed_avg_freqs[each_worker]
        else:
            for key in net_para:
                global_para[key] += net_para[key] * fed_avg_freqs[each_worker]
    global_net.load_state_dict(global_para)
    global_net.to(args.device)
    return global_net


def fedprox(global_net, clients, round, testloader, args, total_data_points=None):
    logging.info("="*50 + " Round: " + str(round) + " " + "="*50)
    for each_worker in range(args.nworkers):
        clients[each_worker].train_net_fedprox(local_round=args.Iteration, batch_size=args.batch_train, device=args.device, args=args, global_net=global_net)
        loss_avg, acc_avg = epoch_global('test', testloader, clients[each_worker].model, clients[each_worker].optimizer,
                                         clients[each_worker].criterion, args, aug=False)
        clients[each_worker].model = clients[each_worker].model.cpu()
        logging.info('%s Evaluate_%02d_%02d: val loss = %.6f val acc = %.4f' % (get_time(), round, each_worker, loss_avg, acc_avg))

    fed_avg_freqs = [len(clients[each_worker].each_worker_data) / total_data_points for each_worker in range(args.nworkers)]
    global_para = global_net.cpu().state_dict()

    for each_worker in range(args.nworkers):
        net_para = clients[each_worker].model.cpu().state_dict()
        if each_worker == 0:
            for key in net_para:
                global_para[key] = net_para[key] * fed_avg_freqs[each_worker]
        else:
            for key in net_para:
                global_para[key] += net_para[key] * fed_avg_freqs[each_worker]
    global_net.load_state_dict(global_para)
    global_net.to(args.device)
    return global_net


def fednova(global_net, clients, round, testloader, args, total_data_points=None):
    logging.info("=" * 50 + " Round: " + str(round) + " " + "=" * 50)
    a_list = []
    d_list = []
    n_list = []
    for each_worker in range(args.nworkers):
        train_loss, a_i, d_i, n_i = clients[each_worker].train_net_fednova(local_round=args.Iteration,
                                    batch_size=args.batch_train, device=args.device, args=args, global_net=global_net)
        a_list.append(a_i)
        d_list.append(d_i)
        # n_list.append(n_i)
        loss_avg, acc_avg = epoch_global('test', testloader, clients[each_worker].model, clients[each_worker].optimizer,
                                         clients[each_worker].criterion, args, aug=False)
        clients[each_worker].model = clients[each_worker].model.cpu()
        logging.info('%s Evaluate_%02d_%02d: val loss = %.6f val acc = %.4f' % (get_time(), round, each_worker, loss_avg, acc_avg))

    fed_avg_freqs = [len(clients[each_worker].each_worker_data) / total_data_points for each_worker in range(args.nworkers)]
    global_para = global_net.state_dict()

    d_total_round = copy.deepcopy(global_net.state_dict())
    for key in d_total_round:
        d_total_round[key] = 0.0
    for each_worker in range(args.nworkers):
        for key in d_list[each_worker]:
            d_total_round[key] += d_list[each_worker][key] * fed_avg_freqs[each_worker]

    coeff = 0.0
    for each_worker in range(args.nworkers):
        coeff += a_list[each_worker] * fed_avg_freqs[each_worker]

    for key in global_para:
        if global_para[key].type() == 'torch.LongTensor':
            global_para[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
        elif global_para[key].type() == 'torch.cuda.LongTensor':
            global_para[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
        else:
            global_para[key] -= coeff * d_total_round[key]
    global_net.load_state_dict(global_para)
    global_net.to(args.device)
    return global_net
