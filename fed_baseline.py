import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import sys
import random
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug, epoch_global, get_logger
from clients import Client
from distill_client import client_update
from glad_utils import *
from baseline_methods import *


def get_fname(args):
    parserString = "Baseline_Method"+str(args.baseline_method)+"_Dataset"+str(args.dataset)+"_Iteration"+\
                   str(args.Iteration)+"_nworkers"+str(args.nworkers)+"_beta"+str(args.beta)+"_lrNet"+\
                   str(args.lr_net)+"_round"+str(args.round)
    return parserString


def main(args):

    input_str = ' '.join(sys.argv)
    print(input_str)
    print(args.local_rank)
    logger = get_logger('log/' + get_fname(args) + time.strftime("%Y%m%d-%H%M%S") + ".log", distributed_rank=args.local_rank)

    for k, v in sorted(vars(args).items()):
        logger.info(str(k) + '=' + str(v))
    logger.info(input_str)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank) if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), get_fname(args))

    args.save_path = os.path.join(args.save_path, "dm", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.distributed = torch.cuda.device_count() > 1

    each_worker_data, each_worker_label, indices_cl_classes = build_client_dataset_dirichlet(dst_train, class_map, num_classes, args)
    total_data_points = len(dst_train)
    fed_avg_freqs = [len(each_worker_data[each_worker]) / total_data_points for each_worker in
                     range(args.nworkers)]

    for each_client in range(args.nworkers):
        print(each_client, " ", len(each_worker_data[each_client]))
        print([len(indices_cl_classes[each_client][c]) for c in range(10)])
        print("="*100)
    print(fed_avg_freqs, sum(fed_avg_freqs))

    def get_images(c, n, each_worker):  # get random n images of class c from client each_worker
        idx_shuffle = np.random.permutation(indices_cl_classes[each_worker][c])[:n]
        return each_worker_data[each_worker][idx_shuffle].cuda(non_blocking=True)

    global_model_acc = []
    global_model_loss = []

    """initialize clients"""
    Clients = []
    for each_worker in range(args.nworkers):
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, args=args).to(
            args.device)  # get a random model
        if not args.single_gpu:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
        net.train()
        if args.dataset == "CIFAR10":
            optimizer_net = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr_net,
                                            momentum=0.9,
                                            weight_decay=args.reg)
        else:
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        criterion = nn.CrossEntropyLoss().cuda()
        client = Client(each_worker, optimizer=optimizer_net, criterion=criterion,
                        each_worker_data=each_worker_data[each_worker],
                        each_worker_label=each_worker_label[each_worker],
                        indices_cl_class=indices_cl_classes[each_worker],
                        model=net, args=args)
        Clients.append(client)

    """initialize global model"""
    net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, args=args).to(args.device)  # get a random model
    if not args.single_gpu:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    net.train()
    if args.dataset == "CIFAR10":
        optimizer_net = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr_net, momentum=0.9,
                          weight_decay=args.reg)
    else:
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
    optimizer_net.zero_grad()
    criterion = nn.CrossEntropyLoss().cuda()

    """federated training"""
    for i in range(args.round):
        global_para = net.cpu().state_dict()
        for each_worker in range(args.nworkers):
            Clients[each_worker].model.load_state_dict(global_para)
        if args.baseline_method == "fedavg":
            net = fedavg(net, Clients, i, testloader, args, total_data_points=total_data_points)
        elif args.baseline_method == 'fedprox':
            net = fedprox(net, Clients, i, testloader, args, total_data_points=total_data_points)
        elif args.baseline_method == 'fednova':
            net = fednova(net, Clients, i, testloader, args, total_data_points=total_data_points)
        else:
            raise NotImplementedError
        with torch.no_grad():
            loss_avg, acc_avg = epoch_global('test', testloader, net, optimizer_net, criterion, args,
                                  aug=True if args.dsa else False)
            global_model_acc.append(acc_avg)
            global_model_loss.append(loss_avg)
            logger.info('%s Evaluate: val loss = %.6f val acc = %.4f' % (get_time(), loss_avg, acc_avg))

    logger.info("Test Loss: " + str(global_model_loss))
    logger.info("Test Acc: " + str(global_model_acc))
    logger.info(input_str)


if __name__ == '__main__':
    import shared_args

    parser = shared_args.baseline_args()

    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--inner_loop', type=int, default=500, help='inner loop')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    main(args)
