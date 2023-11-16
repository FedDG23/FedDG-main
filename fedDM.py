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


def get_fname(args):
    parserString = "FedDM"+str(args.dataset)+"_Layer"+str(args.layer)+"_Iteration"+str(args.Iteration)+"_nworkers"+str(args.nworkers)+\
                   "_beta"+str(args.beta)+"_ipc"+str(args.ipc)+"_round"+str(args.round)
    return parserString


def main(args):

    input_str = ' '.join(sys.argv)
    print(input_str)
    print(args.local_rank)
    logger = get_logger('log/' + get_fname(args) + time.strftime("%Y%m%d-%H%M%S") + ".log", distributed_rank=args.local_rank)
    # exit(0)
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

    args.save_path = os.path.join(args.save_path, "feddm", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.distributed = torch.cuda.device_count() > 1

    each_worker_data, each_worker_label, indices_cl_classes = build_client_dataset_dirichlet(dst_train, class_map, num_classes, args)
    for each_client in range(args.nworkers):
        print(each_client, " ", len(each_worker_data[each_client]))
        print([len(indices_cl_classes[each_client][c]) for c in range(10)])
        print("="*100)

    def get_images(c, n, each_worker):
        idx_shuffle = np.random.permutation(indices_cl_classes[each_worker][c])[:n]
        return each_worker_data[each_worker][idx_shuffle].cuda(non_blocking=True)

    global_model_acc = dict() # record performances of all experiments
    global_model_loss = dict()
    global_model_acc['Origin'] = []
    global_model_loss['Origin'] = []
    for key in model_eval_pool:
        global_model_acc[key] = []
        global_model_loss[key] = []

    """initialize clients"""
    Clients = []
    for each_worker in range(args.nworkers):
        if args.space == 'p':
            G, zdim = None, None
        elif args.space == 'wp':
            G, zdim, w_dim, num_ws = load_sgxl(args.res, args)
        else:
            exit("unknown space: %s"%args.space)
        criterion = nn.CrossEntropyLoss().cuda()
        client = Client(each_worker, G, criterion=criterion, each_worker_data=each_worker_data[each_worker],
                        each_worker_label=each_worker_label[each_worker],
                        indices_cl_class=indices_cl_classes[each_worker], args=args)
        Clients.append(client)

    test_nets = []
    test_optimizers = []
    for real_model in model_eval_pool:
        real_net = get_network(real_model, channel, num_classes, im_size, depth=args.depth, width=args.width, args=args).to(
            args.device)  # get a random model
        real_net = torch.nn.parallel.DistributedDataParallel(real_net, device_ids=[args.local_rank])
        real_net.train()
        optimizer_real_net = torch.optim.SGD(filter(lambda p: p.requires_grad, real_net.parameters()), lr=args.lr_net,
                                        momentum=0.9, weight_decay=args.reg)
        optimizer_real_net.zero_grad()
        test_nets.append(real_net)
        test_optimizers.append(optimizer_real_net)

    """initialize global model"""
    net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, args=args).to(args.device)  # get a random model
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    net.train()
    if False and args.dataset == "CIFAR10":
        optimizer_net = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr_net, momentum=0.9,
                          weight_decay=args.reg)
    else:
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
    optimizer_net.zero_grad()
    criterion = nn.CrossEntropyLoss().cuda()

    if args.recover:
        save_dir = os.path.join(args.saved_model_dir, "local_model")
        net.load_state_dict(torch.load(os.path.join(save_dir, "model_{0:05d}.pt".format(args.recover_round))))
        net = net.to(args.device)
        net.eval()
        for net_name, real_net, optimizer_real_net in zip(model_eval_pool, test_nets, test_optimizers):
            save_dir = os.path.join(args.saved_model_dir, net_name)
            real_net.load_state_dict(torch.load(os.path.join(save_dir, "model_{0:05d}.pt".format(args.recover_round))))
            real_net = real_net.to(args.device)
            real_net.eval()

    """federated training"""
    for i in range(args.round):
        logger.info("="*50 + " Round: " + str(i) + " " + "="*50)
        image_syn_trains = []
        label_syn_trains = []
        m = max(int(args.frac * args.nworkers), 1)
        idxs_users = np.random.choice(range(args.nworkers), m, replace=False)
        logging.info('\nChoosing users {}'.format(' '.join(map(str, idxs_users))))
        if args.recover:
            if i <= args.recover_round:
                continue
        for each_worker in idxs_users:
            logger.info(input_str)
            print('\n')
            logger.info("="*50 + " Client: " + str(each_worker) + " " + "="*50)
            latents, f_latents, label_syn = prepare_latents(channel=channel, client=Clients[each_worker], im_size=im_size,
                                                            zdim=zdim, G=Clients[each_worker].generator,
                                                            class_map_inv=class_map_inv,
                                                            get_images=get_images, each_worker=each_worker, args=args)
            optimizer_img = get_optimizer_img(latents=latents, f_latents=f_latents, G=Clients[each_worker].generator,
                                              args=args)
            Clients[each_worker].latents = latents
            Clients[each_worker].f_latents = f_latents
            Clients[each_worker].optimizer = optimizer_img
            image_syn_train, label_syn_train = client_update(args, Clients[each_worker], testloader, model_eval_pool,
                                                             channel, im_size, num_classes, i, net)
            image_syn_trains.append(image_syn_train)
            label_syn_trains.append(label_syn_train)
            if (i + 1) % 5 == 0:
                logging.info('=' * 50 + 'Saving' + '=' * 50)
                save_dir = os.path.join(args.logdir, args.dataset, args.save_path, str(each_worker))

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_syn_train.cpu(), os.path.join(save_dir, "images_{0:05d}.pt".format(i)))
                torch.save(label_syn_train.cpu(), os.path.join(save_dir, "labels_{0:05d}.pt".format(i)))

        """update global model"""
        image_syn_trains = torch.cat(image_syn_trains)
        label_syn_trains = torch.cat(label_syn_trains)
        dst_syn_train = TensorDataset(image_syn_trains, label_syn_trains)
        trainsampler = torch.utils.data.distributed.DistributedSampler(dst_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, sampler=trainsampler,#shuffle=True,
                                                  num_workers=0)
        for il in range(args.inner_loop):
            loss_avg, acc_avg = epoch_global('train', trainloader, net, optimizer_net, criterion,
                                             args, aug=True if args.dsa else False)
            if il % 100 == 0:
                logger.info(
                    '%s Evaluate_%s_%02d: train loss = %.6f train acc = %.4f' % (args.model, get_time(), il, loss_avg, acc_avg))
            for net_name, real_net, optimizer_real_net in zip(model_eval_pool, test_nets, test_optimizers):
                loss_avg, acc_avg = epoch_global('train', trainloader, real_net, optimizer_real_net, criterion,
                                                 args, aug=True if args.dsa else False)
                if il % 100 == 0:
                    logger.info(
                        '%s Evaluate_%s_%02d: train loss = %.6f train acc = %.4f' % (net_name, get_time(), il, loss_avg, acc_avg))
        with torch.no_grad():
            if (i + 1) % 5 == 0:
                logging.info('=' * 50 + 'Saving model' + '=' * 50)
                save_dir = os.path.join(args.logdir, args.dataset, args.save_path, "local_model")

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(net.cpu().state_dict(), os.path.join(save_dir, "model_{0:05d}.pt".format(i)))
            loss_avg, acc_avg = epoch_global('test', testloader, net, optimizer_net, criterion, args,
                                             aug=True if args.dsa else False)
            global_model_acc['Origin'].append(acc_avg)
            global_model_loss['Origin'].append(loss_avg)
            logger.info('%s Evaluate_%s: val loss = %.6f val acc = %.4f' % (args.model, get_time(), loss_avg, acc_avg))
            for net_name, real_net, optimizer_real_net in zip(model_eval_pool, test_nets, test_optimizers):
                if (i + 1) % 5 == 0:
                    save_dir = os.path.join(args.logdir, args.dataset, args.save_path, net_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(real_net.cpu().state_dict(), os.path.join(save_dir, "model_{0:05d}.pt".format(i)))
                loss_avg, acc_avg = epoch_global('test', testloader, real_net, optimizer_real_net, criterion, args,
                                                 aug=True if args.dsa else False)
                global_model_acc[net_name].append(acc_avg)
                global_model_loss[net_name].append(loss_avg)
                logger.info('%s Evaluate_%s: val loss = %.6f val acc = %.4f' % (net_name, get_time(), loss_avg, acc_avg))

    logger.info("Test Loss: " + str(global_model_loss))
    logger.info("Test Acc: " + str(global_model_acc))
    logger.info(input_str)


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()

    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=1, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_g', type=float, default=0.01, help='learning rate for gan weights')

    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--inner_loop', type=int, default=1000, help='inner loop')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--separate_generate', action='store_true')
    args = parser.parse_args()
    main(args)
