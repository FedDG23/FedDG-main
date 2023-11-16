import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug
# import wandb
from tqdm import tqdm
import torchvision
import random
import gc

from glad_utils import *


def client_update(args, client, testloader, model_eval_pool, channel, im_size, num_classes, round=20, global_net=None):
    model_eval_pool = [args.model]
    G = client.generator
    latents = client.latents
    f_latents = client.f_latents
    images_all = client.each_worker_data
    labels_all = client.each_worker_label
    indices_class = client.indices_cl_class
    optimizer_img = client.optimizer

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()

    def get_images(c, n):
        if len(indices_class[c]) >= n:
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
        else:
            sampled_idx = np.random.choice(indices_class[c], n-len(indices_class[c]), replace=True)
            idx_shuffle = np.concatenate((indices_class[c], sampled_idx), axis=None)
        return images_all[idx_shuffle].cuda()


    ''' initialize the synthetic data '''
    # image_syn = torch.randn(size=(len(client.class_list)*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in client.class_list], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    print('%s training begins'%get_time())

    best_acc = {"{}".format(m): 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    save_this_it = False
    for it in range(args.Iteration+1):

        ''' Train synthetic data '''
        if global_net is not None and round > 0:
            net = random_perturb(copy.deepcopy(global_net))
        else:
            net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, args=args).to(args.device)  # get a random model
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
        net.train()
        if args.client_method != "DC":
            for param in list(net.parameters()):
                param.requires_grad = False
        embed = net.module.embed if torch.cuda.device_count() > 0 else net.embed # for GPU parallel
        if args.fedDM and round > 0:
            new_net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width,
                              args=args).to(args.device)  # get a random model
            new_net = torch.nn.parallel.DistributedDataParallel(new_net, device_ids=[args.local_rank])
            new_net.train()
            for param in list(new_net.parameters()):
                param.requires_grad = False

            new_embed = new_net.module.embed if torch.cuda.device_count() > 0 else new_net.embed  # for GPU parallel

        loss_avg = 0

        if args.space == "wp":
            if args.layer != -1:
                with torch.no_grad():
                    image_syn_w_grad = torch.cat([latent_to_im(G, (syn_image_split, f_latents_split), args) for
                                                  syn_image_split, f_latents_split, label_syn_split in
                                                  zip(torch.split(latents, args.sg_batch),
                                                      torch.split(f_latents, args.sg_batch),
                                                      torch.split(label_syn, args.sg_batch))])
            else:
                with torch.no_grad():
                    image_syn_w_grad = torch.cat([latent_to_im(G, (syn_image_split, None), args) for
                                                  syn_image_split, label_syn_split in
                                                  zip(torch.split(latents, args.sg_batch),
                                                      torch.split(label_syn, args.sg_batch))])
        else:
            image_syn_w_grad = latents

        if args.space == "wp":
            image_syn = image_syn_w_grad.detach()
            image_syn.requires_grad_(True)
        else:
            image_syn = image_syn_w_grad


        ''' update synthetic data '''
        if 'BN' not in args.model: # for ConvNet
            loss = torch.tensor(0.0).cuda()
            for i, c in enumerate(client.class_list):
                img_real = get_images(c, args.batch_real).cuda()# .to(args.device)
                img_syn = image_syn[i*args.ipc:(i+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    # print(img_real.shape)
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                if args.client_method == "DC":
                    criterion = client.criterion
                    output_real = net(img_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, list(net.parameters()))
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, list(net.parameters()), create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)
                    del img_real, output_real, loss_real, gw_real, output_syn, loss_syn, gw_syn

                else:
                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                    if args.fedDM and round > 0:
                        output_real_logits = new_embed(img_real).detach()
                        output_syn_logits = new_embed(img_syn)
                        loss += torch.sum((torch.mean(output_real_logits, dim=0) - torch.mean(output_syn_logits, dim=0))**2)

        else:
            images_real_all = []
            images_syn_all = []
            loss = torch.tensor(0.0).cuda() #.to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)

            output_real = embed(images_real_all).detach()
            output_syn = embed(images_syn_all)

            loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

        optimizer_img.zero_grad()
        loss.backward()

        if args.space == "wp":
            gan_backward(latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

        else:
            latents.grad = image_syn.grad.detach().clone()

        optimizer_img.step()
        loss_avg += loss.item()

        loss_avg /= len(client.class_list)

        if it%10 == 0:
            logging.info('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

        if it == args.Iteration: # only record the final results
            data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%s_%dipc.pt'%("fedDM", args.dataset, "client"+str(client.client_id), args.model, args.ipc)))

        del net

    client.generator = G
    client.latents = latents
    client.f_latents = f_latents

    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())

    return image_syn_train, label_syn_train


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()

    parser.add_argument('--lr_img', type=float, default=10, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=.01, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for gan weights')

    args = parser.parse_args()

