import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset,  match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug

import logging
import copy
import random
from reparam_module import ReparamModule
import torch.utils.data
import warnings
import gc

from glad_utils import *

import time


warnings.filterwarnings("ignore", category=DeprecationWarning)


def server_update(args, latents, f_latents, starting_params, target_params, G, testloader, channel, im_size, num_classes, round):

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()

    model_eval_pool = get_eval_pool('my', args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files


    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = len(args.gpu) > 1

    label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False,
                             device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
    syn_lr = torch.tensor(args.lr_teacher, requires_grad=True).to(args.device)

    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    optimizer_img = get_optimizer_img(latents=latents, f_latents=f_latents, G=G, args=args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    for it in range(0, args.Iteration+1):
        student_net = get_network(args.model, channel, num_classes, im_size, width=args.width, depth=args.depth, dist=False, args=args).to(args.device)  # get a random model

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        student_net = ReparamModule(student_net)

        gradient_sum = torch.zeros(starting_params.shape).requires_grad_(False).to(args.device)

        param_dist = torch.tensor(0.0).to(args.device)

        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        if args.distributed:
            student_net = torch.nn.parallel.DistributedDataParallel(student_net, device_ids=[args.local_rank])

        student_net.train()

        syn_images = latents[:]
        if args.space == "wp":
            if args.layer != -1:
                with torch.no_grad():
                    syn_images = torch.cat([latent_to_im(G, (syn_image_split.detach(), f_latents_split.detach()),
                                                        args=args).detach() for
                                           syn_image_split, f_latents_split, label_syn_split in
                                           zip(torch.split(syn_images, args.sg_batch),
                                               torch.split(f_latents, args.sg_batch),
                                               torch.split(label_syn, args.sg_batch))])
            else:
                with torch.no_grad():
                    syn_images = torch.cat([latent_to_im(G, (syn_image_split.detach(), None), args=args).detach() for
                                           syn_image_split, label_syn_split in
                                           zip(torch.split(syn_images, args.sg_batch),
                                               torch.split(label_syn, args.sg_batch))])
            syn_images.requires_grad_(True)

        image_syn = syn_images.detach()

        y_hat = label_syn
        x_list = []
        y_list = []
        indices_chunks = []
        indices_chunks_copy = []
        original_x_list = []
        gc.collect()

        syn_label_grad = torch.zeros(label_syn.shape).to(args.device).requires_grad_(False)
        syn_images_grad = torch.zeros(syn_images.shape).requires_grad_(False).to(args.device)

        for il in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()
            indices_chunks_copy.append(these_indices.clone())

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            original_x_list.append(x)

            x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            x_list.append(x.clone())
            y_list.append(this_y.clone())

            forward_params = student_params[-1]

            forward_params = copy.deepcopy(forward_params.detach()).requires_grad_(True)

            if args.distributed:
                forward_params_expanded = forward_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params_expanded = forward_params

            x = student_net(x, flat_param=forward_params_expanded)

            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, forward_params, create_graph=True, retain_graph=True)[0]
            student_params.append(forward_params - syn_lr.item() * grad.detach().clone())
            gradient_sum = gradient_sum + grad.detach().clone()

        for il in range(args.syn_steps):
            w = student_params[il]

            if args.distributed:
                w_expanded = w.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                w_expanded = w

            output = student_net(x_list[il], flat_param=w_expanded)

            if args.batch_syn:
                ce_loss = criterion(output, y_list[il])
            else:
                ce_loss = criterion(output, y_hat)

            grad = torch.autograd.grad(ce_loss, w, create_graph=True, retain_graph=True)[0]

            # Square term gradients.
            square_term = syn_lr.item() ** 2 * (grad @ grad)
            single_term = 2 * syn_lr.item() * grad @ (
                        syn_lr.item() * (gradient_sum - grad.detach().clone()) - starting_params + target_params)

            per_batch_loss = (square_term + single_term) / param_dist
            gradients = torch.autograd.grad(per_batch_loss, original_x_list[il], retain_graph=False)[0]

            with torch.no_grad():
                syn_images_grad[indices_chunks_copy[il]] += gradients

        # ---------end of computing input image gradients and learning rates--------------

        del w, output, ce_loss, grad, square_term, single_term, per_batch_loss, gradients, student_net, w_expanded, forward_params, forward_params_expanded

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        syn_lr.requires_grad_(True)
        grand_loss = starting_params - syn_lr * gradient_sum - target_params
        grand_loss = grand_loss.dot(grand_loss)
        grand_loss = grand_loss / param_dist

        lr_grad = torch.autograd.grad(grand_loss, syn_lr)[0]
        syn_lr.grad = lr_grad

        optimizer_lr.step()
        optimizer_lr.zero_grad()

        image_syn.requires_grad_(True)

        image_syn.grad = syn_images_grad.detach().clone()

        del syn_images_grad
        del lr_grad

        for _ in student_params:
            del _
        for _ in x_list:
            del _
        for _ in y_list:
            del _

        torch.cuda.empty_cache()

        gc.collect()

        if args.space == "wp":
            # this method works in-line and back-props gradients to latents and f_latents
            gan_backward(latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

        else:
            latents.grad = image_syn.grad.detach().clone()

        optimizer_img.step()
        optimizer_img.zero_grad()


        if it%10 == 0:
            logging.info('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

        if it == args.Iteration:
            data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%dipc.pt'%(args.dataset, args.model, args.ipc)))

    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())

    return image_syn_train, label_syn_train


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn data')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--max_start_epoch', type=int, default=5)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_experts', type=int, default=None)
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')

    parser.add_argument('--lr_img', type=float, default=10000, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=10, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_lr', type=float, default=1e-06, help='learning rate learning rate')
    parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for gan weights')

    args = parser.parse_args()


