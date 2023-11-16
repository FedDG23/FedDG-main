import torch
import torch.nn.functional as F
import numpy as np
import copy
import utils
# import wandb
import os
import torchvision
import gc
from tqdm import tqdm
import torch.nn as nn
import time
import logging

from utils import get_network, config, evaluate_synset, TensorDataset, DiffAugment, get_time


def build_dataset(ds, class_map, num_classes):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    return images_all, labels_all, indices_class


def build_client_dataset_dirichlet(ds, class_map, num_classes, args):
    min_size = 0
    min_require_size = 10
    n_parties = args.nworkers
    K = num_classes
    beta = args.beta
    y_train = np.array([class_map[torch.tensor(sample[1]).item()] for sample in ds])
    N = len(y_train)
    np.random.seed(2020)
    each_worker_data = [[] for _ in range(n_parties)]
    each_worker_label = [[] for _ in range(n_parties)]
    indices_cl_classes = [[[] for c in range(num_classes)] for _ in range(n_parties)]

    while min_size < min_require_size:
        # print(min_size)
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in tqdm(range(n_parties)):
        np.random.shuffle(idx_batch[j])
        each_worker_data[j] = [torch.unsqueeze(ds[i][0], dim=0) for i in idx_batch[j]]
        each_worker_label[j] = [class_map[torch.tensor(ds[i][1]).item()] for i in idx_batch[j]]

    for j in tqdm(range(n_parties)):
        for i, lab in enumerate(each_worker_label[j]):
            indices_cl_classes[j][lab].append(i)

    each_worker_data = [(torch.cat(each_worker, dim=0)).to('cpu') for each_worker in each_worker_data]
    each_worker_label = [torch.tensor(each_worker, dtype=torch.long, device="cpu") for each_worker in each_worker_label]
    return each_worker_data, each_worker_label, indices_cl_classes


def build_client_dataset(ds, class_map, num_classes, args):
    num_workers = args.nworkers
    indices_cl_classes = [[[] for c in range(num_classes)] for _ in range(num_workers)]
    print("BUILDING CLIENT DATASET")
    bias_weight = args.bias
    other_group_size = (1 - bias_weight) / (num_classes - 1)
    worker_per_group = num_workers / num_classes
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    for i in tqdm(range(len(ds))):
        data, label = ds[i]
        x = torch.unsqueeze(data, dim=0)
        y = class_map[torch.tensor(label).item()]
        upper_bound = y * (1 - bias_weight) / (num_classes - 1) + bias_weight
        lower_bound = y * (1 - bias_weight) / (num_classes - 1)
        rd = np.random.random_sample()
        if rd > upper_bound:
            worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y + 1)
        elif rd < lower_bound:
            worker_group = int(np.floor(rd / other_group_size))
        else:
            worker_group = y

        # assign a data point to a worker
        rd = np.random.random_sample()
        selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
        if args.bias == 0: selected_worker = np.random.randint(num_workers)
        each_worker_data[selected_worker].append(x)
        each_worker_label[selected_worker].append(y)

    for j in tqdm(range(num_workers)):
        for i, lab in enumerate(each_worker_label[j]):
            indices_cl_classes[j][lab].append(i)
    # concatenate the data for each worker
    each_worker_data = [(torch.cat(each_worker, dim=0)).to('cpu') for each_worker in each_worker_data]
    each_worker_label = [torch.tensor(each_worker, dtype=torch.long, device="cpu") for each_worker in each_worker_label]
    # random shuffle the workers
    random_order = np.random.permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    indices_cl_classes = [indices_cl_classes[i] for i in random_order]
    return each_worker_data, each_worker_label, indices_cl_classes


def prepare_latents(channel=3, client=None, im_size=(32, 32), zdim=512, G=None, class_map_inv={}, get_images=None,
                    each_worker=0, args=None, num_classes=10):
    with torch.no_grad():
        ''' initialize the synthetic data '''
        if client is not None:
            class_list = client.class_list
        else:
            class_list = np.array([c for c in range(num_classes)])
        if each_worker == -1:
            ipc = args.ipc * 5
        else:
            ipc = args.ipc
        label_syn = torch.tensor([i * np.ones(ipc, dtype=np.int64) for i in class_list], dtype=torch.long,
                                 requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.space == 'p':
            latents = torch.randn(size=(len(class_list) * ipc, channel, im_size[0], im_size[1]),
                                  dtype=torch.float,
                                  requires_grad=False, device=args.device)
            f_latents = None

        else:
            zs = torch.randn(len(class_list) * ipc, zdim, device=args.device, requires_grad=False)

            if "imagenet" in args.dataset:
                one_hot_dim = 1000
            elif args.dataset == "CIFAR10":
                one_hot_dim = 10
            elif args.dataset == "CIFAR100":
                one_hot_dim = 100
            if args.avg_w:
                G_labels = torch.zeros([label_syn.nelement(), one_hot_dim], device=args.device)
                G_labels[
                    torch.arange(0, label_syn.nelement(), dtype=torch.long), [class_map_inv[x.item()] for x in
                                                                              label_syn]] = 1
                new_latents = []
                for label in G_labels:
                    zs = torch.randn(1000, zdim, device=args.device)
                    ws = G.mapping(zs, torch.stack([label] * 1000))
                    w = torch.mean(ws, dim=0)
                    new_latents.append(w)
                latents = torch.stack(new_latents)
                del zs
                for _ in new_latents:
                    del _
                del new_latents


            else:
                G_labels = torch.zeros([label_syn.nelement(), one_hot_dim], device=args.device)
                G_labels[
                    torch.arange(0, label_syn.nelement(), dtype=torch.long), [class_map_inv[x.item()] for x in
                                                                              label_syn]] = 1
                if args.distributed and False:
                    latents = G.mapping(zs.to("cuda:" + (args.gpu)[1]), G_labels.to("cuda:" + (args.gpu)[1])).to(
                        args.device)
                else:
                    latents = G.mapping(zs, G_labels)
                del zs

            del G_labels

            ws = latents
            if args.layer is not None and args.layer != -1:
                f_latents = torch.cat(
                    [G.forward(split_ws, f_layer=args.layer, mode="to_f").detach() for split_ws in
                     torch.split(ws, args.sg_batch) if split_ws is not None])
                f_type = f_latents.dtype
                f_latents = f_latents.to(torch.float32).cpu()
                f_latents = torch.nan_to_num(f_latents, posinf=5.0, neginf=-5.0)
                f_latents = torch.clip(f_latents, min=-10, max=10)
                f_latents = f_latents.to(f_type).to(args.device)  # to('cuda:'+(args.gpu)[0])

                if args.rand_f:
                    f_latents = (torch.randn(f_latents.shape).to(args.device) * torch.std(
                        f_latents, dim=(1, 2, 3), keepdim=True) + torch.mean(f_latents, dim=(1, 2, 3), keepdim=True))

                f_latents = f_latents.to(f_type)
                f_latents.requires_grad_(True)
            else:
                f_latents = None

        if args.pix_init == 'real' and args.space == "p":
            print('initialize synthetic data from random real images')
            for i, c in enumerate(class_list):
                latents.data[i * ipc:(i + 1) * ipc] = get_images(c, ipc, each_worker).detach().data
        else:
            print('initialize synthetic data from random noise')

        latents = latents.detach().to(args.device).requires_grad_(True)

        return latents, f_latents, label_syn


def get_optimizer_img(latents=None, f_latents=None, G=None, args=None):
    if args.space == "wp" and (args.layer is not None and args.layer != -1):
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_w, momentum=0.5)
        optimizer_img.add_param_group({'params': f_latents, 'lr': args.lr_img, 'momentum': 0.5})
    else:
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_img, momentum=0.5)

    if args.learn_g or args.learn_g_after:
        G.requires_grad_(True)
        optimizer_img.add_param_group({'params': G.parameters(), 'lr': args.lr_g, 'momentum': 0.5})

    optimizer_img.zero_grad()

    return optimizer_img


def get_eval_lrs(args):
    eval_pool_dict = {
        args.model: 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict


def eval_loop(latents=None, f_latents=None, label_syn=None, G=None, best_acc={}, best_std={}, testloader=None,
              model_eval_pool=[], it=0, channel=3, num_classes=10, im_size=(32, 32), args=None):
    curr_acc_dict = {}
    max_acc_dict = {}

    curr_std_dict = {}
    max_std_dict = {}

    eval_pool_dict = get_eval_lrs(args)

    save_this_it = False

    for model_eval in model_eval_pool:

        if model_eval != args.model and args.wait_eval and it != args.Iteration:
            continue
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
            args.model, model_eval, it))

        accs_test = []
        accs_train = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, width=args.width, depth=args.depth,
                                   dist=False, args=args).to(args.device)  # get a random model
            eval_lats = latents
            eval_labs = label_syn
            image_syn = latents
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                eval_labs.detach())  # avoid any unaware modification

            if args.space == "wp":
                if args.layer != -1:
                    with torch.no_grad():
                        image_syn_eval = torch.cat(
                            [latent_to_im(G, (image_syn_eval_split, f_latents_split), args=args).detach() for
                             image_syn_eval_split, f_latents_split, label_syn_split in
                             zip(torch.split(image_syn_eval, args.sg_batch), torch.split(f_latents, args.sg_batch),
                                 torch.split(label_syn, args.sg_batch))])
                else:
                    with torch.no_grad():
                        image_syn_eval = torch.cat(
                            [latent_to_im(G, (image_syn_eval_split, None), args=args).detach() for
                             image_syn_eval_split, label_syn_split in
                             zip(torch.split(image_syn_eval, args.sg_batch), torch.split(label_syn, args.sg_batch))])
            args.lr_net = eval_pool_dict[model_eval]
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader,
                                                     args=args, aug=True)
            del _
            del net_eval
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        print(accs_test)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(np.max(accs_test, axis=1))
        acc_test_std = np.std(np.max(accs_test, axis=1))
        best_dict_str = "{}".format(model_eval)
        if acc_test_mean > best_acc[best_dict_str]:
            best_acc[best_dict_str] = acc_test_mean
            best_std[best_dict_str] = acc_test_std
            save_this_it = True

        curr_acc_dict[best_dict_str] = acc_test_mean
        curr_std_dict[best_dict_str] = acc_test_std

        max_acc_dict[best_dict_str] = best_acc[best_dict_str]
        max_std_dict[best_dict_str] = best_std[best_dict_str]

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
            len(accs_test[:, -1]), model_eval, acc_test_mean, np.std(np.max(accs_test, axis=1))))
        logging.info('Step: %d, Accuracy/%s: %.4f' % (it, model_eval, acc_test_mean))
        logging.info('Step: %d, Max_Accuracy/%s: %.4f' % (it, model_eval, best_acc[best_dict_str]))
        logging.info('Step: %d, Std/%s: %.4f' % (it, model_eval, acc_test_std))
        logging.info('Step: %d, Max_Std/%s: %.4f' % (it, model_eval, best_std[best_dict_str]))

    logging.info(
        'Step: %d, Accuracy/Avg_All/%s: %.4f' % (it, model_eval, np.mean(np.array(list(curr_acc_dict.values())))))
    logging.info(
        'Step: %d, Max_Accuracy/Avg_All/%s: %.4f' % (it, model_eval, np.mean(np.array(list(curr_std_dict.values())))))
    logging.info('Step: %d, Std/Avg_All/%s: %.4f' % (it, model_eval, np.mean(np.array(list(max_acc_dict.values())))))
    logging.info(
        'Step: %d, Max_Std/Avg_All/%s: %.4f' % (it, model_eval, np.mean(np.array(list(max_std_dict.values())))))

    return save_this_it


def load_sgxl(res, args=None, is_global=False):
    import sys
    import os
    p = os.path.join("stylegan_xl")
    if p not in sys.path:
        sys.path.append(p)
    import dnnlib
    import legacy
    from sg_forward import StyleGAN_Wrapper
    device = args.device  # torch.device('cuda:'+(args.gpu)[0])
    if args.special_gan is not None:
        if args.special_gan == "ffhq":
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq{}.pkl".format(res)
            network_pkl = "../stylegan_xl/ffhq{}.pkl".format(res)
            key = "G_ema"
        elif args.special_gan == "pokemon":
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon{}.pkl".format(res)
            network_pkl = "../stylegan_xl/pokemon{}.pkl".format(
                res)
            key = "G_ema"

    elif "imagenet" in args.dataset:
        if args.rand_gan_con:
            network_pkl = "../stylegan_xl/random_conditional_{}.pkl".format(res)
            key = "G"
        elif args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_{}.pkl".format(res)
            key = "G"
        else:
            network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet{}.pkl".format(
                res)
            key = "G_ema"
    elif args.dataset == "CIFAR10":
        if args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_32.pkl"
            key = "G"
        else:
            network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl"
            key = "G_ema"
    elif args.dataset == "CIFAR100":
        if args.rand_gan_con:
            network_pkl = "../stylegan_xl/random_conditional_32.pkl"
            key = "G"
        elif args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_32.pkl"
            key = "G"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)[key]
        G = G.eval().requires_grad_(False).to(device)

    z_dim = G.z_dim
    w_dim = G.w_dim
    num_ws = G.num_ws

    G.eval()
    mapping = G.mapping
    G = StyleGAN_Wrapper(G)
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        G = nn.DataParallel(G, device_ids=[args.local_rank])
        mapping = nn.DataParallel(mapping, device_ids=[args.local_rank])

    G.mapping = mapping

    return G, z_dim, w_dim, num_ws


def latent_to_im(G, latents, args=None):
    if args.space == "p":
        return latents

    mean, std = config.mean, config.std
    mean = torch.tensor(mean, device=args.device).reshape(1, 3, 1, 1)
    std = torch.tensor(std, device=args.device).reshape(1, 3, 1, 1)

    if "imagenet" in args.dataset:
        class_map = {i: x for i, x in enumerate(config.img_net_classes)}

        if args.space == "p":
            im = latents

        elif args.space == "wp":
            if args.layer is None or args.layer == -1:
                im = G(latents[0], mode="wp")
            else:
                im = G(latents[0], latents[1], args.layer, mode="from_f")
        im = (im + 1) / 2
        im = (im - mean) / std

    elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        if args.space == "p":
            im = latents
        elif args.space == "wp":
            if args.layer is None or args.layer == -1:
                im = G(latents[0], mode="wp")
            else:
                im = G(latents[0], latents[1], args.layer, mode="from_f")

        im = (im + 1) / 2
        im = (im - mean) / std

    return im


def image_logging(latents=None, f_latents=None, label_syn=None, G=None, it=None, save_this_it=None, args=None):
    with torch.no_grad():
        image_syn = latents.to(args.device)

        if args.space == "wp":
            with torch.no_grad():
                if args.layer is None or args.layer == -1:
                    image_syn = latent_to_im(G, (image_syn.detach(), None), args=args)
                else:
                    image_syn = torch.cat(
                        [latent_to_im(G, (image_syn_split.detach(), f_latents_split.detach()), args=args).detach() for
                         image_syn_split, f_latents_split, label_syn_split in
                         zip(torch.split(image_syn, args.sg_batch),
                             torch.split(f_latents, args.sg_batch),
                             torch.split(label_syn, args.sg_batch))])

        save_dir = os.path.join(args.logdir, args.dataset, args.save_path)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(image_syn.cpu(), os.path.join(save_dir, "images_{0:05d}.pt".format(it)))
        torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{0:05d}.pt".format(it)))

        if save_this_it:
            torch.save(image_syn.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
            torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))


        if args.ipc < 50 or args.force_save:

            upsampled = image_syn
            if "imagenet" not in args.dataset:
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)

            for clip_val in []:
                upsampled = torch.clip(image_syn, min=-clip_val, max=clip_val)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)

            for clip_val in [2.5]:
                std = torch.std(image_syn)
                mean = torch.mean(image_syn)
                upsampled = torch.clip(image_syn, min=mean - clip_val * std, max=mean + clip_val * std)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)

    del upsampled, grid


def gan_backward(latents=None, f_latents=None, image_syn=None, G=None, args=None):
    if args.layer != -1:
        f_latents.grad = None
        latents_grad_list = []
        f_latents_grad_list = []
        for latents_split, f_latents_split, dLdx_split in zip(torch.split(latents, args.sg_batch),
                                                              torch.split(f_latents, args.sg_batch),
                                                              torch.split(image_syn.grad, args.sg_batch)):
            latents_detached = latents_split.detach().clone().requires_grad_(True)
            f_latents_detached = f_latents_split.detach().clone().requires_grad_(True)
            syn_images = latent_to_im(G=G, latents=(latents_detached, f_latents_detached), args=args)
            latents_detached.retain_grad()
            syn_images.backward((dLdx_split,))

            latents_grad_list.append(latents_detached.grad)
            f_latents_grad_list.append(f_latents_detached.grad)

            del syn_images
            del latents_split
            del f_latents_split
            del dLdx_split
            del f_latents_detached
            del latents_detached

            gc.collect()

        latents.grad = torch.cat(latents_grad_list)
        del latents_grad_list
        f_latents.grad = torch.cat(f_latents_grad_list)
        del f_latents_grad_list

    else:
        latents_grad_list = []
        for latents_split, dLdx_split in zip(torch.split(latents, args.sg_batch), torch.split(image_syn.grad, args.sg_batch)):
            latents_detached = latents_split.detach().clone().requires_grad_(True)
            syn_images = latent_to_im(G=G, latents=(latents_detached, None), args=args)
            latents_detached.retain_grad()
            syn_images.backward((dLdx_split,))

            latents_grad_list.append(latents_detached.grad)

            del syn_images
            del latents_split
            del dLdx_split
            del latents_detached

            gc.collect()

        latents.grad = torch.cat(latents_grad_list)
        del latents_grad_list


def random_perturb(net, new=None):
    if new is None:
        for p in net.parameters():
            gauss = torch.normal(mean=torch.zeros_like(p), std=1)
            if p.grad is None:
                p.grad = gauss
            else:
                p.grad.data.copy_(gauss.data)

        norm = torch.norm(
            torch.stack([(p.grad).norm(p=2) for p in net.parameters() if p.grad is not None]),
            p=2)
        # clip_coef = max_norm / (total_norm + 1e-6)

        with torch.no_grad():
            scale = 5.0 / (norm + 1e-12)
            scale = torch.clamp(scale, max=1.0)
            for p in net.parameters():
                if p.grad is None: continue
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    else:
        norm = torch.norm(
            torch.stack([(q.data).norm(p=2) for q in new.parameters()]), p=2)
        with torch.no_grad():
            # scale = 1 / (norm + 1e-12)
            # scale = 1.0
            for p, q in zip(net.parameters(), new.parameters()):
                # e_w = 1.0 * q.data * scale.to(q)
                e_w = 1.0 * q.data
                p.add_(e_w)

    net.zero_grad()
    return net


def avg_weight(nets_list, weights=None):
    # global_para = global_net.cpu().state_dict()
    global_net = copy.deepcopy(nets_list[0])
    num_workers = len(nets_list)
    if weights == None:
        weights = [1 / num_workers for _ in range(num_workers)]
    for each_worker in range(len(nets_list)):
        net_para = nets_list[each_worker]
        if each_worker == 0:
            for key, param in enumerate(global_net):
                param = net_para[key] * weights[each_worker]
        else:
            for key, param in enumerate(global_net):
                param += net_para[key] * weights[each_worker]
    return global_net
