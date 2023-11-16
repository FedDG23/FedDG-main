import os
import argparse

def add_shared_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='my',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--round', type=int, default=20, help='global training round')

    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--space', type=str, default='p', choices=['p', 'wp'])
    parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--avg_w', action='store_true')

    parser.add_argument('--eval_all', action='store_true')

    parser.add_argument('--min_it', type=bool, default=False)
    parser.add_argument('--no_aug', type=bool, default=False)

    parser.add_argument('--force_save', action='store_true')

    parser.add_argument('--sg_batch', type=int, default=10)

    parser.add_argument('--rand_f', action='store_true')

    parser.add_argument('--logdir', type=str, default='./logged_files')

    parser.add_argument('--wait_eval', action='store_true')

    parser.add_argument('--idc_factor', type=int, default=1)

    parser.add_argument('--rand_gan_un', action='store_true')
    parser.add_argument('--rand_gan_con', action='store_true')

    parser.add_argument('--learn_g', action='store_true', help='generate syn first then aggregate G')
    parser.add_argument('--learn_model', action='store_true', help='use global model rather than random')

    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=5)

    parser.add_argument('--special_gan', default=None)

    parser.add_argument("--bias", help="way to assign data to workers", type=float, default=0.1)
    parser.add_argument("--beta", help="way to assign data to workers", type=float, default=0.1)
    parser.add_argument("--local_rank", help="index of gpu", type=int, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--gpu", help="index of gpu", type=str, default='0')
    parser.add_argument("--nworkers", type=int, help="number of workers", default=10)
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--fedDM', action='store_true')
    parser.add_argument("--update_g_interval", type=int, help="interval of updating generator", default=1)
    parser.add_argument("--frac", help="fraction of workers", type=float, default=1.0)
    parser.add_argument('--client_method', type=str, default='DM')
    parser.add_argument("--not_save_file", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument('--saved_model_dir', type=str, default='./logged_files')
    parser.add_argument("--recover_round", type=int, help="recover from ...", default=5)
    parser.add_argument("--save_all", action="store_true")
    return parser


def baseline_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='my',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--Iteration', type=int, default=20, help='training iterations')
    parser.add_argument('--round', type=int, default=20, help='global training round')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--space', type=str, default='p', choices=['p', 'wp'])
    parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--avg_w', action='store_true')

    parser.add_argument('--eval_all', action='store_true')

    parser.add_argument('--min_it', type=bool, default=False)
    parser.add_argument('--no_aug', type=bool, default=False)

    parser.add_argument('--force_save', action='store_true')

    parser.add_argument('--sg_batch', type=int, default=10)

    parser.add_argument('--rand_f', action='store_true')

    parser.add_argument('--logdir', type=str, default='./logged_files')

    parser.add_argument('--wait_eval', action='store_true')

    parser.add_argument('--idc_factor', type=int, default=1)

    parser.add_argument('--rand_gan_un', action='store_true')
    parser.add_argument('--rand_gan_con', action='store_true')

    parser.add_argument('--learn_g', action='store_true', help='generate syn first then aggregate G')
    parser.add_argument('--learn_g_after', action='store_true', help='aggregate G first then generate syn')
    parser.add_argument('--learn_model', action='store_true', help='use global model rather than random')

    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=5)


    parser.add_argument('--special_gan', default=None)

    parser.add_argument("--bias", help="way to assign data to workers", type=float, default=0.1)
    parser.add_argument("--beta", help="way to assign data to workers", type=float, default=0.1)
    parser.add_argument("--local_rank", help="index of gpu", type=int, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--gpu", help="index of gpu", type=str, default='0')
    parser.add_argument("--nworkers", type=int, help="number of workers", default=10)
    parser.add_argument("--baseline_method", type=str, help="baseline method: fedavg, fedprox, etc.", default="fedavg")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument("--frac", help="fraction of workers", type=float, default=1.0)
    parser.add_argument('--max_start_epoch', type=int, default=5)
    parser.add_argument('--single_gpu', action='store_true')

    return parser