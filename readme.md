# FedDG-main

To setup an environment, please run

```bash
conda env create -f environment.yml
```

To train on CIFAR-10, please use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 fed_main.py --dataset=CIFAR10 --space=wp --layer=5 --data_path=./data --eval_mode CIFAR --beta 0.5 --batch_real 256 --batch_train 256 --batch_test 128 --ipc=10 --nworkers 10 --round 20 --Iteration 100 --Iteration_g 20
```

To train on ImageFruit, please use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 fed_main.py --dataset=imagenet-fruits --space=wp --layer=12 --data_path=./data --eval_mode imagenet --beta 0.5 --batch_real 32 --batch_train 32 --batch_test 32 --ipc=10 --nworkers 10 --round 20 --Iteration 100 --Iteration_g 20
```