import argparse
import os
import random
import shutil
import time
import math
import warnings
import numpy as np
import builtins
import tqdm
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import InterpolationMode
from copy import deepcopy
from collections import OrderedDict
from utils_FKD import RandomResizedCrop_FKD, RandomHorizontalFlip_FKD
from utils_FKD import ImageFolder_FKD, Compose_FKD
from utils_FKD import Soft_CrossEntropy, Recover_soft_label
from utils_FKD import mixup_cutmix,my_mixup_cutmix
from resnet import  ResNet,Bottleneck

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with FKD Scheme')
parser.add_argument('--data', metavar='DIR',
                    # help='path to dataset', default="/datc/data/imagenet")
                    help='path to dataset', default="/work/data/imagenet")
                    # help='path to dataset', default="/work/data/small_imagenet")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 240], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:10012', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
# parser.add_argument('--gpu', default="0", type=int,
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--num_crops', default=4, type=int,
                    help='number of crops in each image, 1 is the standard training')
# parser.add_argument('--softlabel_path', default='/datc/data/FKD_soft_label_500_crops_marginal_smoothing_k_5/imagenet', type=str, metavar='PATH',
parser.add_argument('--softlabel_path', default='/work/zhangzherui/data/FKD_soft_label_500_crops_marginal_smoothing_k_5/imagenet', type=str, metavar='PATH',
                    help='path to soft label files (default: none)')
parser.add_argument("--temp", type=float, default=1.0,
                    help="temperature on student during training (defautl: 1.0)")
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--save_checkpoint_path', default='./FKD_checkpoints_output', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--soft_label_type', default='marginal_smoothing_k5', type=str, metavar='TYPE',
                    help='(1) ori; (2) hard; (3) smoothing; (4) marginal_smoothing_k5; (5) marginal_smoothing_k10; (6) marginal_renorm_k5')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes.')

# mixup and cutmix parameters
parser.add_argument('--mixup_cutmix', default=False, action='store_true',
                    help='use mixup and cutmix data augmentation')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--mixup_cutmix_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled. Mixup only: set to 1.0, Cutmix only: set to 0.0')

best_acc1 = 0



class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        for p in self.ema.parameters():
            p.requires_grad_(False)
    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)



def main():
    args = parser.parse_args()

    if not os.path.exists(args.save_checkpoint_path):
        os.makedirs(args.save_checkpoint_path)

    # convert to TRUE number of loading-images since we use multiple crops from the same image within a minbatch
    args.batch_size = math.ceil(args.batch_size / args.num_crops)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("args.distributed = ", args.distributed)
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node = ", ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=False, num_classes=args.num_classes)

    model = ResNet(Bottleneck,[3,4,6,3])
    weight = torch.load("./FerKD_ResNet50.pt", map_location="cpu")
    weight = {key.replace('module.', ''): value for key, value in weight.items()}
    print(model.load_state_dict(weight, strict=False))
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model_ema = ModelEma(model)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],)
        else:
            model.cuda()
            model_ema = ModelEma(model)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model, )
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_ema = ModelEma(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer

    criterion_sce = Soft_CrossEntropy()
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    # args.lr = args.lr * ((args.batch_size * 1.0) / 512)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = ImageFolder_FKD(
        num_crops=args.num_crops,
        softlabel_path=args.softlabel_path,
        root=traindir,
        transform=Compose_FKD(transforms=[
            RandomResizedCrop_FKD(size=224,
                            interpolation='bilinear'), 
            RandomHorizontalFlip_FKD(),

            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_single_crop = ImageFolder_FKD(
        num_crops=1,
        softlabel_path=args.softlabel_path,
        root=traindir,
        transform=Compose_FKD(transforms=[
            RandomResizedCrop_FKD(size=224,
                            interpolation='bilinear'), 
            RandomHorizontalFlip_FKD(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,collate_fn=my_collate)
    train_loader = tqdm.tqdm(train_loader, file=sys.stdout)
    train_loader_single_crop = torch.utils.data.DataLoader(
        train_dataset_single_crop, batch_size=args.batch_size*args.num_crops, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,collate_fn=my_collate)

    train_loader_ema = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_loader = tqdm.tqdm(val_loader, file=sys.stdout)
    if args.evaluate:
        validate(val_loader, model, criterion_ce, args)
        return

    # for resume
    if args.start_epoch !=0 and args.start_epoch < (args.epochs-args.num_crops):
        args.start_epoch = args.start_epoch + args.num_crops - 1
    count = 0
    args.is_ema = False
    for epoch in range(args.start_epoch, args.epochs, args.num_crops):
        count += 1
        if count % 3 == 0:
            args.is_ema = True
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        # for fine-grained evaluation at last a few epochs
        # for name, param in model.named_parameters():
        #     if 'branch' in name:
        #         param.requires_grad = False
        # train_after(train_loader_single_crop, model, criterion_sce, optimizer, epoch, args, model_ema)
        if epoch >= (args.epochs-args.num_crops):
            for name, param in model.named_parameters():
                if 'branch' in name:
                    param.requires_grad = False
            start_epoch = epoch
            for epoch in range(start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                adjust_learning_rate(optimizer, epoch, args)

                # train for one epoch
                train_after(train_loader_single_crop, model, criterion_sce, optimizer, epoch, args, model_ema)

                # evaluate on validation set
                acc1 = validate(val_loader, model, criterion_ce, args)

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, filename=args.save_checkpoint_path+'/checkpoint.pth.tar')
            return
        else:
            # train for one epoch
            train(train_loader, model, criterion_sce, optimizer, epoch, args,model_ema, train_loader_ema)
            print("after train")
        print("start eval")
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion_ce, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=args.save_checkpoint_path+'/checkpoint_{}.pth.tar'.format(epoch + 1))


def nce_loss_within_batch(embeddings, if_conf, temperature=1.0):
    # 假设 embeddings 的shape是 (batch_size, embedding_dim)
    # 假设 if_conf 包含每个样本是否是自信的， shape: (batch_size, ), True代表自信，False代表不自信

    # 首先，我们把embeddings 分解为不自信和自信的samples
    # print("embeddings shape = ", embeddings.shape)
    # print("if_conf shape = ", if_conf.shape)
    pos_embs = embeddings[~if_conf] # 不自信的
    neg_embs = embeddings[if_conf]  # 自信的

    if pos_embs.shape[0] > 0:  # 如果有不自信的样本
        # 创建一个mask来避免一个样本与其自身进行对比（这会导致过于自信的正样本）
        mask = (torch.ones((pos_embs.shape[0], pos_embs.shape[0])) - torch.eye(pos_embs.shape[0])).bool().to(embeddings.device)

        # 计算所有的positive 和 negative pairs 的logits
        pos_logits = torch.mm(pos_embs, pos_embs.t()) / temperature # shape (n_pos, n_pos)
        pos_logits = pos_logits.masked_select(mask).view(pos_embs.shape[0], -1)  # 对角线元素变为0, shape (n_pos, n_pos-1)

        neg_logits = torch.mm(pos_embs, neg_embs.t()) / temperature  # shape (n_pos, n_neg)

        # 计算损失
        logits = torch.cat([pos_logits, neg_logits], dim=-1) # shape (n_pos, n_pos-1+n_neg)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=-1)

        return F.binary_cross_entropy_with_logits(logits, labels)

    else:  # 如果没有不自信的样本
        return 0.


def contrastive_loss(features, is_conf,margin=1.0):
    # 计算余弦相似度
    similarities = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

    # 获取自信和不自信的 mask
    conf_mask = is_conf.view(-1, 1)
    # print("is_conf = ", is_conf.shape)

    not_conf_mask = ~is_conf.view(-1, 1)

    # 计算正对相似度: 不自信样本与不自信样本之间的相似度
    # positive_mask = not_conf_mask & not_conf_mask.T
    positive_mask = not_conf_mask & not_conf_mask.T
    identity_mask = torch.eye(positive_mask.size(0)).bool().to(positive_mask.device)
    positive_mask &= ~identity_mask
    # print("similarities = ", similarities.shape)
    # print("positive_mask = ", positive_mask.shape)
    positive_similarities = similarities * positive_mask.float()

    # 计算负对相似度: 不自信样本与自信样本之间的相似度
    negative_mask = not_conf_mask & conf_mask.T
    negative_similarities = similarities * negative_mask.float()

    # 正对相似度要高，故减去 positive_similarities
    # 负对相似度要低，故加上 negative_similarities
    # margin 是一个阈值，用以确保 positive_similarity 相对于 negative_similarity 要高出一定的 margin
    # 仅在 positive_similarity 不足够高（小于 margin + negative_similarity）时对模型进行惩罚
    losses = F.relu(margin - positive_similarities + negative_similarities)

    # 只取正对和负对的损失，非对的损失不计入，即忽略0值
    losses = losses[positive_mask | negative_mask]

    # 求平均
    loss = losses.sum() / (positive_mask.sum() + negative_mask.sum())

    return loss
    # similarities = torch.matmul(features, features.T) / temperature
    # # F.cosine_similarity()
    # mask_positive = ~is_conf  # 做一个阵列，指示所选样本为正样本(不自信样本)
    # mask_negative = is_conf  # 做一个阵列，指示哪个样本是负样本(自信样本)
    # positive_similarities = similarities[mask_positive][:, mask_positive]
    # negative_similarities = similarities[mask_positive][:, mask_negative]
    #
    # num_positives = positive_similarities.size(0)
    #
    # # 负对的相似度需要被取负值
    # negative_similarities = -negative_similarities
    #
    # # 对正负对的相似度进行掩码处理，并进行连接
    # logits = torch.cat([positive_similarities, negative_similarities], dim=1)
    #
    # # 为正样本创建标签
    # labels = torch.arange(num_positives).to(features.device)
    #
    # return torch.nn.functional.cross_entropy(logits, labels)

def train(train_loader, model, criterion, optimizer, epoch, args, model_ema, train_loader_ema = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, 'LR {lr:.5f}'.format(lr=_get_learning_rate(optimizer))],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images,targets,soft_label,is_consistent) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # true_indices = np.where(is_have_neg == True)[0]
        # false_indices = np.where(is_have_neg == False)[0]
        #
        # np.random.shuffle(true_indices)
        # while len(true_indices) < len(false_indices):
        #     true_indices = np.concatenate((true_indices, true_indices))
        # # 填充is_have_neg为False的位置
        # if len(false_indices) != 0:
        #     sample_in_conf[false_indices] = sample_in_conf[true_indices[:len(false_indices)]]

        # reshape images and soft label
        # images = torch.cat(images, dim=0)
        confident_len = len(targets)
        in_confident_len = len(images) - len(targets)
        # print("images = ", images.shape)
        # print("sample_in_conf = ", sample_in_conf.shape)
        # soft_label = torch.cat(soft_label, dim=0)
        # target = torch.cat(target, dim=0)

        if args.soft_label_type != 'ori':
            soft_label = Recover_soft_label(soft_label, args.soft_label_type, args.num_classes)
        # sample_ori = images[:32]

        # if epoch < (args.epochs - args.num_crops):
        # threshold = np.log2(args.num_classes) * 0.5
        # entropys = -torch.sum(soft_label * torch.log(soft_label + 1e-9), dim=-1)
        # is_low_confidence = entropys > threshold
        # if not torch.any(is_low_confidence):
        #     continue
        # is_low_confidence = [False] * 128
        # # 将最后的64个元素设置为True
        # is_low_confidence[-32:] = [True] * 32

        # 转换为PyTorch张量类型
        # images = torch.cat([images, sample_in_conf], dim=0)

        is_low_confidence = [False] * len(images)
        # 将最后的64个元素设置为True
        is_low_confidence[-in_confident_len:] = [True] * in_confident_len
        is_low_confidence = torch.tensor(is_low_confidence, dtype=torch.bool)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = targets.cuda(args.gpu, non_blocking=True)
            soft_label = soft_label.cuda(args.gpu, non_blocking=True)
            is_low_confidence = is_low_confidence.cuda(args.gpu, non_blocking=True)
            # sample_in_conf = sample_in_conf.cuda(args.gpu, non_blocking=True)
            # sample_ori_weak = sample_ori_weak[:16].cuda(args.gpu, non_blocking=True)
            # sample_ori_stronger = sample_ori_stronger[:16].cuda(args.gpu, non_blocking=True)
            # sample_ori = sample_ori[sample_ori.shape[0] // 2 :].cuda(args.gpu, non_blocking=True)
            # sample_ori = sample_ori.cuda(args.gpu, non_blocking=True)
        # ori_shape = images.shape
        # low_confidence_images = images[is_low_confidence]
        # if len(low_confidence_images) == 0:
        #     continue
        # high_confidence_images = images[~is_low_confidence]
        # if args.mixup_cutmix:
        # images, soft_label = mixup_cutmix(images, soft_label, args)
        # images_copy = images.clone()
        # print("hhlo")
        # is_mix, images_, soft_label_ = my_mixup_cutmix(images[~is_low_confidence], soft_label[~is_low_confidence], entropys[~is_low_confidence])
        # sample_ori, _ = mixup_cutmix(images[:32], soft_label[:32], args)
        # images = torch.cat((images, sample_ori_stronger), dim=0)
        # if is_mix:
        #     print("mix up")
        # else:
        #     print("not mix up")
        # compute output
        # if not is_mix:
        # images[~is_low_confidence] = images_.clone()
        # soft_label[~is_low_confidence] = soft_label_.clone()
            # images = torch.cat((images[is_low_confidence], images))
            # soft_label = torch.cat((soft_label[is_low_confidence], soft_label))
        # output, branch = model(images)

        output, branch = model(images, True)
        batch_size = confident_len // (args.num_crops - 1)
        new_batches = torch.split(output[:confident_len], batch_size, dim=0)
        consis_loss = 0.0
        # true_positions = torch.nonzero(is_consistent, as_tuple=False)
        # # for is_consistent_ in is_consistent:
        # for true_position in true_positions:
        #     b,i,j = true_position
        #     b1 = b * (args.num_crops - 1) +
        #
        # for t_b in range(batch_size):
        #     # new_batch = output[[0, 63, 127, 255], :]
        #     for m in range(args.num_crops - 1 ):
        #         for n in range(args.num_crops-1):
        #             if is_consistent[t_b, m, n]:
        #                 T = 1
        #                 pred1 = new_batches[m][t_b].unsqueeze(0)
        #                 pred2 = new_batches[n][t_b].unsqueeze(0)
        #                 consis_loss += F.kl_div(
        #                     F.log_softmax(pred1 / T, dim=1),
        #                     F.log_softmax(pred2 / T, dim=1),
        #                     reduction='sum',
        #                     log_target=True
        #                 ) * (T * T) / pred1.numel()


        # cont_loss = contrastive_loss(branch, is_conf=~is_low_confidence)
        # if len(false_indices) !=  0:
        #     cont_loss = nce_loss_within_batch(torch.cat((branch[:len1], branch[len1:][true_indices]), dim=0), if_conf=(~is_low_confidence)[:(len1 + len(true_indices))])
        # else:
        cont_loss = nce_loss_within_batch(branch, ~is_low_confidence)


        # cont_loss = contrastive_loss(branch, is_conf=~is_low_confidence)
        loss = criterion(output[~is_low_confidence] / args.temp,
                         soft_label) + 0.01 * cont_loss
        acc1, acc5 = accuracy(output[~is_low_confidence], target, topk=(1, 5))

        #
        # if is_mix:
        #     output,branch = model(images[~is_low_confidence], True)
        #     # cont_loss = contrastive_loss(branch, is_conf=~is_low_confidence)
        #     loss = criterion(output / args.temp,
        #                      soft_label[~is_low_confidence])
        #     acc1, acc5 = accuracy(output, target[~is_low_confidence], topk=(1, 5))
        # else:
        #     # output, branch = model(images, True)
        #     output,branch = model(images[~is_low_confidence], True)
        #
        #     # cont_loss = contrastive_loss(branch, is_conf=~is_low_confidence)
        #     # consis_loss = 0.0
        #     # batch_size = output.shape[0] // args.num_crops
        #     # new_batches = torch.split(output, batch_size, dim=0)
        #     # for t_b in range(batch_size):
        #     #     # new_batch = output[[0, 63, 127, 255], :]
        #     #     for m in range(args.num_crops):
        #     #         for n in range(args.num_crops):
        #     #             if is_consistent[t_b, m, n]:
        #     #                 T = 1
        #     #                 pred1 = new_batches[m][t_b].unsqueeze(0)
        #     #                 pred2 = new_batches[n][t_b].unsqueeze(0)
        #     #                 consis_loss += F.kl_div(
        #     #                     F.log_softmax(pred1 / T, dim=1),
        #     #                     F.log_softmax(pred2 / T, dim=1),
        #     #                     reduction='sum',
        #     #                     log_target=True
        #     #                 ) * (T * T) / pred1.numel()
        #     loss = criterion(output / args.temp,
        #                      # soft_label[~is_low_confidence]) + consis_loss + 0.01 * cont_loss
        #                      # soft_label[~is_low_confidence]) + 0.01 * cont_loss
        #                      soft_label[~is_low_confidence])
        #     acc1, acc5 = accuracy(output, target[~is_low_confidence], topk=(1, 5))
        # output = output_1[:ori_shape[0]]
        # output_ = output_1[ori_shape[0]:]
        # if epoch < (args.epochs - args.num_crops):

        # new_batches = []
        # split_nums = output.shape[0] // args.num_crops
        # for i in range(split_nums):
        #
        #     new_batch = output[i:i + 4, :]
        #     new_batches.append(new_batch)
        # if epoch >= (args.epochs - args.num_crops):
        #     loss = criterion(output / args.temp, soft_label)
        # else:
        # 从256*1000中，在第一维每隔64划分，生成4个单独的tensor
        # consis_loss = 0.0
        # # cont_loss = 0.0
        # cont_loss = contrastive_loss(branch, is_conf=~is_low_confidence)
        # if not is_mix:
        #     batch_size = output.shape[0] // args.num_crops
        #     new_batches = torch.split(output, batch_size, dim=0)
        #     for t_b in range(batch_size):
        #         # new_batch = output[[0, 63, 127, 255], :]
        #         for m in range(args.num_crops):
        #             for n in range(args.num_crops):
        #                 if is_consistent[t_b,m,n]:
        #                     T = 1
        #                     pred1 = new_batches[m][t_b].unsqueeze(0)
        #                     pred2 = new_batches[n][t_b].unsqueeze(0)
        #                     consis_loss += F.kl_div(
        #                         F.log_softmax(pred1 / T, dim=1),
        #                         F.log_softmax(pred2 / T, dim=1),
        #                         reduction='sum',
        #                         log_target=True
        #                     ) * (T * T) / pred1.numel()
        #                     # ) * (T * T)
        #                     # print("hhelo")
        #
        #     # output_ = model(sample_ori)
        #     # model_ema.ema.eval()
        #     # with torch.no_grad():
        #     #     presudo = model_ema.ema(sample_ori_weak).detach()
        #     # loss = criterion(output / args.temp, presudo)
        #     # T = 0.5
        #     #
        #     # loss_ema = F.kl_div(
        #     #     F.log_softmax(output_ / T, dim=1),
        #     #     F.log_softmax(presudo / T, dim=1),
        #     #     reduction='sum',
        #     #     log_target=True
        #     # ) * (T * T) / output_.numel()
        #     # print("cont_loss = ", cont_loss)
        # loss = criterion(output[~is_low_confidence] / args.temp,
        #                  soft_label[~is_low_confidence]) + consis_loss + 0.01 * cont_loss
        # if epoch < (args.epochs - args.num_crops):
        #     loss = criterion(output[~is_low_confidence] / args.temp, soft_label[~is_low_confidence]) + consis_loss + 0.01 * cont_loss
        # else:
        #
        #     loss = criterion(output / args.temp, soft_label) + consis_loss

                # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        # model_ema.update(model)
    # if epoch < (args.epochs - args.num_crops) and args.is_ema:
    # # if epoch < (args.epochs - args.num_crops):
    #     print("start ema")
    #     for i, (images, target) in enumerate(train_loader_ema):
    #         if args.gpu is not None:
    #             images = images.cuda(args.gpu, non_blocking=True)
    #         if torch.cuda.is_available():
    #             target = target.cuda(args.gpu, non_blocking=True)
    #         output = model(images)
    #         model_ema.ema.eval()
    #         with torch.no_grad():
    #             presudo = model_ema.ema(images).detach()
    #         # loss = criterion(output / args.temp, presudo)
    #         T = 0.06
    #         loss = F.kl_div(
    #             F.log_softmax(output / T, dim=1),
    #             F.log_softmax(presudo / T, dim=1),
    #             reduction='sum',
    #             log_target=True
    #         ) * (T * T) / output.numel()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     model_ema.update(model)
    #     args.is_ema = False



def train_after(train_loader, model, criterion, optimizer, epoch, args, model_ema, train_loader_ema = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, 'LR {lr:.5f}'.format(lr=_get_learning_rate(optimizer))],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, soft_label) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # reshape images and soft label
        images = torch.cat(images, dim=0)
        soft_label = torch.cat(soft_label, dim=0)
        target = torch.cat(target, dim=0)

        if args.soft_label_type != 'ori':
            soft_label = Recover_soft_label(soft_label, args.soft_label_type, args.num_classes)
        # sample_ori = images[:32]



        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
            soft_label = soft_label.cuda(args.gpu, non_blocking=True)
            # is_low_confidence = is_low_confidence.cuda(args.gpu, non_blocking=True)
            # sample_ori_weak = sample_ori_weak[:16].cuda(args.gpu, non_blocking=True)
            # sample_ori_stronger = sample_ori_stronger[:16].cuda(args.gpu, non_blocking=True)
            # sample_ori = sample_ori[sample_ori.shape[0] // 2 :].cuda(args.gpu, non_blocking=True)
            # sample_ori = sample_ori.cuda(args.gpu, non_blocking=True)
        # ori_shape = images.shape
        # low_confidence_images = images[is_low_confidence]
        # if len(low_confidence_images) == 0:
        #     continue
        # high_confidence_images = images[~is_low_confidence]
        # sample_ori, _ = mixup_cutmix(images[:32], soft_label[:32], args)
        # images = torch.cat((images, sample_ori_stronger), dim=0)

        # compute output
        output = model(images, False)
        loss = criterion(output / args.temp, soft_label)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args,):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    # model_ema.eval()
    # device = model.device
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # images = images.to(device, non_blocking=True)
            # target = target.to(device, non_blocking=True)

            # compute output
            output, branch = model(images, True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, filename[:-19]+'/model_best.pth.tar')

def my_collate(batchs):
    images = []
    targets = []
    soft_labels = []
    is_consistent = []
    for batch in batchs:
        for image in batch[0]:
            images.append(image)

        for target in batch[1]:
            targets.append(target)

        for soft_label in batch[2]:
            soft_labels.append(torch.from_numpy(soft_label))

        is_consistent.append(batch[5])

    for batch in batchs:
        if batch[3]:
            images.append(batch[4])
    images = torch.stack(images, dim=0)
    targets = torch.from_numpy(np.array(targets))
    soft_labels = torch.stack(soft_labels, dim=0)
    is_consistent = torch.stack(is_consistent, dim=0)
    return images,targets,soft_labels,is_consistent
    # real_batch=np.array(batch)
    # real_batch=torch.from_numpy(real_batch)
    # return real_batch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _get_learning_rate(optimizer):
    return max(param_group['lr'] for param_group in optimizer.param_groups)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()