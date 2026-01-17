import os,sys
import time
import torch
import torchsummary
from torch import optim
import torch.nn as nn
import timeit
import math
import numpy as np
import matplotlib
import copy
from functools import partial
from torchsummary import summary
from torchvision import transforms
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight, netParams
from utils.metric.metric import get_iou, Metrics
from utils.metric.score import SegmentationMetric
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth,\
    ProbOhemCrossEntropy2d, FocalLoss2d, DiceLoss
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR
from builders.dataset_builder import build_dataset_test
from scipy.spatial import distance
from thop import profile
from model.SegNet import  SegNet
from model.UNet import UNet
from collections import OrderedDict
import torch_pruning as tp


is_float64 = False

if is_float64:
    torch.set_default_dtype(torch.float64)

#torch.set_printoptions(sci_mode=False)

sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
torch_ver = torch.__version__[:3]
print("=====> torch version: ", torch_ver)
GLOBAL_SEED = 1234

def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default="UNet1", help="model name: (default ENet)")
    parser.add_argument('--dataset', type=str, default="cityscapes", help="dataset: cityscapes or camvid or voc or ade20k or sunrgb")
    parser.add_argument('--input_size', type=str, default="360,480", help="input size of model")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=11,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--train_type', type=str, default="train",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    parser.add_argument('--test_small', type=bool, default=True, help="test small model without zeors")
    parser.add_argument('--method', default='group_sl', type=str, help='pruning method')

    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=100,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--random_mirror', type=bool, default=False, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=False, help="input image resize 0.5 to 2")
    parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='training loss')
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=6, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--optim',type=str.lower,default='adam',choices=['sgd','adam','radam','ranger'],help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='warmpoly', help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9,help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', action='store_true', default=False, help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_ohem', action='store_true', default=False, help='OhemCrossEntropy2d Loss for cityscapes dataset')
    parser.add_argument('--use_lovaszsoftmax', action='store_true', default=False, help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', action='store_true', default=False,help=' FocalLoss2d for cityscapes dataset')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="1", help="default GPU devices (0,1)")

    #sparasity learning args
    parser.add_argument("--max-pruning-ratio", type=float, default=1.0)
    parser.add_argument("--global-pruning", action="store_true", default=False)
    parser.add_argument("--sl-total-epochs", type=int, default=50, help="epochs for sparsity learning")
    parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
    parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
    parser.add_argument("--sl-lr-decay-gamma", default=0.1, type=float)

    # checkpoint and log
    parser.add_argument("--reg", type=float, default=5e-4)
    parser.add_argument('--pretrain', type=str, default="",
                        help="pretrain model path")
    parser.add_argument('--savedir', default="./log/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--compress_rate', type=float, default=0.5, help='pruning rate for model')
    parser.add_argument('--flag', default='small', type=str, help='iteration for experiment')
    args = parser.parse_args()

    return args

def get_pruner(model, example_inputs):
    args.sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l2":
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "fpgm":
        imp = tp.importance.FPGMImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "obdc":
        imp = tp.importance.OBDCImportance(group_reduction='mean', num_classes=args.num_classes)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning, group_lasso=True)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "growing_reg":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError

    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.classes:
            ignored_layers.append(m)

    # Here we fix iterative_steps=200 to prune the model progressively with small steps
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps= 1,
        pruning_ratio= (1 - args.compress_rate),
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=args.max_pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )




    return pruner


def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))
    print("=====> args:",args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


    # set the seed
    setup_seed(GLOBAL_SEED)
    print("=====> set Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("=====> building network")

    # build the model
    if args.pretrain:
        model = torch.load(args.pretrain)

    else:
        model = build_model(args.model, num_classes=args.classes)
        # init_weight(model, nn.init.kaiming_normal_,
        #             nn.BatchNorm2d, 1e-3, 0.1,
        #             mode='fan_in')


    #compute network params
    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train(args.dataset, input_size, args.batch_size, args.train_type,
                                                        args.random_scale, args.random_mirror, args.num_workers)

    # load the test set
    testLoader = None
    if args.dataset == 'camvid':
        _, testLoader = build_dataset_test(args.dataset, input_size, 1, args.num_workers)


    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter


    print('=====> Dataset statistics')
    print("data['classWeights']: ", datas['classWeights'])
    print('mean and std: ', datas['mean'], datas['std'])


    # define loss function, respectively
    weight = torch.from_numpy(datas['classWeights'])
    if args.loss == 'CrossEntropyLoss':
        criteria = CrossEntropyLoss2d(weight=None, ignore_label=ignore_label)
    elif args.loss == 'CrossEntropyLossWeight':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif args.loss == 'DiceLoss':
        criteria = DiceLoss(weight=weight, ignore_index=ignore_label)
    elif args.loss == 'FocalLoss':
        criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    if args.cuda:
        criteria = criteria.cuda()
        model = model.cuda()  # 1-card data parallel


    #ccreate log folder
    args.logdir = (args.savedir + args.dataset + '/' + "Lock" + "_P_" +
                    args.model +  "_" + args.method + "_" +
                    args.loss + "_" +
                    str(args.compress_rate) + "Rate" + "_" +
                    args.flag + "Flag" + "_" +
                    str(args.max_epochs) + "Epoch" + '/')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)


    #cudnn.benchmark = True
    # cudnn.deterministic = True ## my add
    logFileLoc = args.logdir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)','IOU(val)','mIOU(test)', 'IOU(test)'))
    logger.flush()



    #Sparsity Learning
    args.example_inputs = args.example_inputs.cuda()
    pruner = get_pruner(model, example_inputs=args.example_inputs)

    sl_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.sl_lr,
        momentum=0.9,
        weight_decay=5e-4 if pruner is None else 0,
    )

    milestones = [int(ms) for ms in args.sl_lr_decay_milestones.split(",")]
    sl_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        sl_optimizer, milestones=milestones, gamma=args.sl_lr_decay_gamma
    )

    print('=====> beginning sparsity traing')
    for sl_epoch in range(args.sl_total_epochs):
        #sparsity training
        lossTr, lr = sparsity_train(args, trainLoader, model, criteria, sl_optimizer, sl_epoch, pruner, sl_scheduler)

        # validation
        mIOU_val, per_class_iu_val = val(args, valLoader, model)

        # testing
        if testLoader != None:
            mIOU_test, per_class_iu_test = val(args, testLoader, model)
        else:
            mIOU_test = mIOU_val
            per_class_iu_test = per_class_iu_val

        print("Sparsity Training Epoch No.: %d\t Train Loss = %.7f\t mIOU(val) = %.7f\t mIOU(test) = %.7f\t lr= %.7f" % (sl_epoch,
                                                                                                       lossTr,
                                                                                                       mIOU_val,
                                                                                        mIOU_test, lr))

    #Pruning
    print('=====> beginning pruning')
    model.eval()
    pruner.step()
    print(model)
    print("=====> computing network parameters and FLOPs after pruning")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))


    #Finetuning
    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'radam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.90, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'ranger':
        optimizer = Ranger(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.95, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)

    start_epoch = 0
    model.train()

    lossTr_list = []
    epoches = []
    mIOU_val_list = []
    mIOU_test_list = []

    best_test_miou = 0
    print('=====> beginning finetuning')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        lossTr_list.append(lossTr)


        # validation
        epoches.append(epoch)
        mIOU_val, per_class_iu_val = val(args, valLoader, model)
        mIOU_val_list.append(mIOU_val)

        #testing
        if testLoader != None:
            mIOU_test, per_class_iu_test = val(args, testLoader, model)
            mIOU_test_list.append(mIOU_test)
        else:
            mIOU_test = mIOU_val
            per_class_iu_test = per_class_iu_val
            mIOU_test_list.append(mIOU_test)

        # save best model
        if mIOU_test > best_test_miou:
            best_test_miou = mIOU_test
            save_path = args.logdir + 'best_model.pth'
            torch.save(model, save_path)

        # record train information
        logger.write("\n%d\t%.7f\t%.7f\t%s\t%.7f\t%s" % (epoch, lossTr, mIOU_val, str(per_class_iu_val), mIOU_test, str(per_class_iu_test)))
        logger.flush()
        print("Epoch No.: %d\t Train Loss = %.7f\t mIOU(val) = %.7f\t mIOU(test) = %.7f\t lr= %.7f" % (epoch,
                                                                                    lossTr,
                                                                                    mIOU_val, mIOU_test, lr))

        # Plot the figures
        fig1, ax1 = plt.subplots(figsize=(11, 8))

        ax1.plot(range(start_epoch, epoch + 1), lossTr_list)
        ax1.set_title("Average training loss vs epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current loss")
        plt.savefig(args.logdir + "loss_vs_epochs.png")
        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(11, 8))
        ax2.plot(epoches, mIOU_val_list, label="Val mIoU")
        ax2.plot(epoches, mIOU_test_list, label="Test mIoU")
        ax2.set_title("mIoU vs epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Current mIoU")
        plt.legend(loc='lower right')

        plt.savefig(args.logdir + "iou_vs_epochs.png")

        plt.close('all')

    logger.close()



def sparsity_train(args, train_loader, model, criterion, optimizer, epoch, pruner, scheduler):
    model.train()
    epoch_loss = []

    for iteration, batch in enumerate(train_loader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()

        if pruner is not None:
            pruner.regularize(model)

        optimizer.step()
        scheduler.step()  # In pytorch 1.1 .0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item())  #

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    lr = optimizer.param_groups[0]['lr']

    return average_epoch_loss_train, lr



def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """

    model.train()
    epoch_loss = []
    total_batches = len(train_loader)

    for iteration, batch in enumerate(train_loader, 0):
        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration
        # learming scheduling
        if args.lr_schedule == 'poly':
            lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.lr_schedule == 'warmpoly':
            scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=args.warmup_iters, power=0.9)
        lr = optimizer.param_groups[0]['lr']


        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.cuda()


        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # In pytorch 1.1 .0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item()) #


    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr

def val(args, val_loader, model):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = Metrics(args.classes, ignore_label, device)

    for i, (input, label, size, name) in enumerate(val_loader):
        with torch.no_grad():
            # input_var = Variable(input).cuda()
            input_var = input.cuda()
            output = model(input_var)
        metrics.update(output, label.cuda())


    per_class_iu, meanIoU = metrics.compute_iou()

    return meanIoU, per_class_iu


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_args()

    #args.pretrain = '/home/ubuntu/projects/Torch-Pruning/log/camvid/Train_Train_SegNet_CrossEntropyLoss_testFlag_20Epoch/best_model.pth'
    if args.dataset == 'cityscapes':
        args.classes = 19
        args.input_size = '512,1024'
        ignore_label = 255
        args.example_inputs = torch.randn(1,3,512,1024)
    elif args.dataset == 'camvid':
        args.classes = 11
        args.input_size = '360,480'
        ignore_label = 11
        args.example_inputs = torch.randn(1, 3, 360, 480)
    elif args.dataset == 'voc':
        args.classes = 21
        args.input_size = '375, 500'
        ignore_label = 255
        args.example_inputs = torch.randn(1, 3, 375, 500)
    elif args.dataset == 'ade20k':
        args.classes = 150
        args.input_size = '512,512'
        ignore_label = -1
        args.example_inputs = torch.randn(1, 3, 512, 512)
    elif args.dataset == 'sunrgb':
        args.classes = 37
        args.input_size = '265,365'
        ignore_label = -1
        args.example_inputs = torch.randn(1, 3, 262, 365)
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    # args.compress_rate = 0.25
    # train_model(args)
    # args.compress_rate = 0.1
    # train_model(args)
    args.compress_rate = 0.07
    train_model(args)
    # rename log folder name
    # old_dir = args.logdir
    # new_dir = old_dir.replace("Lock", "Finish")
    # os.rename(old_dir, new_dir)
