import os, sys
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
from model.AdjustSegNet import AdjustSegNet
from torchsummary import summary

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
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, DiceLoss
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR
from builders.dataset_builder import build_dataset_test
from scipy.spatial import distance
import torch.nn.functional as F
import cv2

sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
torch_ver = torch.__version__[:3]
print("=====> torch version: ", torch_ver)
GLOBAL_SEED = 1234

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default="UNet1", help="model name: (default ENet)")
    parser.add_argument('--dataset', type=str, default="camvid",
                        help="dataset: cityscapes or camvid or voc or ade20k or sunrgb")
    parser.add_argument('--input_size', type=str, default="360,480", help="input size of model")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=11,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--train_type', type=str, default="train",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    parser.add_argument('--test_small', type=bool, default=False, help="test small model without zeors")
    parser.add_argument('--method', default='Taylor Small Group', type=str,
                        help='pruning method Taylor Small or Taylor Small Group')
    parser.add_argument('--decay_rate', default='0', type=float, help='decay rate for do mask')

    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=300,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--finetune_epochs', type=int, default=50,
                        help="the number of epochs for finetune training after set params to zeros")
    parser.add_argument('--random_mirror', type=bool, default=False, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=False, help="input image resize 0.5 to 2")
    parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='training loss')
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam', 'radam', 'ranger'],
                        help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='warmpoly', help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9, help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', action='store_true', default=False,
                        help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_ohem', action='store_true', default=False,
                        help='OhemCrossEntropy2d Loss for cityscapes dataset')
    parser.add_argument('--use_lovaszsoftmax', action='store_true', default=False,
                        help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', action='store_true', default=False, help=' FocalLoss2d for cityscapes dataset')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")

    # checkpoint and log
    parser.add_argument('--pretrain', type=str, default="",
                        help="pretrain model path")
    parser.add_argument('--savedir', default="./log/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--compress_rate', type=float, default=1, help='pruning rate for model')
    parser.add_argument('--flag', default='test', type=str, help='iteration for experiment')
    args = parser.parse_args()

    return args


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.prune_filters = {}
        self.mat = {}
        self.mat_other = {}
        self.mat_next_conv = {}
        self.model = model
        self.mask_index = []

    def get_taylor_filter_codebook(self, weight_torch, compress_rate, length, filter_grads):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_grads = filter_grads.data.cpu()
            filter_grads = torch.abs(filter_grads)
            filter_ranks = torch.argsort(filter_grads, dim=1, descending=True)
            # print(filter_ranks)

            filter_num = weight_torch.size()[0]
            class_num = filter_grads.size()[0]
            filter_score = torch.zeros(filter_num)

            mask = torch.torch.arange(filter_num - 1, -1, -1, dtype=torch.float)
            mask = mask.repeat(class_num, 1)

            for filter_index in range(filter_num):
                indices = (filter_ranks == filter_index)

                score = torch.sum(indices * mask)
                filter_score[filter_index] = score

            filter_pruned_num = int(filter_num * (1 - compress_rate))
            # filter_index = (filter_score.numpy().argsort()[::-1])[:filter_pruned_num]
            filter_index = filter_score.numpy().argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            # generate prune filter index for per layer
            prune_filter_index = list(np.sort(filter_index))
        else:
            pass
        return codebook, prune_filter_index

    def get_taylor_small_filter_codebook(self, weight_torch, compress_rate, length, filter_grads, decay_rate,
                                         class_weights):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_grads = filter_grads.data.cpu()
            filter_grads = torch.abs(filter_grads)
            filter_ranks = torch.argsort(filter_grads, dim=1, descending=True)

            class_weights = torch.tensor(class_weights)
            sorted_indices = torch.argsort(class_weights)

            filter_ranks = torch.index_select(filter_ranks, dim=0, index=sorted_indices)
            filter_ranks = filter_ranks.transpose(1, 0)

            filter_ranks = filter_ranks.reshape(-1)
            filter_ranks = torch.unique(filter_ranks, sorted=False)

            filter_num = weight_torch.size()[0]

            filter_pruned_num = int(filter_num * (1 - compress_rate))
            # filter_index = (filter_score.numpy().argsort()[::-1])[:filter_pruned_num]
            filter_index = filter_ranks.numpy()[:filter_pruned_num]

            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = decay_rate

            # generate prune filter index for per layer
            prune_filter_index = list(np.sort(filter_index))
        else:
            pass
        return codebook, prune_filter_index

    def get_taylor_small_group_filter_codebook(self, weight_torch, compress_rate, length, filter_grads, decay_rate,
                                               class_weights):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_grads = filter_grads.data.cpu()
            filter_grads = torch.abs(filter_grads)  # 绝对值
            filter_ranks = torch.argsort(filter_grads, dim=1, descending=True)  # 降序排列

            # 选小类别
            class_weights = torch.tensor(class_weights)
            sorted_indices = torch.argsort(class_weights)
            if args.dataset == 'camvid':
                sorted_indices = sorted_indices[0:5]
            elif args.dataset == 'cityscapes':
                sorted_indices = sorted_indices[0:10]
            elif args.dataset == 'voc':
                sorted_indices = sorted_indices[0:10]
            elif args.dataset == 'ade20k':
                sorted_indices = sorted_indices[0:73]
            elif args.dataset == 'sunrgb':
                sorted_indices = sorted_indices[0:25]
            else:
                raise NotImplementedError("Not support current dataset")

            filter_ranks = torch.index_select(filter_ranks, dim=0, index=sorted_indices)

            filter_ranks = filter_ranks.transpose(1, 0)
            filter_ranks = filter_ranks.reshape(-1)  # 展平
            filter_ranks = torch.unique(filter_ranks, sorted=False)  # 去重复

            filter_num = weight_torch.size()[0]
            class_num = filter_grads.size()[0]

            filter_pruned_num = int(filter_num * (1 - compress_rate))
            filter_index = filter_ranks.numpy()[filter_num-filter_pruned_num:filter_num]

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]


            for x in range(0, len(filter_index)):

                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = decay_rate


            # generate prune filter index for per layer
            prune_filter_index = list(np.sort(filter_index))
        else:
            pass
        return codebook, prune_filter_index

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        # 计算参数量
        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        # confirm conv layer mask
        self.mask_index.clear()
        for index, item in enumerate(self.model.parameters()):
            if (len(item.size()) == 4):
                self.mask_index.append(index)

        # remove last conv layer index
        self.mask_index.pop()

        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                self.compress_rate[index] = layer_rate
            else:
                self.compress_rate[index] = 1

    def forward_segnet(self, x):
        self.activations = []
        self.grad_index = 0

        # Stage 1
        temp = self.model.conv11(x)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x11 = F.relu(self.model.bn11(temp))

        temp = self.model.conv12(x11)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x12 = F.relu(self.model.bn12(temp))

        x1_size = x12.size()
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        temp = self.model.conv21(x1p)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x21 = F.relu(self.model.bn21(temp))

        temp = self.model.conv22(x21)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x22 = F.relu(self.model.bn22(temp))

        x2_size = x22.size()
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        temp = self.model.conv31(x2p)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x31 = F.relu(self.model.bn31(temp))

        temp = self.model.conv32(x31)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x32 = F.relu(self.model.bn32(temp))

        temp = self.model.conv33(x32)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x33 = F.relu(self.model.bn33(temp))
        x3_size = x33.size()
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        temp = self.model.conv41(x3p)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x41 = F.relu(self.model.bn41(temp))

        temp = self.model.conv42(x41)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x42 = F.relu(self.model.bn42(temp))

        temp = self.model.conv43(x42)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x43 = F.relu(self.model.bn43(temp))
        x4_size = x43.size()
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        temp = self.model.conv51(x4p)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x51 = F.relu(self.model.bn51(temp))

        temp = self.model.conv52(x51)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x52 = F.relu(self.model.bn52(temp))

        temp = self.model.conv53(x52)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x53 = F.relu(self.model.bn53(temp))
        x5_size = x53.size()
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        temp = self.model.conv53d(x5d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x53d = F.relu(self.model.bn53d(temp))

        temp = self.model.conv52d(x53d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x52d = F.relu(self.model.bn52d(temp))

        temp = self.model.conv51d(x52d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x51d = F.relu(self.model.bn51d(temp))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        temp = self.model.conv43d(x4d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x43d = F.relu(self.model.bn43d(temp))

        temp = self.model.conv42d(x43d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x42d = F.relu(self.model.bn42d(temp))

        temp = self.model.conv41d(x42d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x41d = F.relu(self.model.bn41d(temp))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        temp = self.model.conv33d(x3d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x33d = F.relu(self.model.bn33d(temp))

        temp = self.model.conv32d(x33d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x32d = F.relu(self.model.bn32d(temp))

        temp = self.model.conv31d(x32d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x31d = F.relu(self.model.bn31d(temp))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        temp = self.model.conv22d(x2d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x22d = F.relu(self.model.bn22d(temp))

        temp = self.model.conv21d(x22d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x21d = F.relu(self.model.bn21d(temp))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=x1_size)
        temp = self.model.conv12d(x1d)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        x12d = F.relu(self.model.bn12d(temp))

        x11d = self.model.conv11d(x12d)
        x11d.register_hook(self.compute_rank)
        self.activations.append(x11d)

        return x11d

    def forward_unet(self, x):
        self.activations = []
        self.grad_index = 0

        temp = self.model.in_conv1(x)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        c1 = F.relu(self.model.in_conv1_bn(temp))

        temp = self.model.in_conv2(c1)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        c2 = F.relu(self.model.in_conv2_bn(temp))

        e1_ = F.max_pool2d(c2, kernel_size=2, stride=2)

        temp = self.model.conv1_1(e1_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e1_ = F.relu(self.model.conv1_1_bn(temp))

        temp = self.model.conv1_2(e1_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e1 = F.relu(self.model.conv1_2_bn(temp))

        e2_ = F.max_pool2d(e1, kernel_size=2, stride=2)

        temp = self.model.conv2_1(e2_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e2_ = F.relu(self.model.conv2_1_bn(temp))

        temp = self.model.conv2_2(e2_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e2 = F.relu(self.model.conv2_2_bn(temp))

        e3_ = F.max_pool2d(e2, kernel_size=2, stride=2)

        temp = self.model.conv3_1(e3_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e3_ = F.relu(self.model.conv3_1_bn(temp))

        temp = self.model.conv3_2(e3_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e3 = F.relu(self.model.conv3_2_bn(temp))

        e4_ = F.max_pool2d(e3, kernel_size=2, stride=2)

        temp = self.model.conv4_1(e4_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e4_ = F.relu(self.model.conv4_1_bn(temp))

        temp = self.model.conv4_2(e4_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        e4 = F.relu(self.model.conv4_2_bn(temp))

        d1_1 = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = e3.size()[2] - d1_1.size()[2]
        diffX = e3.size()[3] - d1_1.size()[3]
        d1_1 = F.pad(d1_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d1_2 = torch.cat([d1_1, e3], dim=1)

        temp = self.model.conv5_1(d1_2)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d1_ = F.relu(self.model.conv5_1_bn(temp))

        temp = self.model.conv5_2(d1_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d1 = F.relu(self.model.conv5_2_bn(temp))

        d2_1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = e2.size()[2] - d2_1.size()[2]
        diffX = e2.size()[3] - d2_1.size()[3]
        d2_1 = F.pad(d2_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d2_2 = torch.cat([d2_1, e2], dim=1)

        temp = self.model.conv6_1(d2_2)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d2_ = F.relu(self.model.conv6_1_bn(temp))

        temp = self.model.conv6_2(d2_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d2 = F.relu(self.model.conv6_2_bn(temp))

        d3_1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = e1.size()[2] - d3_1.size()[2]
        diffX = e1.size()[3] - d3_1.size()[3]
        d3_1 = F.pad(d3_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d3_2 = torch.cat([d3_1, e1], dim=1)

        temp = self.model.conv7_1(d3_2)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d3_ = F.relu(self.model.conv7_1_bn(temp))

        temp = self.model.conv7_2(d3_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d3 = F.relu(self.model.conv7_2_bn(temp))

        d4_1 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = c2.size()[2] - d4_1.size()[2]
        diffX = c2.size()[3] - d4_1.size()[3]
        d4_1 = F.pad(d4_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d4_2 = torch.cat([d4_1, c2], dim=1)

        temp = self.model.conv8_1(d4_2)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d4_ = F.relu(self.model.conv8_1_bn(temp))

        temp = self.model.conv8_2(d4_)
        temp.register_hook(self.compute_rank)
        self.activations.append(temp)
        d4 = F.relu(self.model.conv8_2_bn(temp))

        output = self.model.out_conv(d4)
        output.register_hook(self.compute_rank)
        self.activations.append(output)

        return output

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        taylor = activation * grad
        # Get the average value for every filter,
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_grads:
            self.filter_grads[activation_index] = \
                torch.FloatTensor(args.classes, activation.size(1)).zero_()
            if args.cuda:
                self.filter_grads[activation_index] = self.filter_grads[activation_index].cuda()

        self.filter_grads[activation_index][self.class_index] += taylor
        self.grad_index += 1

    def compute_filters_score(self, trainLoader, criterion):
        self.filter_grads = {}

        for i in range(args.classes):
            self.class_index = i

            for iteration, batch in enumerate(trainLoader, 0):
                images, labels, _, _ = batch
                images = images.cuda()
                labels = labels.long().cuda()

                if args.model == 'SegNet':
                    output = self.forward_segnet(images)
                elif args.model == 'UNet1':
                    output = self.forward_unet(images)
                else:
                    raise NotImplementedError("Not support current Network")

                result = torch.argmax(output, axis=1)
                replace_indices = (labels != self.class_index)
                labels[replace_indices] = result[replace_indices]

                loss = criterion(output, labels)
                loss.backward()
                self.activations.clear()

    def init_mask(self, layer_rate, decay_rate, class_weiths):
        self.init_rate(layer_rate)

        i = 0
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                if args.method == 'Taylor':
                    self.mat[index], self.prune_filters[index] = self.get_taylor_filter_codebook(item.data,
                                                                                                 self.compress_rate[
                                                                                                     index],
                                                                                                 self.model_length[
                                                                                                     index],
                                                                                                 self.filter_grads[i])
                elif args.method == 'Taylor Small':
                    self.mat[index], self.prune_filters[index] = self.get_taylor_small_filter_codebook(item.data,
                                                                                                       self.compress_rate[
                                                                                                           index],
                                                                                                       self.model_length[
                                                                                                           index],
                                                                                                       self.filter_grads[
                                                                                                           i],
                                                                                                       decay_rate,
                                                                                                       class_weiths)
                elif args.method == 'Taylor Small Group':
                    self.mat[index], self.prune_filters[index] = self.get_taylor_small_group_filter_codebook(item.data,
                                                                                                             self.compress_rate[
                                                                                                                 index],
                                                                                                             self.model_length[
                                                                                                                 index],
                                                                                                             self.filter_grads[
                                                                                                                 i],
                                                                                                             decay_rate,
                                                                                                             class_weiths)
                else:
                    raise NotImplementedError("Not support current pruning method")

                self.mat[index] = self.convert2tensor(self.mat[index])
                i += 1
                if args.cuda:
                    self.mat[index] = self.mat[index].cuda()

                # generate conv_bias bn_weight and bn_bias mask
                other_codebook = np.ones(item.size()[0])
                for filter_index in self.prune_filters[index]:
                    other_codebook[filter_index] = decay_rate
                self.mat_other[index] = other_codebook
                self.mat_other[index] = self.convert2tensor(self.mat_other[index])
                if args.cuda:
                    self.mat_other[index] = self.mat_other[index].cuda()

    def do_mask(self):
        param_list = list(self.model.parameters())
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])

                # do mask for conv_bias bn_weight and bn_bias
                a = param_list[index + 1].data
                b = a * self.mat_other[index]
                param_list[index + 1].data = b

                a = param_list[index + 2].data
                b = a * self.mat_other[index]
                param_list[index + 2].data = b

                a = param_list[index + 3].data
                b = a * self.mat_other[index]
                param_list[index + 3].data = b

                # # do mask for next conv weight
                # a = param_list[index + 4].data.view(self.model_length[index + 4])
                # b = a * self.mat_next_conv[index]
                # param_list[index + 4].data = b.view(self.model_size[index + 4])

    def do_prune_correct(self):
        prune_model = copy.deepcopy(self.model)


        param_list = list(prune_model.named_parameters())
        for index, (name, item) in enumerate(prune_model.named_parameters()):
            if (index in self.mask_index):
                prune_filter_index = self.prune_filters[index]

                # get old conv layer
                conv_name = name.split('.')[0]
                old_conv = prune_model._modules[conv_name]
                old_conv_weight = old_conv.weight.data
                old_conv_bias = old_conv.bias.data
                # generate new conv layer
                new_conv = torch.nn.Conv2d(old_conv.in_channels,
                                           old_conv.out_channels - len(prune_filter_index),
                                           kernel_size=old_conv.kernel_size,
                                           padding=old_conv.padding)
                if args.cuda:
                    new_conv = new_conv.cuda()
                new_conv_weight = new_conv.weight.data
                new_conv_bias = new_conv.bias.data
                copy_index = 0
                for i in range(old_conv.out_channels):
                    if i not in prune_filter_index:
                        new_conv_weight[copy_index] = old_conv_weight[i]
                        new_conv_bias[copy_index] = old_conv_bias[i]
                        copy_index += 1
                new_conv.weight.data = new_conv_weight
                new_conv.bias.data = new_conv_bias

                # replace conv layer
                # prune_model._modules[conv_name] = new_conv
                setattr(prune_model, conv_name, new_conv)
                del old_conv

                # get old bn layer
                bn_name, _ = param_list[index + 2]
                bn_name = bn_name.split('.')[0]
                old_bn = prune_model._modules[bn_name]
                old_bn_weight = old_bn.weight.data
                old_bn_bias = old_bn.bias.data
                old_bn_running_mean = old_bn.running_mean.data
                old_bn_running_var = old_bn.running_var.data

                # generate new bn layer
                new_bn = torch.nn.BatchNorm2d(old_bn.num_features - len(prune_filter_index),
                                              momentum=old_bn.momentum,
                                              eps=old_bn.eps)
                if args.cuda:
                    new_bn = new_bn.cuda()
                new_bn_weight = new_bn.weight.data
                new_bn_bias = new_bn.bias.data
                new_bn_running_mean = new_bn.running_mean.data
                new_bn_running_var = new_bn.running_var.data

                copy_index = 0
                for i in range(old_bn.num_features):
                    if i not in prune_filter_index:
                        new_bn_weight[copy_index] = old_bn_weight[i]
                        new_bn_bias[copy_index] = old_bn_bias[i]
                        new_bn_running_mean[copy_index] = old_bn_running_mean[i]
                        new_bn_running_var[copy_index] = old_bn_running_var[i]
                        copy_index += 1
                new_bn.weight.data = new_bn_weight
                new_bn.bias.data = new_bn_bias
                new_bn.running_mean.data = new_bn_running_mean
                new_bn.running_var.data = new_bn_running_var
                new_bn.num_batches_tracked.data = old_bn.num_batches_tracked.data

                # replace bn layer
                # prune_model._modules[bn_name] = new_bn
                setattr(prune_model, bn_name, new_bn)
                del old_bn

                #get unet next skip connection layer
                if args.model == 'UNet1':
                    skip_index = -1
                    if index == 28:
                        skip_index = 40
                    elif index == 20:
                        skip_index = 48
                    elif index == 12:
                        skip_index = 56
                    elif index == 4:
                        skip_index = 64

                    if skip_index != -1:
                        # get next conv layer
                        next_conv_name, _ = param_list[skip_index]
                        next_conv_name = next_conv_name.split('.')[0]
                        old_next_conv = prune_model._modules[next_conv_name]
                        old_next_conv_weights = old_next_conv.weight.data
                        # generate new next conv layer
                        new_next_conv = torch.nn.Conv2d(old_next_conv.in_channels - len(prune_filter_index),
                                                        old_next_conv.out_channels,
                                                        kernel_size=old_next_conv.kernel_size,
                                                        padding=old_next_conv.padding)

                        k = old_next_conv.in_channels // 2
                        if args.cuda:
                            new_next_conv = new_next_conv.cuda()
                        new_next_conv_weights = new_next_conv.weight.data
                        for i in range(old_next_conv.out_channels):
                            copy_index = k
                            for j in range(k):
                                if j not in prune_filter_index:
                                    new_next_conv_weights[i][copy_index] = old_next_conv_weights[i][j+k]
                                    copy_index += 1
                        new_next_conv.weight.data = new_next_conv_weights

                        new_next_conv.bias.data = old_next_conv.bias.data
                        # replace layer
                        # prune_model._modules[next_conv_name] = new_next_conv
                        setattr(prune_model, next_conv_name, new_next_conv)
                        del old_next_conv


                # get next conv layer
                next_conv_name, _ = param_list[index + 4]
                next_conv_name = next_conv_name.split('.')[0]
                old_next_conv = prune_model._modules[next_conv_name]
                old_next_conv_weights = old_next_conv.weight.data
                # generate new next conv layer
                new_next_conv = torch.nn.Conv2d(old_next_conv.in_channels - len(prune_filter_index),
                                                old_next_conv.out_channels,
                                                kernel_size=old_next_conv.kernel_size,
                                                padding=old_next_conv.padding)
                if args.cuda:
                    new_next_conv = new_next_conv.cuda()
                new_next_conv_weights = new_next_conv.weight.data
                for i in range(old_next_conv.out_channels):
                    copy_index = 0
                    for j in range(old_next_conv.in_channels):
                        if j not in prune_filter_index:
                            new_next_conv_weights[i][copy_index] = old_next_conv_weights[i][j]
                            copy_index += 1
                new_next_conv.weight.data = new_next_conv_weights

                new_next_conv.bias.data = old_next_conv.bias.data
                # replace layer
                # prune_model._modules[next_conv_name] = new_next_conv
                setattr(prune_model, next_conv_name, new_next_conv)
                del old_next_conv



        return prune_model

    def do_prune(self):
        prune_model = copy.deepcopy(self.model)

        param_list = list(prune_model.named_parameters())
        for index, (name, item) in enumerate(prune_model.named_parameters()):
            if (index in self.mask_index):
                prune_filter_index = self.prune_filters[index]

                # get old conv layer
                conv_name = name.split('.')[0]
                old_conv = prune_model._modules[conv_name]
                old_conv_weight = old_conv.weight.data
                old_conv_bias = old_conv.bias.data
                # generate new conv layer
                new_conv = torch.nn.Conv2d(old_conv.in_channels,
                                           old_conv.out_channels - len(prune_filter_index),
                                           kernel_size=old_conv.kernel_size,
                                           padding=old_conv.padding)
                if args.cuda:
                    new_conv = new_conv.cuda()
                new_conv_weight = new_conv.weight.data
                new_conv_bias = new_conv.bias.data
                copy_index = 0
                for i in range(old_conv.out_channels):
                    if i not in prune_filter_index:
                        new_conv_weight[copy_index] = old_conv_weight[i]
                        new_conv_bias[copy_index] = old_conv_bias[i]
                        copy_index += 1
                new_conv.weight.data = new_conv_weight
                new_conv.bias.data = new_conv_bias

                # replace conv layer
                # prune_model._modules[conv_name] = new_conv
                setattr(prune_model, conv_name, new_conv)
                del old_conv

                # get old bn layer
                bn_name, _ = param_list[index + 2]
                bn_name = bn_name.split('.')[0]
                old_bn = prune_model._modules[bn_name]
                old_bn_weight = old_bn.weight.data
                old_bn_bias = old_bn.bias.data
                old_bn_running_mean = old_bn.running_mean.data
                old_bn_running_var = old_bn.running_var.data

                # generate new bn layer
                new_bn = torch.nn.BatchNorm2d(old_bn.num_features - len(prune_filter_index),
                                              momentum=old_bn.momentum,
                                              eps=old_bn.eps)
                if args.cuda:
                    new_bn = new_bn.cuda()
                new_bn_weight = new_bn.weight.data
                new_bn_bias = new_bn.bias.data
                new_bn_running_mean = new_bn.running_mean.data
                new_bn_running_var = new_bn.running_var.data

                copy_index = 0
                for i in range(old_bn.num_features):
                    if i not in prune_filter_index:
                        new_bn_weight[copy_index] = old_bn_weight[i]
                        new_bn_bias[copy_index] = old_bn_bias[i]
                        new_bn_running_mean[copy_index] = old_bn_running_mean[i]
                        new_bn_running_var[copy_index] = old_bn_running_var[i]
                        copy_index += 1
                new_bn.weight.data = new_bn_weight
                new_bn.bias.data = new_bn_bias
                new_bn.running_mean.data = new_bn_running_mean
                new_bn.running_var.data = new_bn_running_var
                new_bn.num_batches_tracked.data = old_bn.num_batches_tracked.data

                # replace bn layer
                # prune_model._modules[bn_name] = new_bn
                setattr(prune_model, bn_name, new_bn)
                del old_bn

                # get next conv layer
                next_conv_name, _ = param_list[index + 4]
                next_conv_name = next_conv_name.split('.')[0]
                old_next_conv = prune_model._modules[next_conv_name]
                old_next_conv_weights = old_next_conv.weight.data
                # generate new next conv layer
                new_next_conv = torch.nn.Conv2d(old_next_conv.in_channels - len(prune_filter_index),
                                                old_next_conv.out_channels,
                                                kernel_size=old_next_conv.kernel_size,
                                                padding=old_next_conv.padding)
                if args.cuda:
                    new_next_conv = new_next_conv.cuda()
                new_next_conv_weights = new_next_conv.weight.data
                for i in range(old_next_conv.out_channels):
                    copy_index = 0
                    for j in range(old_next_conv.in_channels):
                        if j not in prune_filter_index:
                            new_next_conv_weights[i][copy_index] = old_next_conv_weights[i][j]
                            copy_index += 1
                new_next_conv.weight.data = new_next_conv_weights

                new_next_conv.bias.data = old_next_conv.bias.data
                # replace layer
                # prune_model._modules[next_conv_name] = new_next_conv
                setattr(prune_model, next_conv_name, new_next_conv)
                del old_next_conv

        return prune_model

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                print("number of nonzero weight is %d, zero is %d" % (
                    np.count_nonzero(b), len(b) - np.count_nonzero(b)))


def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))
    print("=====> args:", args)

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
        init_weight(model, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, 1e-3, 0.1,
                    mode='fan_in')

    # make mask
    mask = Mask(model)
    mask.init_length()

    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train(args.dataset, input_size, args.batch_size, args.train_type,
                                                        args.random_scale, args.random_mirror, args.num_workers)

    # load the test set
    testLoader = None
    if args.dataset == 'camvid':
        _, testLoader = build_dataset_test(args.dataset, input_size, args.batch_size, args.num_workers)

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

    # ccreate log folder
    args.logdir = (args.savedir + '/' + args.dataset + '/' + "Decay=0" + "_P_" +
                   args.model + "_" + args.method + "_" +
                   args.loss + "_" +
                   str(args.compress_rate) + "Rate" + "_" +
                   args.flag + "Flag" + "_" +
                   str(args.max_epochs) + "Epoch" + '/')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    start_epoch = 0

    model.train()
    cudnn.benchmark = True
    # cudnn.deterministic = True ## my add

    logFileLoc = args.logdir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'IOU(val)', 'mIOU(test)', 'IOU(test)'))
    logger.flush()

    # define optimization strategy
    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # auto_decay_rate_optimizer = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, auto_decay_rate.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
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

    lossTr_list = []
    epoches = []
    mIOU_val_list = []
    mIOU_test_list = []

    best_test_miou = 0

    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        lossTr_list.append(lossTr)

        # execute sfp
        mask.model = model
        if epoch == 0:
            mask.compute_filters_score(trainLoader, criteria)

        if epoch < (args.max_epochs - args.finetune_epochs):
            mask.init_mask(args.compress_rate, args.decay_rate, datas['classWeights'])
        else:
            mask.init_mask(args.compress_rate, 0, datas['classWeights'])
        mask.do_mask()
        model = mask.model

        # validation
        epoches.append(epoch)
        mIOU_val, per_class_iu_val = val(args, valLoader, model)
        mIOU_val_list.append(mIOU_val)

        # testing
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

        # prune model and test
        if args.test_small:
            prune_model = mask.do_prune_correct()
            save_path = args.logdir + 'small_model.pth'
            torch.save(prune_model, save_path)
            if testLoader != None:
                mIOU_prune, per_class_iu_prune = val(args, testLoader, prune_model)
            else:
                mIOU_prune, per_class_iu_prune = val(args, valLoader, prune_model)
        else:
            mIOU_prune = 0.0
            per_class_iu_prune = []

        # record train information
        logger.write("\n%d\t%.7f\t%.7f\t%s\t%.7f\t%s" % (
        epoch, lossTr, mIOU_val, str(per_class_iu_val), mIOU_test, str(per_class_iu_test)))
        logger.flush()
        print(
            "Epoch No.: %d\t Train Loss = %.7f\t mIOU(val) = %.7f\t mIOU(test) = %.7f\t mIOU(prune)=%.7f\t lr= %.7f" % (
            epoch,
            lossTr,
            mIOU_val, mIOU_test, mIOU_prune, lr))

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
        scheduler.step()  # In pytorch 1.1 .0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item())  #

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

    if args.dataset == 'cityscapes':
        args.classes = 19
        args.input_size = '512,1024'
        args.batch_size = 6
        ignore_label = 255
    elif args.dataset == 'camvid':
        args.classes = 11
        args.input_size = '360,480'
        args.batch_size = 8
        ignore_label = 11
    elif args.dataset == 'voc':
        args.classes = 21
        args.input_size = '375, 500'
        args.batch_size = 20
        ignore_label = 255
    elif args.dataset == 'ade20k':
        args.classes = 150
        args.input_size = '512,512'
        args.batch_size = 8
        ignore_label = -1
    elif args.dataset == 'sunrgb':
        args.classes = 37
        args.input_size = '265,365'
        args.batch_size = 32
        ignore_label = -1
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    args.compress_rate = 0.1
    train_model(args)
    # rename log folder name
    # old_dir = args.logdir
    # new_dir = old_dir.replace("Lock", "HHHHFFFinish")
    # os.rename(old_dir, new_dir)
    # args.compress_rate = 0.1
    # train_model(args)
    # old_dir = args.logdir
    # new_dir = old_dir.replace("Lock", "HHHHFFFinish")
    # os.rename(old_dir, new_dir)
    # args.compress_rate = 0.07
    # train_model(args)





