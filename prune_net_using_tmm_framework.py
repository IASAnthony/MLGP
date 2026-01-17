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
from scipy import spatial as spatial
from scipy.spatial import distance
import torch.nn.functional as F
import cv2

sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
torch_ver = torch.__version__[:3]
print("=====> torch version: ", torch_ver)
GLOBAL_SEED = 1234

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default="UNet1", help="model name: (default ENet)")
    parser.add_argument('--dataset', type=str, default="cityscapes",
                        help="dataset: cityscapes or camvid or voc or ade20k or sunrgb")
    parser.add_argument('--input_size', type=str, default="360,480", help="input size of model")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=11,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--train_type', type=str, default="train",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    parser.add_argument('--test_small', type=bool, default=False, help="test small model without zeors")
    parser.add_argument('--method', default='TMM', type=str,
                        help='pruning method Taylor Small or Taylor Small Group')
    parser.add_argument('--decay_rate', default='0.9', type=float, help='decay rate for do mask')

    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=300,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--finetune_epochs', type=int, default=20,
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
    parser.add_argument('--gpus', type=str, default="1", help="default GPU devices (0,1)")

    # checkpoint and log
    parser.add_argument('--pretrain', type=str, default="",
                        help="pretrain model path")
    parser.add_argument('--savedir', default="./log/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--compress_rate', type=float, default=1, help='pruning rate for model')
    parser.add_argument('--flag', default='test', type=str, help='iteration for experiment')
    parser.add_argument('--efficient_pruning', action='store_true', help='pruning in an efficient way')
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

    def get_tmm_filter_codebook(self, weight_torch, compress_rate, length, decay_rate, k):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            filter_index = []

            similarities = spatial.distance.pdist(weight_vec.cpu())  # similar vector
            similarities_matrix = spatial.distance.squareform(similarities)
            # if k < weight_torch.size()[0] - filter_pruned_num, perform iterative prune
            if k < (weight_torch.size()[0] - filter_pruned_num):

                # pruning rate is larger than 0.5, deleting large local power filters first
                if compress_rate > 0.5 and args.efficient_pruning:
                    small_sim_matrix = similarities_matrix.copy()

                    filter_keeped_num = weight_torch.size()[0] - filter_pruned_num
                    # for indexing knn, after deleting points
                    ori_index = np.array(list(range(len(similarities_matrix))))
                    pruned_list = []
                    for i in range(filter_keeped_num):
                        # find k smallest neighbors for
                        nb_idx = []
                        nb_dst = []
                        # notice the deleted filters + k < n
                        for j in range(len(small_sim_matrix)):
                            nb_idx.append(np.argpartition(small_sim_matrix[j], k)[:k])
                            nb_dst.append(small_sim_matrix[j][nb_idx[j]])

                        # 4.2 search points with largest local power i and j
                        nb_dst_sumK = np.array(nb_dst).sum(axis=1)
                        fi = np.argmax(nb_dst_sumK)  # deleting large local power
                        # print(nb_dst.sum(axis=1))
                        min_edge_list = list(np.where(np.isclose(nb_dst_sumK, nb_dst_sumK[fi], 1e-5, 1e-8))[0])
                        # print(fi,min_edge_list, nb_idx[fi][1])

                        global_power_list = []
                        for i in range(len(min_edge_list)):
                            idx_ori = ori_index[min_edge_list[i]]
                            global_power_list.append(np.average(similarities_matrix[idx_ori, :]))

                        # print(min_edge_list[np.argmin(global_power_list)],global_power_list)

                        delete_idx = min_edge_list[np.argmax(global_power_list)]  # deleting large local power
                        delete_idx_ori = ori_index[delete_idx]
                        pruned_list.append(delete_idx_ori)

                        similarities_matrix[delete_idx_ori, :] = 0
                        similarities_matrix[:, delete_idx_ori] = 0

                        # delete points
                        small_sim_matrix = np.delete(small_sim_matrix, delete_idx, 0)
                        small_sim_matrix = np.delete(small_sim_matrix, delete_idx, 1)
                        # adjust indexs
                        ori_index[delete_idx:len(ori_index) - 1] = ori_index[delete_idx + 1:len(ori_index)]

                    filter_index = list(set([i for i in range(weight_torch.size()[0])]) - set(pruned_list))
                # pruning rate is smaller than 0.5, deleting small local power filters first
                else:
                    small_sim_matrix = similarities_matrix.copy()
                    # for indexing knn, after deleting points
                    ori_index = np.array(list(range(len(similarities_matrix))))
                    pruned_list = []
                    for i in range(filter_pruned_num):
                        # find k smallest neighbors for
                        nb_idx = []
                        nb_dst = []
                        # notice the deleted filters + k < n
                        for j in range(len(small_sim_matrix)):
                            nb_idx.append(np.argpartition(small_sim_matrix[j], k)[:k])
                            nb_dst.append(small_sim_matrix[j][nb_idx[j]])

                        # 4.2 search points with smallest local power i and j
                        nb_dst_sumK = np.array(nb_dst).sum(axis=1)
                        fi = np.argmin(nb_dst_sumK)
                        # print(nb_dst.sum(axis=1))
                        min_edge_list = list(np.where(np.isclose(nb_dst_sumK, nb_dst_sumK[fi], 1e-5, 1e-8))[0])
                        # print(fi,min_edge_list, nb_idx[fi][1])

                        global_power_list = []
                        for i in range(len(min_edge_list)):
                            idx_ori = ori_index[min_edge_list[i]]
                            global_power_list.append(np.average(similarities_matrix[idx_ori, :]))

                        # print(min_edge_list[np.argmin(global_power_list)],global_power_list)

                        delete_idx = min_edge_list[np.argmin(global_power_list)]
                        delete_idx_ori = ori_index[delete_idx]
                        pruned_list.append(delete_idx_ori)

                        similarities_matrix[delete_idx_ori, :] = 0
                        similarities_matrix[:, delete_idx_ori] = 0

                        # delete points
                        small_sim_matrix = np.delete(small_sim_matrix, delete_idx, 0)
                        small_sim_matrix = np.delete(small_sim_matrix, delete_idx, 1)
                        # adjust indexs
                        ori_index[delete_idx:len(ori_index) - 1] = ori_index[delete_idx + 1:len(ori_index)]

                        filter_index = pruned_list

            else:  # global similarity
                print('global prune')
                sum_vec_similairty = np.argsort(np.sum(similarities_matrix, axis=0))
                pruned_list = sum_vec_similairty[:filter_pruned_num]
                print(pruned_list)
                filter_index = pruned_list

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = decay_rate

            #print("filter codebook done")
        else:
            pass
        return codebook, filter_index



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



    def init_mask(self, layer_rate, decay_rate, k):
        self.init_rate(layer_rate)

        i = 0
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                if args.method == 'TMM':
                    self.mat[index], self.prune_filters[index] = self.get_tmm_filter_codebook(item.data,
                                                                                                 self.compress_rate[
                                                                                                     index],
                                                                                                 self.model_length[
                                                                                                     index],
                                                                                              decay_rate,
                                                                                                 k)
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
    args.logdir = (args.savedir + '/' + args.dataset + '/' + "Lock" + "_P_" +
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
    
        if epoch < (args.max_epochs - args.finetune_epochs):
            mask.init_mask(args.compress_rate, args.decay_rate, k=2)
        else:
            mask.init_mask(args.compress_rate, 0, k=2)
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

    args.compress_rate = 0.2
    train_model(args)
    # rename log folder name
    # old_dir = args.logdir
    # new_dir = old_dir.replace("Lock", "Finish")
    # os.rename(old_dir, new_dir)
    args.compress_rate = 0.1
    train_model(args)
    # old_dir = args.logdir
    # new_dir = old_dir.replace("Lock", "Finish")
    # os.rename(old_dir, new_dir)
    args.compress_rate = 0.07
    train_model(args)
    # old_dir = args.logdir
    # new_dir = old_dir.replace("Lock", "Finish")
    # os.rename(old_dir, new_dir)





