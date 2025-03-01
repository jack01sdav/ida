
from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pprint
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model.da_faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import (
    adjust_learning_rate,
    clip_gradient,
    load_net,
    save_checkpoint,
    save_net,
    weights_normal_init,
)
from roi_da_data_layer.roibatchLoader import roibatchLoader
from roi_da_data_layer.roidb import combined_roidb
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler



def infinite_data_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def parse_args():
    """
  Parse input arguments
  """
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    # 添加命令行参数--dataset，用于指定训练数据集，默认为HRSC
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="HRSC",
        type=str,
    )
    # 添加命令行参数--net，用于指定网络结构，默认为res101
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="res101", type=str
    )
    # 添加命令行参数--start_epoch，用于指定起始训练轮次，默认为1
    parser.add_argument(
        "--start_epoch", dest="start_epoch", help="starting epoch", default=1, type=int
    )
    # 添加命令行参数--max_epochs，用于指定最大训练轮次，默认为10
    parser.add_argument(
        "--max_epochs",
        dest="max_epochs",
        help="max epoch for train",
        default=10,
        type=int,
    )
    # 添加命令行参数--disp_interval，用于指定显示训练信息的间隔迭代次数，默认为100
    parser.add_argument(
        "--disp_interval",
        dest="disp_interval",
        help="number of iterations to display",
        default=100,
        type=int,
    )
    # 添加命令行参数--checkpoint_interval，用于指定保存检查点的间隔迭代次数，默认为1
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        help="number of iterations to save checkpoint",
        default=1,
        type=int,
    )
    # 添加命令行参数--save_dir，用于指定保存模型的目录，默认为./data
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="directory to save models",
        default="./data",
        type=str,
    )
    # 添加命令行参数--nw，用于指定加载数据的线程数，默认为0
    parser.add_argument(
        "--nw",
        dest="num_workers",
        help="number of worker to load data",
        default=0,
        type=int,
    )
    # 添加命令行参数--ls，用于指定是否使用大图像尺度，默认为False
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    # 添加命令行参数--bs，用于指定批量大小，默认为1
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=2, type=int
    )
    # 添加命令行参数--cag，用于指定是否进行类别无关的边界框回归，默认为False
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )
    
    
    # 添加命令行参数--pretrained_path，用于指定预训练模型的路径，默认为./data/pretrained_model/resnet101_caffe.pth
    parser.add_argument(
        "--pretrained_path",
        dest="pretrained_path",
        help="vgg16, res101",
        default="./data/pretrained_model/resnet101_caffe.pth",
        type=str,
    )
    # 添加命令行参数--cuda，用于指定是否使用CUDA，默认为True
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", default = "True", action="store_true"
    )
    # 添加命令行参数--o，用于指定优化器，默认为sgd
    parser.add_argument(
        "--o", dest="optimizer", help="training optimizer", default="sgd", type=str
    )
    # 添加命令行参数--lr，用于指定初始学习率，默认为0.002
    parser.add_argument(
        "--lr", dest="lr", help="starting learning rate", default=0.002, type=float
    )
    # 添加命令行参数--lr_decay_step，用于指定学习率衰减的步长，单位为迭代次数，默认为5
    parser.add_argument(
        "--lr_decay_step",
        dest="lr_decay_step",
        help="step to do learning rate decay, unit is iter",
        default=5,
        type=int,
    )
    # 添加命令行参数--lr_decay_gamma，用于指定学习率衰减的比例，默认为0.1
    parser.add_argument(
        "--lr_decay_gamma",
        dest="lr_decay_gamma",
        help="learning rate decay ratio",
        default=0.1,
        type=float,
    )
    # 添加命令行参数--lamda，用于指定DA损失参数，默认为0.1
    parser.add_argument(
        "--lamda", dest="lamda", help="DA loss param", default=0.1, type=float
    )
    # 添加命令行参数--alpha，用于指定IDA损失参数，默认为10
    parser.add_argument(
        "--alpha", dest="alpha", help="IDA loss param", default=10, type=float
    )
    # set training session
    parser.add_argument(
        "--s", dest="session", help="training session", default=1, type=int
    )
    # resume trained model
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=False, type=bool
    )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="resume from which model",
        default="",
        type=str,
    )

    # setting display config
    
    parser.add_argument(
        "--dataset_s",
        dest="dataset_s",
        help="training dataset",
        default="HRSC",
        type=str,
    )
    parser.add_argument(
        "--dataset_t",
        dest="dataset_t",
        help="training target dataset",
        default="SSDD",
        type=str,
    )

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):

        # 初始化采样器，设置训练数据大小和批次大小
        self.num_data = train_size  # 训练数据总大小
        self.num_per_batch = int(train_size / batch_size)  # 每个批次包含的数据数量
        self.batch_size = batch_size  # 批次大小
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()  # 创建一个从0到batch_size-1的序列，并调整为1行batch_size列的二维张量
        self.leftover_flag = False  # 标记是否有剩余数据
        if train_size % batch_size:  # 如果训练数据不能整除批次大小
            self.leftover = torch.arange(
                self.num_per_batch * batch_size, train_size
            ).long()  # 计算剩余数据的索引
            self.leftover_flag = True  # 设置剩余数据标记为True

    def __iter__(self):

        # 迭代器方法，用于生成数据索引
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size  # 生成一个随机排列的索引，并调整为num_per_batch行1列的二维张量，然后乘以批次大小
        self.rand_num = (
            rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        )  # 将随机索引扩展为num_per_batch行batch_size列，并加上范围索引

        self.rand_num_view = self.rand_num.view(-1)  # 将二维张量展平为一维张量

        if self.leftover_flag:  # 如果有剩余数据
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)  # 将剩余数据索引拼接到随机索引后面

        return iter(self.rand_num_view)  # 最后生成的是一个包含所有数据索引的迭代器

    def __len__(self):
        # 返回数据总大小
        return self.num_data


if __name__ == "__main__":

    args = parse_args()

    print("Called with args:")
    print(args)

    if args.dataset_s == "LEVIR":
        args.imdb_name = "LEVIR_train"
        args.imdbval_name = "LEVIR_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset_s == "HRSC":
        args.imdb_name = "HRSC_train"
        args.imdbval_name = "HRSC_train"
        # ANCHOR_SCALES指的是锚框的基本大小， ANCHOR_RATIOS指的是锚框的宽高比， MAX_NUM_GT_BOXES指的是每张图像中最大目标框数量
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset_s == "DIOR":
        args.imdb_name = "DIOR_train"
        args.imdbval_name = "DIOR_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    if args.dataset_t == "SSDD":
        args.imdb_name_target = "SSDD_train"
        args.imdbval_name_target = "SSDD_train"
        args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset_t == "SAR":
        args.imdb_name_target = "SAR_train"
        args.imdbval_name_target = "SAR_train"
        args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # 如果是large_scale，则使用大尺度模型训练，但是在本训练中为False，使用小尺度模型训练
    # 以faster-rcnn为例，如果是large_scale，则使用faster_rcnn_ls.yml，如果不是large_scale，则使用faster_rcnn.yml
    args.cfg_file = (
        "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else "cfgs/{}.yml".format(args.net)
    )

    # 如果cfg_file不为空，cfg_file是.yml模型文件，则使用cfg_file中的配置
    # 如果set_cfgs不为空，set_cfgs指的是刚才的ANCHOR那几个参数，则使用set_cfgs中的配置
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    # 美观的打印配置信息，直接打印 print(cfg) 可能输出不友好的字符串（如内存地址或未格式化的内容）。
    # 使用 pprint 可以按缩进层级展示嵌套的配置参数。
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 在训练阶段启用图像水平翻转作为数据增强手段, 验证时应该关闭
    # 使用 GPU 加速非极大值抑制
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    
    # s_imdb指的是数据集元信息管理对象（包含数据集名称、类别、路径等）
    # s_roidb指的是处理后的标注信息列表（每个元素对应一张图像的标注）
    # s_ratio_list指的是每张图像的宽高比例（width/height）列表
    # s_ratio_index指的是按宽高比例排序后的索引列表（用于优化数据加载顺序）  
    s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.imdb_name)
    s_train_size = len(s_roidb)  # add flipped         image_index*2

    t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.imdb_name_target)
    t_train_size = len(t_roidb)  # add flipped         image_index*2

    print("源域 {:d} 目标域 {:d} roidb entries".format(s_train_size, t_train_size))

    # 创建数据加载器，用于批量加载训练数据
    # s_dataloader和t_dataloader分别对应源域和目标域的数据加载器
    # sampler函数用于生成数据加载时的采样器，用于控制数据加载的顺序和批次大小)

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    s_sampler_batch = sampler(s_train_size, args.batch_size)
    t_sampler_batch = sampler(t_train_size, args.batch_size)

    s_dataset = roibatchLoader(
        s_roidb,
        s_ratio_list,
        s_ratio_index,
        args.batch_size,
        s_imdb.num_classes,
        training=True,
    )

    s_dataloader = torch.utils.data.DataLoader(
        s_dataset,
        batch_size=args.batch_size,
        sampler=s_sampler_batch,
        num_workers=args.num_workers,
    )

    t_dataset = roibatchLoader(
        t_roidb,
        t_ratio_list,
        t_ratio_index,
        args.batch_size,
        t_imdb.num_classes,
        training=True,
    )

    t_dataloader = torch.utils.data.DataLoader(
        t_dataset,
        batch_size=args.batch_size,
        sampler=t_sampler_batch,
        num_workers=args.num_workers,
    )


    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1) #  创建一个大小为1的FloatTensor，用于存储源域图像数据
    im_info = torch.FloatTensor(1) #  用于存储源域图像信息
    num_boxes = torch.LongTensor(1) # 用于存储源域图像中的目标框数量
    gt_boxes = torch.FloatTensor(1) # 用于存储源域图像中的真实目标框
    need_backprop = torch.FloatTensor(1) # 用于指示是否需要反向传播（0/1）

    tgt_im_data = torch.FloatTensor(1) # 用于存储目标域图像数据
    tgt_im_info = torch.FloatTensor(1) # 用于存储目标域图像信息
    tgt_num_boxes = torch.LongTensor(1) # 用于存储目标域图像中的目标框数量
    tgt_gt_boxes = torch.FloatTensor(1) # 用于存储目标域图像中的真实目标框
    tgt_need_backprop = torch.FloatTensor(1) # 用于指示目标域图像是否需要反向传播（0/1）

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        need_backprop = need_backprop.cuda()

        tgt_im_data = tgt_im_data.cuda()
        tgt_im_info = tgt_im_info.cuda()
        tgt_num_boxes = tgt_num_boxes.cuda()
        tgt_gt_boxes = tgt_gt_boxes.cuda()
        tgt_need_backprop = tgt_need_backprop.cuda()

    # make variable
    # 将图像数据转换为Variable类型，以便于后续的自动求导操作
    im_data = Variable(im_data)
    # 将图像信息转换为Variable类型，包含图像的尺寸等信息
    im_info = Variable(im_info)
    # 将目标框的数量转换为Variable类型，用于后续的批处理操作
    num_boxes = Variable(num_boxes)
    # 将真实的目标框数据转换为Variable类型，用于训练过程中的损失计算
    gt_boxes = Variable(gt_boxes)
    # 将是否需要反向传播的标志转换为Variable类型，控制是否进行反向传播
    need_backprop = Variable(need_backprop)

    # 将目标图像数据转换为Variable类型，用于目标图像的处理
    tgt_im_data = Variable(tgt_im_data)
    # 将目标图像信息转换为Variable类型，包含目标图像的尺寸等信息
    tgt_im_info = Variable(tgt_im_info)
    # 将目标图像中目标框的数量转换为Variable类型，用于目标图像的批处理操作
    tgt_num_boxes = Variable(tgt_num_boxes)
    # 将目标图像中真实的目标框数据转换为Variable类型，用于目标图像的训练过程中的损失计算
    tgt_gt_boxes = Variable(tgt_gt_boxes)
    # 将目标图像是否需要反向传播的标志转换为Variable类型，控制目标图像是否进行反向传播
    tgt_need_backprop = Variable(tgt_need_backprop)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            s_imdb.classes,
            pretrained=True,
            pretrained_path=args.pretrained_path,
            class_agnostic=args.class_agnostic,
        )
    elif args.net == "res101":
        fasterRCNN = resnet(
            s_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic
        )
    elif args.net == "res50":
        fasterRCNN = resnet(
            s_imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic
        )
    elif args.net == "res152":
        fasterRCNN = resnet(
            s_imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture() #  创建Faster R-CNN的网络架构

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = [] #  初始化参数列表
    # 遍历fasterRCNN模型的所有命名参数
    for key, value in dict(fasterRCNN.named_parameters()).items():
        # 检查参数是否需要梯度更新
        if value.requires_grad:
            # 如果参数名称中包含"bias"（偏置项）
            if "bias" in key:
                # 将偏置项参数添加到params列表中，并设置其学习率和权重衰减
                params += [
                    {
                        "params": [value],  # 当前参数
                        "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),  # 偏置项的学习率为原学习率的两倍
                        "weight_decay": cfg.TRAIN.BIAS_DECAY  # 偏置项的权重衰减
                        and cfg.TRAIN.WEIGHT_DECAY  # 如果BIAS_DECAY为True，则使用WEIGHT_DECAY，否则为0
                        or 0,
                    }
                ]
            else:
                # 将非偏置项参数添加到params列表中，并设置其学习率和权重衰减
                params += [
                    {
                        "params": [value],  # 当前参数
                        "lr": lr,  # 使用默认的学习率
                        "weight_decay": cfg.TRAIN.WEIGHT_DECAY,  # 使用配置中的权重衰减
                    }
                ]

    if args.optimizer == "adam":
        #lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM) #  使用SGD优化器，并设置动量参数

    if args.resume:
        load_name = os.path.join(output_dir, args.model_name)
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint["session"]
        args.start_epoch = checkpoint["epoch"]
        fasterRCNN.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        lr = optimizer.param_groups[0]["lr"] #  获取当前的学习率
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (load_name))

    if args.cuda:
        fasterRCNN.cuda()


    iters_per_epoch = int(min(s_train_size,t_train_size) / args.batch_size) #  计算每个epoch的迭代次数
    test_time_start=0 #  初始化测试时间开始点
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train() #  将fasterRCNN模型设置为训练模式，启用dropout和batch normalization
        loss_temp = 0 #  初始化临时损失变量，用于累加当前epoch中的损失值
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0: #  检查当前epoch是否是学习率衰减的步长
            adjust_learning_rate(optimizer, args.lr_decay_gamma) #  调整学习率，乘以衰减因子
            lr *= args.lr_decay_gamma #  更新学习率变量，用于后续显示或计算

        data_iter_s = iter(s_dataloader) #  创建源数据集的迭代器
        data_iter_t = iter(t_dataloader) #  创建目标数据集的迭代器
        for step in range(iters_per_epoch): #  遍历每个epoch的迭代次数
            try:
                data = next(data_iter_s)
            except:
                data_iter_s = iter(s_dataloader)
                data = next(data_iter_s)
            try:
                tgt_data = next(data_iter_t)
            except:
                data_iter_t = iter(t_dataloader)
                tgt_data = next(data_iter_t)

            im_data.resize_(data[0].size()).copy_(data[0])  # change holder size
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            need_backprop.resize_(data[4].size()).copy_(data[4])
            tgt_im_data.resize_(tgt_data[0].size()).copy_(
                tgt_data[0]
            )  # change holder size
            tgt_im_info.resize_(tgt_data[1].size()).copy_(tgt_data[1])
            tgt_gt_boxes.resize_(tgt_data[2].size()).copy_(tgt_data[2])
            tgt_num_boxes.resize_(tgt_data[3].size()).copy_(tgt_data[3])
            tgt_need_backprop.resize_(tgt_data[4].size()).copy_(tgt_data[4])

            """   faster-rcnn loss + DA loss for source and DA loss for target    """
            fasterRCNN.zero_grad()
            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls,
                rpn_loss_box,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                DA_img_loss_cls,
                DA_ins_loss_cls,
                tgt_DA_img_loss_cls,
                tgt_DA_ins_loss_cls,
                DA_cst_loss,
                tgt_DA_cst_loss
            ) = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                need_backprop,
                tgt_im_data,
                tgt_im_info,
                tgt_gt_boxes,
                tgt_num_boxes,
                tgt_need_backprop
            )

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
            + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
            +args.lamda*(DA_img_loss_cls.mean()+DA_ins_loss_cls.mean() \
            +tgt_DA_img_loss_cls.mean()+tgt_DA_ins_loss_cls.mean()+DA_cst_loss.mean()+tgt_DA_cst_loss.mean())
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            #if args.net == "vgg16":
            clip_gradient(fasterRCNN, 10.0)
            optimizer.step()

            if step % args.disp_interval == 0 and step>0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval + 1

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                loss_DA_img_cls = (
                    args.lamda
                    * (DA_img_loss_cls.item() + tgt_DA_img_loss_cls.item())
                    / 2
                )
                loss_DA_ins_cls = (
                    args.lamda
                    * (DA_ins_loss_cls.item() + tgt_DA_ins_loss_cls.item())
                    / 2
                )
                loss_DA_cst = (
                    args.alpha * (DA_cst_loss.item() + tgt_DA_cst_loss.item()) / 2
                )
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print(
                    "[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                    % (args.session, epoch, step, iters_per_epoch, loss_temp, lr)
                )
                print(
                    "\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start)
                )
                print(
                    "\t\t\t rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f,\n\t\t\timg_loss %.4f,ins_loss %.4f,cst_loss %.4f"
                    % (
                        loss_rpn_cls,
                        loss_rpn_box,
                        loss_rcnn_cls,
                        loss_rcnn_box,
                        loss_DA_img_cls,
                        loss_DA_ins_cls,
                        loss_DA_cst,
                    )
                )
                test_time_start=test_time_start+end - start
                print('totaltime',test_time_start)

                loss_temp = 0
                start = time.time()

        if epoch > 0:
            save_name = os.path.join(
                output_dir, "{}.pth".format(args.dataset_t + "_" + str(epoch)),
            )
            save_checkpoint(
                {
                    "session": args.session,
                    "iter": step + 1,
                    "model": fasterRCNN.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pooling_mode": cfg.POOLING_MODE,
                    "class_agnostic": args.class_agnostic,
                },
                save_name,
            )
            print("save model: {}".format(save_name))
