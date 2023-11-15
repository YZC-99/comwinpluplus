import random
from tqdm import tqdm
from dataloader.fundus import SemiDataset
from dataloader.samplers import LabeledBatchSampler,UnlabeledBatchSampler
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.com_co_traning_net import TriUNet_before2
from utils import ramps
import time

import os
import shutil
import logging
import sys

from model.unet import ResUNet_dsba_before2

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--amp',type=bool,default=True)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--base_lr',type=float,default=0.01)

parser.add_argument('--batch_size',type=int,default=4)

parser.add_argument('--image_size',type=int,default=64)

parser.add_argument('--labeled_bs',type=int,default=2)
parser.add_argument('--labeled_num',type=int,default=100)
parser.add_argument('--total_num',type=int,default=360)
parser.add_argument('--scale_num',type=int,default=2)
parser.add_argument('--max_iterations',type=int,default=100)

parser.add_argument('--with_dice',type=bool,default=False)

parser.add_argument('--cps_la_weight_final', type=float,  default=0.1)
parser.add_argument('--cps_la_rampup_scheme', type=str,  default='None', help='cps_la_rampup_scheme')
parser.add_argument('--cps_la_rampup',type=float,  default=40.0)
parser.add_argument('--cps_la_with_dice',type=bool,default=False)

parser.add_argument('--cps_un_weight_final', type=float,  default=0.1, help='consistency')
parser.add_argument('--cps_un_rampup_scheme', type=str,  default='None', help='cps_rampup_scheme')
parser.add_argument('--cps_un_rampup', type=float,  default=40.0, help='cps_rampup')
parser.add_argument('--cps_un_with_dice', type=bool,  default=True, help='cps_un_with_dice')

parser.add_argument('--exp',type=str,default='refuge400')


def get_unsup_cont_weight(epoch, weight, scheme, ramp_up_or_down ):
    if  scheme == 'sigmoid_rampup':
        return weight * ramps.sigmoid_rampup(epoch, ramp_up_or_down)
    elif scheme == 'linear_rampup':
        return weight * ramps.linear_rampup(epoch, ramp_up_or_down)
    elif scheme == 'log_rampup':
        return weight * ramps.log_rampup(epoch, ramp_up_or_down)
    elif scheme == 'exp_rampup':
        return weight * ramps.exp_rampup(epoch, ramp_up_or_down)
    elif scheme == 'quadratic_rampdown':
        return weight * ramps.quadratic_rampdown(epoch, ramp_up_or_down)
    elif scheme == 'cosine_rampdown':
        return weight * ramps.cosine_rampdown(epoch, ramp_up_or_down)
    else:
        return weight


def get_supervised_loss(outputs, label_batch,  with_dice=True):
    loss_seg = F.cross_entropy(outputs, label_batch)
    # outputs_soft = F.softmax(outputs, dim=1)
    # if with_dice:
    #     loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
    #     supervised_loss = 0.5 * (loss_seg + loss_seg_dice)
    # else:
    loss_seg_dice = torch.zeros([1]).cuda()
    supervised_loss = loss_seg + loss_seg_dice
    return supervised_loss, loss_seg, loss_seg_dice

def pseudo_labeling_from_most_confident_prediction(pred_1, pred_2, max_1, max_2):
        # stack很有意思，因为是堆叠，所以会产生一个新的维度，dim就是指定这个维度应该放在哪里
        prob_all_ex_3 = torch.stack([pred_1, pred_2], dim=2)  # bs, n_c, n_branch - 1, h, w
        max_all_ex_3 = torch.stack([max_1, max_2], dim=1)  # bs, n_branch - 1, h, w
        # 获取堆叠后的最大分数,注意，这里返回后的维度，现在分数不再是softmax的，而是来自不同branch的最大分数
        max_conf_each_branch_ex_3, _ = torch.max(prob_all_ex_3, dim=1)  # bs, n_branch - 1, h, w
        # 再计算伪标签的值和索引，branch_id_max_conf_ex_3代表的是第几个branch
        max_conf_ex_3, branch_id_max_conf_ex_3 = torch.max(max_conf_each_branch_ex_3, dim=1,
                                                              keepdim=True)  # bs, h, w
        # branch_id_max_conf_ex_3是索引，因此可以通过索引从堆叠的标签中获取最终通过竞争机制得到的伪标签
        pseudo_12 = torch.gather(max_all_ex_3, dim=1, index=branch_id_max_conf_ex_3)[:, 0]
        # 返回的是一个布尔值，因此从max_conf_ex_3的伪标签
        max_conf_fg_ex_3 = max_conf_ex_3[:, 0][pseudo_12 == 1]
        try:
            mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3 = torch.mean(
                max_conf_fg_ex_3).detach().cpu(), torch.var(max_conf_fg_ex_3).detach().cpu(), torch.min(
                max_conf_fg_ex_3).detach().cpu(), torch.max(max_conf_fg_ex_3).detach().cpu()
        except:
            mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3 = 0, 0, 0, 0
        return pseudo_12, branch_id_max_conf_ex_3, [mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3]



args = parser.parse_args()
snapshot_path = "./exp/" + args.exp + "/"
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs


if __name__ == '__main__':
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # init model
    scale_num = 2
    model = TriUNet_before2(num_classes=args.num_classes)
    model.cuda()

    # init dataset
    labeled_dataset = SemiDataset(name='./dataset/semi_refuge400',
                                  root='D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE',
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='labeled.txt')

    unlabeled_dataset = SemiDataset(name='./dataset/semi_refuge400',
                                    root='D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE',
                                    mode='semi_train',
                                    size=args.image_size,
                                    id_path='unlabeled.txt')

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num,args.total_num))
    labeled_batch_sampler = LabeledBatchSampler(labeled_idxs,args.labeled_bs)
    unlabeled_batch_sampler = UnlabeledBatchSampler(labeled_idxs, args.batch_size - args.labeled_bs)

    # init dataloader
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    labeledtrainloader = DataLoader(labeled_dataset, batch_sampler=labeled_batch_sampler, num_workers=0, pin_memory=True,
                                    worker_init_fn=worker_init_fn)
    unlabeledtrainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_batch_sampler, num_workers=0,
                                      pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    # init optimizer
    optimizer_1 = optim.SGD(model.branch1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model.branch2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_3 = optim.SGD(model.branch3.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    # init summarywriter
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_iterations // len(labeledtrainloader) + 1
    lr_ = args.base_lr
    model.train()

    # AMP
    scaler = GradScaler()

    # 开始训练
    for epoch_num in tqdm(range(max_epoch),ncols=70):
        time1 = time.time()
        for i_batch,(labeled_sampled_batch, unlabeled_sampled_batch) in enumerate(zip(labeledtrainloader,unlabeledtrainloader)):
            time2 = time.time()

            unlabeled_batch, unlabel_label_batch = unlabeled_sampled_batch['image'], unlabeled_sampled_batch['label']
            labeled_batch, label_label_batch = labeled_sampled_batch['image'], labeled_sampled_batch['label']
            # push to gpu
            unlabeled_batch, unlabel_label_batch = unlabeled_batch.cuda(), unlabel_label_batch.cuda()
            labeled_batch, label_label_batch = labeled_batch.cuda(), label_label_batch.cuda()

            forward_step_num = args.scale_num * 2 - 1
            # organize labels and input data at diff scales
            label_label_batch_list, unlabel_label_batch_list = [label_label_batch], [unlabel_label_batch]
            labeled_batch_list, unlabeled_batch_list = [labeled_batch], [unlabeled_batch]

            # todo:scale_num = 2 实质上就是做图像的缩放
            label_label_batch_cur_res = F.interpolate(label_label_batch_list[0].unsqueeze(1).float(),size=(args.image_size // 2,args.image_size // 2),mode='nearest')
            unlabeld_label_batch_cur_res = F.interpolate(unlabel_label_batch_list[0].unsqueeze(1).float(),size=(args.image_size // 2,args.image_size // 2),mode='nearest')
            label_label_batch_list.insert(0, label_label_batch_cur_res.squeeze(1).long())
            unlabel_label_batch_list.insert(0, unlabeld_label_batch_cur_res.squeeze(1).long())

            labeled_batch_cur_res = F.interpolate(labeled_batch_list[0],size=(args.image_size // 2,args.image_size // 2),mode='nearest')
            unlabeled_batch_cur_res = F.interpolate(unlabeled_batch_list[0],size=(args.image_size // 2,args.image_size // 2),mode='nearest')
            labeled_batch_list.insert(0, labeled_batch_cur_res)
            unlabeled_batch_list.insert(0, unlabeled_batch_cur_res)


            logits_sup_1_list, logits_sup_2_list, logits_sup_3_list = [], [], []
            logits_unsup_1_list, logits_unsup_2_list, logits_unsup_3_list = [], [], []
            max_la_1_list, max_la_2_list, max_la_3_list = [], [], []
            max_un_1_list, max_un_2_list, max_un_3_list = [], [], []
            loss_sup_1, loss_sup_2, loss_sup_3 = 0, 0, 0
            cps_la_loss, cps_un_loss = 0, 0

            # step1 forward propagation
            scale_id = 0
            forward_step = 1
            with autocast():
                x1_sup_1, dec3_sup_1, out_at2_sup_1 = model(labeled_batch, step=1, forward_step=forward_step)
                x1_unsup_1, dec3_unsup_1, out_at2_unsup_1 = model(unlabeled_batch, step=1, forward_step=forward_step)
                x1_sup_2, dec3_sup_2, out_at2_sup_2 = model(labeled_batch, step=2, forward_step=forward_step)
                x1_unsup_2, dec3_unsup_2, out_at2_unsup_2 = model(unlabeled_batch, step=2, forward_step=forward_step)
                x1_sup_3, dec3_sup_3, out_at2_sup_3 = model(labeled_batch, step=3, forward_step=forward_step)
                x1_unsup_3, dec3_unsup_3, out_at2_unsup_3 = model(unlabeled_batch, step=3, forward_step=forward_step)
                # gather all logits at current resolution scale
                logits_sup_1_list.append(out_at2_sup_1)
                logits_sup_2_list.append(out_at2_sup_2)
                logits_sup_3_list.append(out_at2_sup_3)
                logits_unsup_1_list.append(out_at2_unsup_1)
                logits_unsup_2_list.append(out_at2_unsup_2)
                logits_unsup_3_list.append(out_at2_unsup_3)
                # gather supervised loss at current resolution scale
                loss_sup_1 += \
                get_supervised_loss(logits_sup_1_list[scale_id], label_label_batch_list[scale_id], args.with_dice)[0]
                loss_sup_2 += \
                get_supervised_loss(logits_sup_2_list[scale_id], label_label_batch_list[scale_id], args.with_dice)[0]
                loss_sup_3 += \
                get_supervised_loss(logits_sup_3_list[scale_id], label_label_batch_list[scale_id], args.with_dice)[0]

                # generate pseudo labels for labeled data at current resolution scale
                _, max_la_1 = torch.max(logits_sup_1_list[scale_id], dim=1)
                _, max_la_2 = torch.max(logits_sup_2_list[scale_id], dim=1)
                _, max_la_3 = torch.max(logits_sup_3_list[scale_id], dim=1)
                max_la_1_list.append(max_la_1.long());
                max_la_2_list.append(max_la_2.long());
                max_la_3_list.append(max_la_3.long())
                pred_sup_1 = F.softmax(logits_sup_1_list[scale_id], dim=1)
                pred_sup_2 = F.softmax(logits_sup_2_list[scale_id], dim=1)
                pred_sup_3 = F.softmax(logits_sup_3_list[scale_id], dim=1)
                # 根据伪标签使用竞争机制产生新的伪标签
                pseudo_la_12, branch_id_la_max_conf_ex_3, _ = pseudo_labeling_from_most_confident_prediction(pred_sup_1,
                                                                                                             pred_sup_2,
                                                                                                             max_la_1, max_la_2)
                pseudo_la_13, branch_id_la_max_conf_ex_2, _ = pseudo_labeling_from_most_confident_prediction(pred_sup_1,
                                                                                                             pred_sup_3,
                                                                                                             max_la_1, max_la_3)
                pseudo_la_23, branch_id_la_max_conf_ex_1, _ = pseudo_labeling_from_most_confident_prediction(pred_sup_2,
                                                                                                             pred_sup_3,
                                                                                                             max_la_2, max_la_3)


                # 进行低分辨率监督
                cps_la_loss += \
                    get_supervised_loss(logits_sup_1_list[scale_id], pseudo_la_23, args.cps_la_with_dice)[0] + \
                    get_supervised_loss(logits_sup_2_list[scale_id], pseudo_la_13, args.cps_la_with_dice)[0] + \
                    get_supervised_loss(logits_sup_3_list[scale_id], pseudo_la_12, args.cps_la_with_dice)[0]

                # generate pseudo labels for unlabeled at current resolution scale
                _, max_un_1 = torch.max(logits_unsup_1_list[scale_id], dim=1)
                _, max_un_2 = torch.max(logits_unsup_2_list[scale_id], dim=1)
                _, max_un_3 = torch.max(logits_unsup_3_list[scale_id], dim=1)
                max_un_1_list.append(max_un_1.long());
                max_un_2_list.append(max_un_2.long());
                max_un_3_list.append(max_un_3.long())
                pred_unsup_1 = F.softmax(logits_unsup_1_list[scale_id], dim=1)
                pred_unsup_2 = F.softmax(logits_unsup_2_list[scale_id], dim=1)
                pred_unsup_3 = F.softmax(logits_unsup_3_list[scale_id], dim=1)
                pseudo_un_12, branch_id_un_max_conf_ex_3, _ = pseudo_labeling_from_most_confident_prediction(pred_unsup_1,
                                                                                                             pred_unsup_2,
                                                                                                             max_un_1, max_un_2)
                pseudo_un_13, branch_id_un_max_conf_ex_2, _ = pseudo_labeling_from_most_confident_prediction(pred_unsup_1,
                                                                                                             pred_unsup_3,
                                                                                                             max_un_1, max_un_3)
                pseudo_un_23, branch_id_un_max_conf_ex_1, _ = pseudo_labeling_from_most_confident_prediction(pred_unsup_2,
                                                                                                             pred_unsup_3,
                                                                                                             max_un_2, max_un_3)

                # todo: step2 forward propagation
                # 这一步的主要作用是根据竞争机制产生的低分辨率伪标签来实现边界感知attention，从而获得attention后的特征
                forward_step = 2
                dec3_after_sup_1 = model(dec3_sup_1, pseudo_labels=pseudo_la_23, step=1, forward_step=forward_step)
                dec3_after_unsup_1 = model(dec3_unsup_1, pseudo_labels=pseudo_un_23, step=1, forward_step=forward_step)
                dec3_after_sup_2 = model(dec3_sup_2, pseudo_labels=pseudo_la_13, step=2, forward_step=forward_step)
                dec3_after_unsup_2 = model(dec3_unsup_2, pseudo_labels=pseudo_un_13, step=2, forward_step=forward_step)
                dec3_after_sup_3 = model(dec3_sup_3, pseudo_labels=pseudo_la_12, step=3, forward_step=forward_step)
                dec3_after_unsup_3 = model(dec3_unsup_3, pseudo_labels=pseudo_un_12, step=3, forward_step=forward_step)

                # gather cps loss for unlabeled data at current resolution scale
                # 计算未标记数据的深度监督损失
                cps_un_loss += \
                    get_supervised_loss(logits_unsup_1_list[scale_id], pseudo_un_23, args.cps_un_with_dice)[0] + \
                    get_supervised_loss(logits_unsup_2_list[scale_id], pseudo_un_13, args.cps_un_with_dice)[0] + \
                    get_supervised_loss(logits_unsup_3_list[scale_id], pseudo_un_12, args.cps_un_with_dice)[0]

                # todo: step3 forward propagation at full scale
                forward_step = 3
                scale_id = 1
                logits_sup_1 = model([x1_sup_1, dec3_after_sup_1], step=1, forward_step=forward_step)
                logits_unsup_1 = model([x1_unsup_1, dec3_after_unsup_1], step=1, forward_step=forward_step)
                logits_sup_2 = model([x1_sup_2, dec3_after_sup_2], step=2, forward_step=forward_step)
                logits_unsup_2 = model([x1_unsup_2, dec3_after_unsup_2], step=2, forward_step=forward_step)
                logits_sup_3 = model([x1_sup_3, dec3_after_sup_3], step=3, forward_step=forward_step)
                logits_unsup_3 = model([x1_unsup_3, dec3_after_unsup_3], step=3, forward_step=forward_step)
                # todo: step3 gather all logits at full scale
                logits_sup_1_list.append(logits_sup_1)
                logits_sup_2_list.append(logits_sup_2)
                logits_sup_3_list.append(logits_sup_3)
                logits_unsup_1_list.append(logits_unsup_1)
                logits_unsup_2_list.append(logits_unsup_2)
                logits_unsup_3_list.append(logits_unsup_3)
                # todo: step3 gather supervised loss at full scale
                loss_sup_1 += \
                get_supervised_loss(logits_sup_1_list[scale_id], label_label_batch_list[scale_id], args.with_dice)[0]
                loss_sup_2 += \
                get_supervised_loss(logits_sup_2_list[scale_id], label_label_batch_list[scale_id], args.with_dice)[0]
                loss_sup_3 += \
                get_supervised_loss(logits_sup_3_list[scale_id], label_label_batch_list[scale_id], args.with_dice)[0]

                # todo: step3 generate pseudo labels for unlabeled data at full scale
                _, max_un_1 = torch.max(logits_unsup_1_list[scale_id], dim=1)
                _, max_un_2 = torch.max(logits_unsup_2_list[scale_id], dim=1)
                _, max_un_3 = torch.max(logits_unsup_3_list[scale_id], dim=1)
                max_un_1_list.append(max_un_1.long());
                max_un_2_list.append(max_un_2.long());
                max_un_3_list.append(max_un_3.long())
                pred_unsup_1 = F.softmax(logits_unsup_1_list[scale_id], dim=1)
                pred_unsup_2 = F.softmax(logits_unsup_2_list[scale_id], dim=1)
                pred_unsup_3 = F.softmax(logits_unsup_3_list[scale_id], dim=1)
                pseudo_un_12, branch_id_un_max_conf_ex_3, statistics_un_3 = pseudo_labeling_from_most_confident_prediction(pred_unsup_1,
                                                                                                                           pred_unsup_2,
                                                                                                                           max_un_1,
                                                                                                                           max_un_2)
                pseudo_un_13, branch_id_un_max_conf_ex_2, statistics_un_2 = pseudo_labeling_from_most_confident_prediction(pred_unsup_1,
                                                                                                                           pred_unsup_3,
                                                                                                                           max_un_1,
                                                                                                                           max_un_3)
                pseudo_un_23, branch_id_un_max_conf_ex_1, statistics_un_1 = pseudo_labeling_from_most_confident_prediction(pred_unsup_2,
                                                                                                                           pred_unsup_3,
                                                                                                                           max_un_2,
                                                                                                                           max_un_3)
                # todo: step3 evaluate statistics of pseudo labels for unlabeled data at full scale
                mean_un_max_conf_fg_ex_1, var_un_max_conf_fg_ex_1, min_un_max_conf_fg_ex_1, max_un_max_conf_fg_ex_1 = statistics_un_1
                mean_un_max_conf_fg_ex_2, var_un_max_conf_fg_ex_2, min_un_max_conf_fg_ex_2, max_un_max_conf_fg_ex_2 = statistics_un_2
                mean_un_max_conf_fg_ex_3, var_un_max_conf_fg_ex_3, min_un_max_conf_fg_ex_3, max_un_max_conf_fg_ex_3 = statistics_un_3
                # todo: gather cps loss for unlabeled data at current resolution scale
                cps_un_loss += \
                    get_supervised_loss(logits_unsup_1_list[scale_id], pseudo_un_23, args.cps_un_with_dice)[0] + \
                    get_supervised_loss(logits_unsup_2_list[scale_id], pseudo_un_13, args.cps_un_with_dice)[0] + \
                    get_supervised_loss(logits_unsup_3_list[scale_id], pseudo_un_12, args.cps_un_with_dice)[0]

                """gather all losses"""
                cps_la_weight = get_unsup_cont_weight(iter_num // 150, args.cps_la_weight_final,
                                                      scheme=args.cps_la_rampup_scheme, ramp_up_or_down=args.cps_la_rampup)
                cps_un_weight = get_unsup_cont_weight(iter_num // 150, args.cps_un_weight_final,
                                                      scheme=args.cps_un_rampup_scheme, ramp_up_or_down=args.cps_un_rampup)

                loss = loss_sup_1 + loss_sup_2 + loss_sup_3 + cps_la_loss * cps_la_weight + cps_un_loss * cps_un_weight

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()

            # AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer_1)
            scaler.step(optimizer_2)
            scaler.step(optimizer_3)
            scaler.update()

            #
            # optimizer_1.zero_grad()
            # optimizer_2.zero_grad()
            # optimizer_3.zero_grad()
            # loss.backward()
            # optimizer_1.step()
            # optimizer_2.step()
            # optimizer_3.step()

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_sup_1', loss_sup_1, iter_num)
            writer.add_scalar('loss/loss_sup_2', loss_sup_2, iter_num)
            writer.add_scalar('loss/loss_sup_3', loss_sup_3, iter_num)
            writer.add_scalar('loss/loss_cps_la', cps_la_loss, iter_num)
            writer.add_scalar('loss/cps_weight_la', cps_la_weight, iter_num)
            writer.add_scalar('loss/loss_cps_un', cps_un_loss, iter_num)
            writer.add_scalar('loss/cps_weight_un', cps_un_weight, iter_num)


            basic_info = 'iteration %d : loss : %f loss_sup_1: %f, loss_sup_2: %f, loss_sup_3: %f, cps_la_loss: %f,  cps_la_weight: %f, cps_un_loss: %f,  cps_un_weight: %f '% \
                         (iter_num, loss.item(), loss_sup_1.item(), loss_sup_2.item(), loss_sup_3.item(), cps_la_loss.item(), cps_la_weight, cps_un_loss.item(),  cps_un_weight)



            iter_num = iter_num + 1

            logging.info(basic_info)
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_3.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save({'model': model.state_dict(),
                            'max_iterations': max_iterations}, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save({'model': model.state_dict(),
                'max_iterations': max_iterations}, save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()