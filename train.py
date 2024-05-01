import time
import datetime
import torch.nn as nn
import torch.autograd
from torch import optim
import torch.nn.functional as F
import yaml
import argparse
import shutil
from glob import glob
import os
from absl import app, flags
from models.loss import get_loss, weighted_BCE_logits
from dataset.choose_dataset import get_dataloader
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from models.define_net import choose_net
from models.define_scheduler import get_scheduler
import warnings

warnings.filterwarnings('ignore')

FLAGS = flags.FLAGS
flags.DEFINE_string('config', None, 'training config file', required=True)
flags.DEFINE_string('gpu', '0', 'visible GPU ids', required=False)


def makedir(path):
    time_tag = datetime.datetime.now().strftime('%Y%m%d%H%M')
    if os.path.exists(path):
        path = f"{path}_{time_tag}"
    os.makedirs(path)
    return path


def main(argv):
    with open(FLAGS.config, encoding='utf-8') as file:
        args = yaml.safe_load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    working_path = os.path.dirname(os.path.abspath(__file__))
    args['trained_model_dir'] = os.path.join(working_path, 'exps',
                                             f"{args['NET_NAME']}_{args['DATA_NAME']}_"
                                             f"{args['lr']}_b{args['train_batch_size']}_e{args['epochs']}")

    args['trained_model_dir'] = makedir(args['trained_model_dir'])
    shutil.copy(FLAGS.config, args['trained_model_dir'])

    # dataloader
    train_loader, val_loader, num_classes = get_dataloader(args['DATA_NAME'],
                                                           args['train_batch_size'],
                                                           args['val_batch_size'])

    # network
    net = choose_net(args['NET_NAME'], num_classes)
    if isinstance(args['weight_decay'], str):
        args['weight_decay'] = eval(args['weight_decay'])

    # loss
    criterion, criterion_sc = get_loss(args['ss_loss'], args['cr_loss'])

    # optimizer & scheduler
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = get_scheduler(optimizer, args)

    bestaccT = 0
    bestmIoU = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    curr_epoch = 0

    def adjust_lr(optimizer, curr_iter, all_iter, init_lr):
        scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
        running_lr = init_lr * scale_running_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = running_lr

    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1

            if args['lr_strategy'] == 'default':
                adjust_lr(optimizer, running_iter, all_iters, init_lr=args['lr'])

            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

            optimizer.zero_grad()
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

            assert outputs_A.size()[1] == num_classes

            loss_seg = criterion(outputs_A, labels_A) * 0.5 + criterion(outputs_B, labels_B) * 0.5
            loss_bn = weighted_BCE_logits(out_change, labels_bn)
            loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
            loss = loss_seg + loss_bn + loss_sc
            loss.backward()
            optimizer.step()

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f  acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, acc_meter.val * 100))

        if args['lr_strategy'] != 'default' and args['lr_strategy'] != 'plateau':
            scheduler.step()

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(net, val_loader, criterion, args, num_classes)

        if args['lr_strategy'] == 'plateau':
            scheduler.step(loss_v)

        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if mIoU_v > bestmIoU:
            bestmIoU = mIoU_v
            bestaccV = acc_v
            bestloss = loss_v
            torch.save(net.state_dict(),
                       os.path.join(args['trained_model_dir'], '%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth'
                                    % (curr_epoch, mIoU_v * 100, Sek_v * 100, Fscd_v * 100, acc_v * 100)))

            # update best checkpoint file
            history_best = glob(f"{args['trained_model_dir']}/best*.pth")
            for file in history_best:
                os.remove(file)
            torch.save(net.state_dict(),
                       os.path.join(args['trained_model_dir'], 'best_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth' \
                                    % (mIoU_v * 100, Sek_v * 100, Fscd_v * 100, acc_v * 100)))

        print('Total time: %.1fs Best rec: Train acc %.2f, Val mIoU %.2f acc %.2f loss %.4f' % (
            time.time() - begin_time, bestaccT * 100, bestmIoU * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            print('Training finished.')
            return


def validate(net, val_loader, criterion, args, num_classes):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


if __name__ == '__main__':
    app.run(main)
