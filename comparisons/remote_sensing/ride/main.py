
import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil

import argparse
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils import LinearAverage
from utils import NCA_RI_Add_CrossEntropy
from model import ResNet18, ResNet50, BNInception, ResNet34
from modelSCCov import SCCov_Res34_emb, SCCov_Res50_emb
from metrics import KNNClassification, MetricTracker
from dataGen import DataGenSNCA_RI
from alive_progress import alive_bar
from layers.resnext import ResNext

parser = argparse.ArgumentParser(description='PyTorch SNCA Training for RS')
parser.add_argument('--data', metavar='DATA_DIR',  default='../data',
                        help='path to dataset (default: ../data)')
parser.add_argument('--dataset', metavar='DATASET',  default='ucmerced',
                        help='learning on the dataset (ucmerced)')
parser.add_argument('--dataset_RI', metavar='DATASET_RI',  default='ucmerced',
                        help='the rotation invariant version of dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch, (default:8)')
parser.add_argument('--epochs', default=130, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dim', default=128, type=int,
                    metavar='D', help='embedding dimension (default:128)')
parser.add_argument('--imgEXT', metavar='IMGEXT',  default='tif',
                        help='img extension of the dataset (default: tif)')
parser.add_argument('--temperature', default=0.05, type=float,
                    metavar='T', help='temperature parameter')
parser.add_argument('--memory-momentum', '--m-mementum', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--margin', default=0.0, type=float,
                    help='classification margin')
parser.add_argument('--model', metavar='MODEL',  default='resnet50',
                        help='CNN model (BNInception or resnet50, 34, SCCov_Res34, SCCov_Res50)')
parser.add_argument('--lambda_', default=0.1, type=float,
                    help='penalty para')

args = parser.parse_args()

sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
print('saving file name is ', sv_name)

checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
logs_dir = os.path.join('./', sv_name, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):
    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))


def main():
    global args, sv_name, logs_dir, checkpoint_dir

    write_arguments_to_file(args, os.path.join('./', sv_name, sv_name+'_arguments.txt'))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data_transform = transforms.Compose([
                                        transforms.Resize((256,256), antialias=True),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])

    val_data_transform = transforms.Compose([
                                            transforms.Resize((256,256), antialias=True),
                                            transforms.ToTensor(),
                                            normalize])


    train_dataGen = DataGenSNCA_RI(data=args.data, 
                                    dataset=args.dataset,
                                    dataset_RI=args.dataset_RI,
                                    # angles=[0,45,90,135,180,225,270,315],
                                    angles=[0,90,180,270],
                                    rate=0.7,
                                    imgTransform=train_data_transform,
                                    phase='train')

    train_dataGen_ = DataGenSNCA_RI(data=args.data, 
                                    dataset=args.dataset,
                                    dataset_RI=args.dataset_RI,
                                    # angles=[0,45,90,135,180,225,270,315],
                                    angles=[0,90,180,270],
                                    rate=0.7,
                                    imgTransform=val_data_transform,
                                    phase='train')

    val_dataGen = DataGenSNCA_RI(data=args.data, 
                                    dataset=args.dataset,
                                    dataset_RI=args.dataset_RI,
                                    # angles=[0,45,90,135,180,225,270,315],
                                    angles=[0,90,180,270],
                                    rate=0.7,
                                    imgTransform=val_data_transform,
                                    phase='val')


    train_data_loader = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    trainloader_wo_shuf = DataLoader(train_dataGen_, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # if args.model == 'resnet50':
    #     model = ResNet50(dim=args.dim)
    # elif args.model == 'resnet34':
    #     model = ResNet34(dim=args.dim)
    # elif args.model == 'SCCov_Res34':
    #     model = SCCov_Res34_emb(dim=args.dim)
    # elif args.model == 'SCCov_Res50':
    #     model = SCCov_Res50_emb(dim=args.dim)
    # else:
    #     model = BNInception(dim=args.dim)

    # if use_cuda:
    #     model.cuda()
    
    model = ResNext(att_type='se').cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4, nesterov=True)

    best_acc = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            # lemniscate = checkpoint['lemniscate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # define lemniscate and loss function (criterion)
        ndata = len(train_dataGen)
        lemniscate = LinearAverage(args.dim, ndata, args.temperature, args.memory_momentum).cuda()

    cls_y_true = []
    ins_y_true = []

    # for idx, data in enumerate(tqdm(trainloader_wo_shuf, desc="extracting training labels", ascii=True, ncols=20)):
    print('Extracting training labels...')
    with alive_bar(len(trainloader_wo_shuf)) as bar:
        for idx, data in enumerate(trainloader_wo_shuf):
            label_batch = data['label'].to(torch.device("cpu"))
            ins_label_batch = data['insLabel'].to(torch.device("cpu"))
            cls_y_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))
            ins_y_true += list(np.squeeze(ins_label_batch.numpy()).astype(np.float32))
            bar()
    cls_y_true = np.asarray(cls_y_true)
    ins_y_true = np.asarray(ins_y_true)

    criterion = NCA_RI_Add_CrossEntropy(torch.LongTensor(cls_y_true),
            torch.LongTensor(ins_y_true),
            args.lambda_,
            args.margin / args.temperature).cuda()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        adjust_memory_update_rate(lemniscate, epoch)

        # train for one epoch
        
        train(train_data_loader, model, lemniscate, criterion, optimizer, epoch, train_writer)

        # acc = val(val_data_loader, trainloader_wo_shuf, model, epoch, val_writer)

        # is_best_acc = acc > best_acc
        is_best_acc = True
        # best_acc = max(best_acc, acc)
        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            # 'lemniscate': lemniscate,
            'best_acc': 0.0,
            'optimizer' : optimizer.state_dict(),
        }, is_best_acc, sv_name)

        scheduler.step()

def train(trainloader, model, lemniscate, criterion, optimizer, epoch, train_writer):

    losses = MetricTracker()
    clslosses = MetricTracker()
    inslosses = MetricTracker()

    model.train()

    # for idx, data in enumerate(tqdm(trainloader, desc="training", ascii=True, ncols=20)):
    print('Training...')
    with alive_bar(len(trainloader)) as bar:
        for idx, data in enumerate(trainloader):
            imgs = data['img'].to(torch.device("cuda"))
            index = data["idx"].to(torch.device("cuda"))

            feature = model(imgs)
            output = lemniscate(feature, index)
            clsLoss, insLoss = criterion(output, index)
            loss = clsLoss + insLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), imgs.size(0))
            clslosses.update(clsLoss.item(), imgs.size(0))
            inslosses.update(insLoss.item(), imgs.size(0))
            bar()

    train_writer.add_scalar("loss", losses.avg, epoch)
    train_writer.add_scalar("clsloss", clslosses.avg, epoch)
    train_writer.add_scalar("insloss", inslosses.avg, epoch)

    print(f'Train loss: {losses.avg:.6f}, cls loss: {clslosses.avg:.6f}, ins loss: {inslosses.avg:.6f}')

def val(valloader, trainloader_wo_shuf, model, epoch, val_writer):

    model.eval()

    y_true = []
    train_features = []

    val_features = []
    y_val_true = []

    with torch.no_grad():
        # for batch_idx, data in enumerate(tqdm(trainloader_wo_shuf, desc="extracting training labels", ascii=True, ncols=20)):
        print('VAL - extracting training labels...')
        with alive_bar(len(trainloader_wo_shuf)) as bar:
            for idx, data in enumerate(trainloader_wo_shuf):
                imgs = data['img'].to(torch.device("cuda"))
                label_batch = data['label'].to(torch.device("cpu"))
                feature = model(imgs)
                train_features += list(feature.cpu().numpy().astype(np.float32))
                y_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))
                bar()

        # for batch_idx, data in enumerate(tqdm(valloader, desc="validation", ascii=True, ncols=20)):
        print('VAL - validation...')
        with alive_bar(len(valloader)) as bar:
            for idx, data in enumerate(valloader):
                imgs = data['img'].to(torch.device("cuda"))
                label_batch = data['label'].to(torch.device("cpu"))
                feature = model(imgs)
                val_features += list(feature.cpu().numpy().astype(np.float32))
                y_val_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))
                bar()

    y_true = np.asarray(y_true)
    train_features = np.asarray(train_features)

    y_val_true = np.asarray(y_val_true)
    val_features = np.asarray(val_features)

    knn_classifier = KNNClassification(train_features, y_true)

    acc = knn_classifier(val_features, y_val_true)

    val_writer.add_scalar('KNN-Acc', acc, epoch)

    print('Validation KNN-Acc: {:.6f} '.format(
            acc,
            # hammingBallRadiusPrec.val,
            ))

    return acc

def adjust_memory_update_rate(lemniscate, epoch):
    if epoch >= 80:
        lemniscate.params[1] = 0.8
    if epoch >= 120:
        lemniscate.params[1] = 0.9


if __name__ == "__main__":
    main()
