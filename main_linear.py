from __future__ import print_function

import sys
import argparse
import time
import math
import copy
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, _accuracy
from util import set_optimizer
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, datasets

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCon_Linear_Age_{}_{}_lr_{}_decay_{}_bsz_{}_size_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.size)

    # opt.model_name = 'SupCon_Linear_{}_{}_lr_{}_decay_{}_bsz_{}'.\
    # opt.model_name = 'SimCLR_Linear_{}_{}_lr_{}_decay_{}_bsz_{}'.\
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = _accuracy(output, labels, topk=(1, ))
        top1.update(acc1[0], bsz)

        # print('top1', top1.val.item())
        # print('top1', top1.avg.item())
        # print('loss', losses.val)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1_val=top1.val.item(), top1_avg=top1.avg.item()))
            sys.stdout.flush()

    return losses.avg, top1.avg.item()


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = _accuracy(output, labels, topk=(1, ))
            top1.update(acc1[0], bsz)
            all_outputs.extend(output.detach().cpu().numpy())  # Convert tensor to numpy and append
            all_labels.extend(labels.detach().cpu().numpy())  # Convert tensor to numpy and append
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1_val=top1.val.item(), top1_avg=top1.avg.item()))

    print(' * Acc@1 {top1_avg:.3f}'.format(top1_avg=top1.avg.item()))
    predicted_classes = np.argmax(all_outputs, axis=1)  # Assuming outputs are logits or probabilities
    accuracy = accuracy_score(all_labels, predicted_classes)
    precision = precision_score(all_labels, predicted_classes)
    recall = recall_score(all_labels, predicted_classes)
    f1 = f1_score(all_labels, predicted_classes)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    return losses.avg, top1.avg.item()

def dataset_loader(opt, fold, is_test):
    mean = eval(opt.mean)
    std = eval(opt.std)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize((opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = f'{opt.data_folder}train_fold_{fold}'
    if is_test:
        val_folder = f'{opt.data_folder}test'
    else:
        val_folder = f'{opt.data_folder}val_fold_{fold}'
    train_dataset = datasets.ImageFolder(root=train_folder, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_folder, transform=val_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def main():
    best_acc = 0
    best_model = None
    best_classifier = None
    opt = parse_option()

    for fold in range(1, 6):
        # build data loader
        train_loader, val_loader = dataset_loader(opt, fold, False)

        # build model and criterion
        model, classifier, criterion = set_model(opt)

        # build optimizer
        optimizer = set_optimizer(opt, classifier)

        # opt.epochs = 1
        # training routine
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, criterion,
                            optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

            # eval for one epoch
            loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model)
                best_classifier = copy.deepcopy(classifier)
    
    print('best val accuracy: {:.2f}'.format(best_acc))
    print("start testing dataset!!!")

    _, test_loader =  dataset_loader(opt, 1, True)
    loss, val_acc = validate(test_loader, best_model, best_classifier, criterion, opt)
    print('testing accuracy: {:.4f}'.format(val_acc))
    # save the last model
    save_file_model = os.path.join(
        opt.save_folder, 'best_model.pth')
    state_model = {
        'model': best_model.state_dict(),
    }
    torch.save(state_model, save_file_model)

    save_file_classifier = os.path.join(
        opt.save_folder, 'best_classifier.pth')
    state_classifier = {
        'model': best_classifier.state_dict(),
    }
    torch.save(state_classifier, save_file_classifier)


if __name__ == '__main__':
    main()
