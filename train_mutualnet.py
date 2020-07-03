import importlib
import os
import time
import random

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import ComputePostBN
from utils.setlogger import get_logger

from utils.model_profiling import model_profiling
from utils.transforms import Lighting, InputList
from utils.config import FLAGS

# set log files
saved_path = os.path.join("logs", 'mutual_{}'.format(FLAGS.model[10:]))
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, '{}.log'.format('test' if FLAGS.test_only else 'train')))
def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    return model


def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(FLAGS.image_size, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            InputList(FLAGS.resolution_list),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(FLAGS.image_resize),
            transforms.CenterCrop(FLAGS.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset_dir))
    postset = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
    return train_set, val_set, test_set, postset


def data_loader(train_set, val_set, test_set, postset):
    """get data loader"""
    if FLAGS.data_loader == 'imagenet1k_basic':
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=FLAGS.batch_size, shuffle=True,
                pin_memory=True, num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        else:
            train_loader = None
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=FLAGS.batch_size, shuffle=False,
            pin_memory=True, num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    postloader = torch.utils.data.DataLoader(
                postset, batch_size=FLAGS.batch_size, shuffle=True,
                pin_memory=True, num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))

    return train_loader, val_loader, test_loader, postloader


def get_lr_scheduler(optimizer):
    """get learning rate"""
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            else:
                lr_dict[i] = lr_dict[i - 1] * FLAGS.exp_decaying_lr_gamma
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - i / FLAGS.num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, FLAGS.num_epochs)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    FLAGS.lr_scheduler))
    return lr_scheduler


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # weight decay only on normal conv and fc
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:  # normal conv
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:  # fc
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


def set_random_seed():
    """set random seed"""
    if hasattr(FLAGS, 'random_seed'):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
        model.apply(
            lambda m: setattr(m, 'width_mult', width_mult))
        print('Model profiling with width mult {}x:'.format(width_mult))
        verbose = width_mult == max(FLAGS.width_mult_list)
        model_profiling(
            model, FLAGS.image_size, FLAGS.image_size,
            verbose=getattr(FLAGS, 'model_profiling_verbose', verbose))


def train(epoch, loader, model, criterion, optimizer):
    t_start = time.time()
    model.train()
    for batch_idx, (input_list, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        # do max width
        max_width = FLAGS.width_mult_range[1]
        model.apply(lambda m: setattr(m, 'width_mult', max_width))
        max_output = model(input_list[0])
        loss = torch.mean(criterion(max_output, target))
        loss.backward()
        max_output_detach = max_output.detach()
        # do other widths and resolution
        min_width = FLAGS.width_mult_range[0]
        width_mult_list = [min_width]
        sampled_width = list(np.random.uniform(FLAGS.width_mult_range[0], FLAGS.width_mult_range[1], 2))
        width_mult_list.extend(sampled_width)
        for width_mult in sorted(width_mult_list, reverse=True):
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            output = model(input_list[random.randint(0, 3)])
            loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))
            loss.backward()
        optimizer.step()
        # print training log
        if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader)-1:
            with torch.no_grad():
                for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                    model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    output = model(input_list[0])
                    loss = torch.mean(criterion(output, target)).cpu().numpy()
                    indices = torch.max(output, dim=1)[1]
                    acc = (indices == target).sum().cpu().numpy() / indices.size()[0]
                    logger.info('TRAIN {:.1f}s LR:{:.4f} {} Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                        time.time() - t_start, optimizer.param_groups[0]['lr'], str(width_mult), epoch,
                        FLAGS.num_epochs, batch_idx, len(loader), loss, acc))


def validate(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    resolution = FLAGS.image_size
    with torch.no_grad():
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            model = ComputePostBN.ComputeBN(model, postloader, resolution)
            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output = model(input)
                loss += torch.sum(criterion(output, target)).cpu().numpy()
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
                if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader)-1:
                    logger.info('VAL {:.1f}s {} Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                        time.time() - t_start, str(width_mult), epoch,
                        FLAGS.num_epochs, batch_idx, len(loader), loss/cnt, acc/cnt))

def test(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        for resolution in FLAGS.resolution_list:
            for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                model = ComputePostBN.ComputeBN(model, postloader, resolution)
                loss, acc, cnt = 0, 0, 0
                for batch_idx, (input, target) in enumerate(loader):
                    input, target =input.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    output = model(F.interpolate(input, (resolution, resolution), mode='bilinear', align_corners=True))
                    loss += torch.sum(criterion(output, target)).cpu().numpy()
                    indices = torch.max(output, dim=1)[1]
                    acc += (indices==target).sum().cpu().numpy()
                    cnt += target.size()[0]
                    if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader)-1:
                        logger.info('VAL {:.1f}s {}-{} Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                            time.time() - t_start, str(width_mult), str(resolution), epoch,
                            FLAGS.num_epochs, batch_idx, len(loader), loss/cnt, acc/cnt))

def train_val_test():
    """train and val"""
    # seed
    set_random_seed()

    # model
    model = get_model()
    model_wrapper = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        new_keys = [key for key in new_keys if 'bn' not in key]
        old_keys = [key for key in old_keys if 'bn' not in key]
        if not FLAGS.test_only:
            old_keys = old_keys[:-2]
            new_keys = new_keys[:-2]

        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = checkpoint[key_old]
        model_wrapper.load_state_dict(new_checkpoint, strict=False)
        print('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = get_optimizer(model_wrapper)
    # check resume training
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = get_lr_scheduler(optimizer)
        lr_scheduler.last_epoch = last_epoch
        print('Loaded checkpoint {} at epoch {}.'.format(
            FLAGS.resume, last_epoch))
    else:
        lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = lr_scheduler.last_epoch
        # print model and do profiling
        print(model_wrapper)
        if FLAGS.profiling:
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set, post_set = dataset(
        train_transforms, val_transforms, test_transforms)
    train_loader, val_loader, test_loader, post_loader = data_loader(
        train_set, val_set, test_set, post_set)

    if FLAGS.test_only and (test_loader is not None):
        logger.info('Start testing.')
        test(last_epoch, val_loader, model_wrapper, criterion, post_loader)
        return

    logger.info('Start training.')
    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        lr_scheduler.step()
        # train
        train(epoch, train_loader, model_wrapper, criterion, optimizer)

        # val
        validate(epoch, val_loader, model_wrapper, criterion, post_loader)
        torch.save(
            {
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
            },
            os.path.join(saved_path, 'checkpoint_{}.pt'.format(epoch)))
    return


def main():
    """train and eval model"""
    train_val_test()


if __name__ == "__main__":
    main()
