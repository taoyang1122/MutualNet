import os
import torch
import torch.nn.functional as F
from utils.config import FLAGS
from torchvision import datasets, transforms
from utils.transforms import Lighting, InputList

def get_imagenet():
    if FLAGS.data_transforms == 'imagenet1k_basic':
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
    train_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'train'), transform=train_transforms)
    val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'), transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=FLAGS.batch_size, shuffle=True,
        pin_memory=True, num_workers=FLAGS.data_loader_workers,
        drop_last=getattr(FLAGS, 'drop_last', False))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=FLAGS.batch_size, shuffle=False,
        pin_memory=True, num_workers=FLAGS.data_loader_workers,
        drop_last=getattr(FLAGS, 'drop_last', False))

    return train_loader, val_loader

def get_cifar():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        InputList(FLAGS.resolution_list)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert (FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[FLAGS.dataset.upper()]('../data', train=True, download=True,
                                                transform=train_transform),
        batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[FLAGS.dataset.upper()]('../data', train=False, transform=test_transform),
        batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader

def get_dataset():
    if FLAGS.dataset == 'imagenet1k':
        return get_imagenet()
    elif 'cifar' in FLAGS.dataset:
        return get_cifar()
    else:
        raise NotImplementedError('dataset not implemented.')