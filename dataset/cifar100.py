from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch
import datetime
import numpy as np
from torch import nn



"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class MIXUPNet(nn.Module):
    def __init__(self, mix_len):
        super(MIXUPNet, self).__init__()
        self.mix_len = mix_len
    
    def forward(self, remain_data,lost_data, remain, lost, remain_en, lost_en, idx):
        with torch.no_grad():
            # sum_en = np.exp(remain_en) + np.exp(lost_en)
            # remain_rate = np.exp(remain_en) / sum_en
            # lost_rate = np.exp(lost_en) / sum_en
            remain_rate = (1 - 0.7) * ((self.mix_len * torch.ones(len(idx)) - idx) / self.mix_len) + 0.7
            lost_rate = torch.ones(len(idx)) - remain_rate
            print('remain_rate')
            print(remain_rate)
            print('lost_rate')
            print(lost_rate)
            print('remain_en')
            print(remain_en)
            print('lost_en')
            print(lost_en)
            minup = remain_data[0].unsqueeze(0) * remain_rate[0] + lost_data[0].unsqueeze(0) * lost_rate[0]
            for i in range(1,remain.shape[0]):
                minup = torch.cat((minup, remain_data[i].unsqueeze(0) * remain_rate[i] + lost_data[i].unsqueeze(0) * lost_rate[i]))

        return torch.tensor(minup.clone().detach(),dtype = torch.uint8).cpu()
 

class MIXUPDataset(Dataset):
    def __init__(self, remain_data, lost_data, remain, lost, remain_en, lost_en):
        self.remain = remain
        self.lost = lost

        self.remain_data = remain_data
        self.lost_data = lost_data

        self.remain_en = remain_en
        self.lost_en = lost_en

    def __len__(self):
        return len(self.remain_data)

    def __getitem__(self, idx):
        return self.remain_data[idx],self.lost_data[idx], self.remain[idx], self.lost[idx], self.remain_en[idx], self.lost_en[idx], idx


class SuperCIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __init__(self, root, train = True, download=False, transform = None, target_transform = None,
    threshold = 0.5,loaddir = None ,savedir = None, mix_up_data_p = None):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        if loaddir == None:
            self.remain = np.array([i for i in range(len(self.data))])
        else:
            self.remain = np.loadtxt(loaddir).astype(int)
            print(len(self.remain))
            print('success load ' + loaddir)

        self.threshold = threshold
        # self.method = method
        self.savedir = savedir

        self.entropy_record = None

        if mix_up_data_p is None:
            self.mix_up_data = torch.tensor(self.data, requires_grad=False)
        else:
            self.mix_up_data = torch.load(mix_up_data_p)
            print('success load pt' + mix_up_data_p)

        self.mix_remain = None
        self.lost = None

    def __getitem__(self, index):
        if self.train:
            img, target = self.mix_up_data[self.remain[index]].numpy(), self.targets[self.remain[index]]
        else:
            img, target = self.test_data[self.remain[index]], self.test_labels[self.remain[index]]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.remain[index]

    
    def __len__(self):
        return len(self.remain)
    

    def init_len(self):
        return len(self.data)


    def save(self,savedir):
        if self.savedir != None:
            np.savetxt(self.savedir , self.remain)
        else:
            now = datetime.datetime.now()
            np.savetxt('./myRemain_'+ savedir + now.strftime("%Y-%m-%d %H:%M:%S") +'.txt', self.remain)
        print('remain saved')


    def clear(self):
        self.entropy_record = None


    def update_statistics(self, entropy):
        if self.entropy_record is None:
            self.entropy_record = entropy.reshape((len(entropy),1))
        else:
            self.entropy_record = np.concatenate((self.entropy_record, entropy.reshape((len(entropy),1))),axis=1)


    def update_remain(self, remain):
        self.remain = remain

    def save_mix_up(self, epoch, model_name):
        torch.save(self.mix_up_data, './mix_up_data_'+model_name+str(epoch)+'.pt')
        np.savetxt('./epoch_mix_remain_'+model_name+str(epoch)+'.txt', self.mix_remain)
        np.savetxt('./epoch_remain_'+model_name+str(epoch)+'.txt', self.remain)

    @torch.no_grad()
    def mixup(self, _entropy):
        # lost = np.setdiff1d(np.array([i for i in range(self.init_len())]),self.remain)
        dataset = MIXUPDataset(self.mix_up_data[self.mix_remain.copy()], self.mix_up_data[self.lost.copy()], self.mix_remain, self.lost, _entropy[self.mix_remain.copy()]
        , _entropy[self.lost.copy()])

        # dataset = MIXUPDataset(self.mix_up_data[self.mix_remain.copy()],
        #  torch.flip(self.mix_up_data[self.lost.copy()],dims=[0]), 
        #  self.mix_remain, 
        #  torch.flip(torch.from_numpy(self.lost.copy()),dims=[0]), 
        #  _entropy[self.mix_remain.copy()]
        # , torch.flip(torch.from_numpy(_entropy[self.lost.copy()]),dims=[0]))

        mix_loader = DataLoader(dataset,
                              batch_size=64,
                              shuffle=False,
                              num_workers=1)

        mixnet = MIXUPNet(len(self.mix_remain))
        
        for data in mix_loader:
            remain_data, lost_data, remain, lost, remain_en, lost_en, idx = data
            remain_data = remain_data #.cuda()
            lost_data = lost_data #.cuda()
            mixup = mixnet( remain_data, lost_data, remain, lost, remain_en, lost_en, idx)
            self.mix_up_data[remain] = mixup
            
            
    def update_dataset_mul_entropy_V2(self):
        if len(self.remain) - int(len(self.data) * (1 - self.threshold) / 6) < int(self.threshold * len(self.data)):
            remain_num = int(self.threshold * len(self.data))
        else:
            remain_num = len(self.remain) - int(len(self.data) * (1 - self.threshold) / 6)

        print(int(len(self.data) * (1 - self.threshold) / 6))

        _entropy = np.clip(self.entropy_record,0.0,a_max = self.entropy_record.max())
        _entropy = np.sum(_entropy, axis=1)
        _sum = np.where(self.entropy_record > 0.0, 1.0, 0.0)
        _sum = np.sum(_sum, axis=1)
        ET = np.true_divide(_sum, 40.0)
        ET = np.power(ET, 0.03)
        _entropy = np.true_divide(_entropy, _sum)
        scores = ET * _entropy
        indice = scores.argsort()[::-1]
        pre_len = len(self.remain)
        self.remain = indice[:remain_num]
    
    def update_dataset_mul_entropy_V3(self):
        if len(self.remain) - int(len(self.data) * (1 - self.threshold) / 6) < int(self.threshold * len(self.data)):
            remain_num = int(self.threshold * len(self.data))
        else:
            remain_num = len(self.remain) - int(len(self.data) * (1 - self.threshold) / 6)

        print(int(len(self.data) * (1 - self.threshold) / 6))

        _entropy = np.clip(self.entropy_record,0.0,a_max = self.entropy_record.max())
        _entropy = np.sum(_entropy, axis=1)
        _sum = np.where(self.entropy_record > 0.0, 1.0, 0.0)
        _sum = np.sum(_sum, axis=1)
        ET = np.true_divide(_sum, 40.0)
        ET = np.power(ET, 0.03)
        _entropy = np.true_divide(_entropy, _sum)
        scores = ET * _entropy
        indice = scores.argsort()[::-1]
        pre_len = len(self.remain)
        self.remain = indice[:remain_num]
        self.lost = indice[remain_num:pre_len]

        # tail
        self.mix_remain = self.remain[-len(self.lost):]

        # head
        # self.mix_remain = self.remain[:len(self.lost)]

        # random
        # self.mix_remain = np.random.choice(self.remain, len(self.lost))

        self.mixup(_entropy)
        


def super_get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False, init = False , threshold = 0.5,loaddir = None, savedir = None, mix_up_data_p = None):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        if not init:
            train_set = SuperCIFAR100Instance(root=data_folder,
                                         download=True,
                                         train=True,
                                         transform=train_transform,
                                         threshold = threshold,
                                         loaddir = loaddir,
                                         savedir = savedir,
                                         mix_up_data_p = mix_up_data_p)
        else:
            train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data, train_set
    else:
        return train_loader, test_loader, train_set

class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            num_samples = len(self.train_data)
            label = self.train_labels
        else:
            num_samples = len(self.test_data)
            label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
