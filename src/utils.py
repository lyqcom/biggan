#  
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=C0330
# pylint: disable=C0325
# pylint: disable=W0702
# pylint: disable=W0703
# pylint: disable=W0612
# pylint: disable=W0105
# pylint: disable=R1705
# pylint: disable=R1710
# pylint: disable=C1801
# pylint: disable=W0212

from __future__ import print_function

import sys
import os
import random
import math
import time
import datetime
import json
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.nn.optim import Optimizer
from src import datasets as dset
try:
    import moxing as mox
except:
    mox = False


def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)
    ### Dataset/Dataloader stuff ###
    parser.add_argument('--dataset', type=str, default='I128_hdf5',
                        help=';''Append "_hdf5" to use the hdf5 version for ISLVRC ''(default: %(default)s)')
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--no_pin_memory', action='store_false',
                        dest='pin_memory', default=True)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--load_in_mem', action='store_true', default=False)
    parser.add_argument('--use_multiepoch_sampler',
                        action='store_true', default=False)
    parser.add_argument('--datasets_path', default='',
                        help='training imagenet dataset path')
    parser.add_argument('--ckpt_pth', default='',
                        help='trans to mindir .ckpt path')
    parser.add_argument('--use_device', default=0, help="set trained device")
    parser.add_argument('--distributed', default=False,
                        help="set distributed model for training")
    parser.add_argument('--input_dim', type=int, default=100)
    parser.add_argument('--output_dim', type=int, default=3)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='BigGAN',
                        help='Name of the model module (default: %(default)s)')
    parser.add_argument('--G_param', type=str, default='SN')
    parser.add_argument('--D_param', type=str, default='SN')
    parser.add_argument('--G_ch', type=int, default=64,
                        help='Channel multiplier for G (default: %(default)s)')
    parser.add_argument('--D_ch', type=int, default=64,
                        help='Channel multiplier for D (default: %(default)s)')
    parser.add_argument('--G_depth', type=int, default=1)
    parser.add_argument('--D_depth', type=int, default=1)
    parser.add_argument('--dim_z', type=int, default=128,
                        help='Noise dimensionality: %(default)s)')
    parser.add_argument('--z_var', type=float, default=1.0,
                        help='Noise variance: %(default)s)')
    parser.add_argument('--cross_replica', action='store_true', default=False)
    parser.add_argument('--mybn', action='store_true', default=False)
    parser.add_argument('--G_nl', type=str, default='relu',
                        help='Activation function for G (default: %(default)s)')
    parser.add_argument('--D_nl', type=str, default='relu',
                        help='Activation function for D (default: %(default)s)')
    parser.add_argument('--norm_style', type=str, default='bn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip_init', action='store_true', default=False)
    parser.add_argument('--G_lr', type=float, default=5e-5)
    parser.add_argument('--D_lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument('--G_batch_size', type=int, default=0)
    parser.add_argument('--num_G_accumulations', type=int, default=1)
    parser.add_argument('--num_D_steps', type=int, default=2)
    parser.add_argument('--num_D_accumulations', type=int, default=1)
    parser.add_argument('--split_D', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--G_fp16', action='store_true', default=False)
    parser.add_argument('--D_fp16', action='store_true', default=False)
    parser.add_argument('--accumulate_stats')
    parser.add_argument('--num_standing_accumulations', type=int, default=16)
    parser.add_argument('--G_eval_mode', action='store_true', default=False)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--num_save_copies', type=int, default=2)
    parser.add_argument('--num_best_copies', type=int, default=2)
    parser.add_argument('--no_fid', action='store_true', default=False)
    parser.add_argument('--num_inception_images', type=int, default=50000)
    parser.add_argument('--hashname', action='store_true', default=False)
    parser.add_argument('--base_root', type=str, default='')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--weights_root', type=str, default='weights')
    parser.add_argument('--logs_root', type=str, default='logs')
    parser.add_argument('--samples_root', type=str, default='samples')
    parser.add_argument('--name_suffix', type=str, default='')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--config_from_name',
                        action='store_true', default=False)
    parser.add_argument('--ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate (default: %(default)s)')
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_start', type=int, default=0)
    parser.add_argument('--which_train_fn', type=str, default='GAN',
                        help='How2trainyourbois (default: %(default)s)')
    parser.add_argument('--load_weights', type=str, default='')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training? (default: %(default)s)')
    parser.add_argument('--logstyle', type=str, default='%3.3e')
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--data_thr', default=116500, type=int)
    parser.add_argument('--data_url', type=str,
                        help='set imagent dataset obs url')
    parser.add_argument('--train_url', type=str,
                        help='restore trained weights url')
    parser.add_argument('--imagehdf5_data_url', type=str,
                        help='set imagenet hdf5 dataset obs url')
    parser.add_argument('--project_url', type=str,
                        help='set imagenet hdf5 dataset obs url')
    return parser


def add_sample_parser(parser):
    parser.add_argument(
        '--sample_npz', action='store_true', default=False,
        help='Sample "sample_num_npz" images and save to npz? '
        '(default: %(default)s)')
    parser.add_argument(
        '--sample_num_npz', type=int, default=50000,
        help='Number of images to sample when sampling NPZs '
        '(default: %(default)s)')
    parser.add_argument(
        '--sample_sheets', action='store_true', default=False,
        help='Produce class-conditional sample sheets and stick them in '
        'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_interps', action='store_true', default=False,
        help='Produce interpolation sheets and stick them in '
        'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_sheet_folder_num', type=int, default=-1,
        help='Number to use for the folder for these sample sheets '
        '(default: %(default)s)')
    parser.add_argument(
        '--sample_random', action='store_true', default=False,
        help='Produce a single random sheet? (default: %(default)s)')
    parser.add_argument(
        '--sample_trunc_curves', type=str, default='',
        help='Get inception metrics with a range of variances?'
        'To use this, specify a startpoint, step, and endpoint, e.g. '
        '--sample_trunc_curves 0.2_0.1_1.0 for a startpoint of 0.2, '
        'endpoint of 1.0, and stepsize of 1.0.  Note that this is '
        'not exactly identical to using tf.truncated_normal, but should '
        'have approximately the same effect. (default: %(default)s)')
    parser.add_argument(
        '--sample_inception_metrics', action='store_true', default=False,
        help='Calculate Inception metrics with sample.py? (default: %(default)s)')

    return parser


# Convenience dicts
dset_dict = {'I32': dset.ImageFolder, 'I64': dset.ImageFolder,
             'I128': dset.ImageFolder, 'I256': dset.ImageFolder,
             'I32_hdf5': dset.ILSVRC_HDF5, 'I64_hdf5': dset.ILSVRC_HDF5,
             'I128_hdf5': dset.ILSVRC_HDF5, 'I256_hdf5': dset.ILSVRC_HDF5,
             'C10': dset.CIFAR10, 'C100': dset.CIFAR100}
imsize_dict = {'I32': 32, 'I32_hdf5': 32,
               'I64': 64, 'I64_hdf5': 64,
               'I128': 128, 'I128_hdf5': 128,
               'I256': 256, 'I256_hdf5': 256,
               'C10': 32, 'C100': 32}
root_dict = {'I32': 'ImageNet', 'I32_hdf5': 'ILSVRC32.hdf5',
             'I64': 'ImageNet', 'I64_hdf5': 'ILSVRC64.hdf5',
             'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
             'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
             'C10': 'cifar', 'C100': 'cifar'}
nclass_dict = {'I32': 1000, 'I32_hdf5': 1000,
               'I64': 1000, 'I64_hdf5': 1000,
               'I128': 1000, 'I128_hdf5': 1000,
               'I256': 1000, 'I256_hdf5': 1000,
               'C10': 10, 'C100': 100}
# Number of classes to put per sample sheet
classes_per_sheet_dict = {'I32': 50, 'I32_hdf5': 50,
                          'I64': 50, 'I64_hdf5': 50,
                          'I128': 20, 'I128_hdf5': 20,
                          'I256': 20, 'I256_hdf5': 20,
                          'C10': 10, 'C100': 100}
activation_dict = {'inplace_relu': mindspore.nn.ReLU(),
                   'relu': mindspore.nn.ReLU(),
                   'ir': mindspore.nn.ReLU()}


class CenterCropLongEdge():
    def __call__(self, img):
        print("get img.size", img.size)
        # return transforms.functional.center_crop(img, min(img.size))
        return mindspore.dataset.vision.py_transforms.CenterCrop(min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class RandomCropLongEdge():
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
             else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
             else np.random.randint(low=0, high=img.size[1] - size[1]))
        # return transforms.functional.crop(img, i, j, size[0], size[1])
        return mindspore.dataset.vision.c_transforms.Crop((i, j), (size[0], size[1]))

    def __repr__(self):
        return self.__class__.__name__


# multi-epoch Dataset sampler to avoid memory leakage and enable resumption of
# training from the same sample regardless of if we stop mid-epoch
class MultiEpochSampler():
    def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.num_epochs = num_epochs
        self.start_itr = start_itr
        self.batch_size = batch_size

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integral "
                             "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self):
        n = len(self.data_source)
        num_epochs = int(np.ceil((n * self.num_epochs
                                  - (self.start_itr * self.batch_size)) / float(n)))
        randperm_ = mindspore.ops.Randperm()
        out = [randperm_(n) for epoch in range(self.num_epochs)][-num_epochs:]
        # Ignore the first start_itr % n indices of the first epoch
        out[0] = out[0][(self.start_itr * self.batch_size % n):]
        cat_ = mindspore.ops.Concat()
        output = cat_(out).tolist()
        return iter(output)

    def __len__(self):
        return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size


# Convenience function to centralize all data loaders
def get_data_loaders(dataset, data_root=None, augment=True, batch_size=64,
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     **kwargs):
    args = prepare_parser()
    args = args.parse_args()
    data_root += '/%s' % root_dict[dataset]

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    image_size = imsize_dict[dataset]

    try:
        obs_imagenet_data_url = args.data_url  # for imagenet and imagenet hdf5
        obs_imagehdf5_data_url = args.imagehdf5_data_url
        # /cache/user-job-dir/workspace/device0  # /home/work/user-job-dir/code
        print("get current path:", os.getcwd())
        dataset_path = os.path.join(os.getcwd(), 'data/')
        print("get dataset_path", dataset_path)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        args.data_url = dataset_path  # retrans
        args.imagehdf5_data_url = dataset_path
        args.project_url = '/home/work/user-job-dir/code/'
        mox.file.copy_parallel(obs_imagenet_data_url, args.data_url)
        mox.file.copy_parallel(
            "obs://open-data/attachment/a/b/ab0c7c06-969e-425f-b35c-162965693158/", args.data_url)
        print("Successfully Download obs_imagenet_data {} to {}".format(
            obs_imagenet_data_url, args.data_url))
        print('Successfully Download obs_imagehdf5_data {} to {}'.format(
            obs_imagehdf5_data_url, args.imagehdf5_data_url))
    except Exception as e:
        print(str(e))

    train_transform = mindspore.dataset.transforms.py_transforms.Compose([
        mindspore.dataset.vision.py_transforms.Decode(),
        mindspore.dataset.vision.py_transforms.ToPIL(),
        mindspore.dataset.vision.py_transforms.RandomHorizontalFlip(),
        mindspore.dataset.vision.py_transforms.Resize((256, 256)),
        mindspore.dataset.vision.py_transforms.ToTensor(),
        mindspore.dataset.vision.py_transforms.Normalize(norm_mean, norm_std)])
    if mox:
        train_set = mindspore.dataset.ImageFolderDataset(dataset_dir=os.path.join(
            args.data_url, 'imagenet/train'), num_parallel_workers=8, decode=False, shuffle=True)
    else:
        print("get real dataset path:", os.path.join(
            args.datasets_path, 'train'))
        train_set = mindspore.dataset.ImageFolderDataset(dataset_dir=os.path.join(
            args.datasets_path, 'train'), num_parallel_workers=8, decode=False, shuffle=True)

    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(
            train_set, num_epochs, start_itr, batch_size)
        train_loader = train_set.map(operations=train_transform, input_columns=[
                                     "image"], num_parallel_workers=8)
    else:
        #print("not use_multiepoch_sampler")
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        train_transformz = [
            mindspore.dataset.vision.py_transforms.Decode(),
            mindspore.dataset.vision.py_transforms.ToPIL(),
            # RandomCropLongEdge(),
            mindspore.dataset.vision.py_transforms.RandomHorizontalFlip(),
            mindspore.dataset.vision.py_transforms.Resize((128, 128)),
            mindspore.dataset.vision.py_transforms.ToTensor(),
            mindspore.dataset.vision.py_transforms.Normalize(norm_mean, norm_std)]  # no cap
        train_loader = train_set.map(operations=train_transformz, input_columns=[
                                     "image"], num_parallel_workers=8)
    return train_loader


# Utility file to seed rngs
def seed_rng(seed):
    mindspore.set_seed(seed)
    np.random.seed(seed)


def update_config_roots(config):
    if config['base_root']:
        print('Pegging all root folders to base root %s' % config['base_root'])
        for key in ['data', 'weights', 'logs', 'samples']:
            config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
    return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
    for key in ['weights_root', 'logs_root', 'samples_root']:
        if not os.path.exists(config[key]):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key])


class ema():
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        for key in self.source_dict:
            self.target_dict[key].data.copy_(self.source_dict[key].data)
            # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay

        for key in self.source_dict:
            self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                             + self.source_dict[key].data * (1 - decay))


def toggle_grad(model, on_or_off):
    for param in model.get_parameters():
        param.requires_grad = on_or_off


def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


# Save a model's weights, optimizer, and the state_dict
def save_weights(G, D, epoch, weights_root='./weights', name_suffix=None, G_ema=None):
    args = prepare_parser()
    args = args.parse_args()
    root = weights_root
    if not os.path.exists(root):
        os.mkdir(root)
    if name_suffix:
        print('Saving weights to %s/%s...' % (root, name_suffix))
    else:
        print('Saving weights to %s...' % root)

    mindspore.save_checkpoint(G, '%s/%s.ckpt' %
                              (root, join_strings('_', ['G', str(epoch)])))
    if G_ema is not None:
        mindspore.save_checkpoint(G_ema, '%s/%s.ckpt' %
                                  (root, join_strings('_', ['G_ema', str(epoch)])))
        mindspore.save_checkpoint(D, '%s/%s.ckpt' %
                                  (root, join_strings('_', ['D', str(epoch)])))
    if mox:
        mox.file.copy_parallel("./weights/", args.train_url)


# Load a model's weights, optimizer, and the state_dict
def load_weights(G, D, state_dict, weights_root, experiment_name,
                 name_suffix=None, G_ema=None, strict=True, load_optim=True):
    root = '/'.join([weights_root, experiment_name])
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    if G is not None:
        G.load_state_dict(
            mindspore.load_checkpoint(
                '%s/%s.pth' % (root, join_strings('_', ['G', name_suffix]))),
            strict=strict)
        if load_optim:
            G.optim.load_state_dict(
                mindspore.load_checkpoint('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
    if D is not None:
        D.load_state_dict(
            mindspore.load_checkpoint(
                '%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
            strict=strict)
        if load_optim:
            D.optim.load_state_dict(
                mindspore.load_checkpoint('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
    # Load state dict
    for item in state_dict:
        state_dict[item] = mindspore.load_checkpoint(
            '%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))[item]
    if G_ema is not None:
        G_ema.load_state_dict(
            mindspore.load_checkpoint(
                '%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))),
            strict=strict)


class MetricsLogger():
    def __init__(self, fname, reinitialize=False):
        self.fname = fname
        self.reinitialize = reinitialize
        if os.path.exists(self.fname):
            if self.reinitialize:
                print('{} exists, deleting...'.format(self.fname))
                os.remove(self.fname)

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')


class MyLogger():
    def __init__(self, fname, reinitialize=False, logstyle='%3.3f'):
        self.root = fname
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.reinitialize = reinitialize
        self.metrics = []
        self.logstyle = logstyle  # One of '%3.3f' or like '%3.3e'

    # Delete log if re-starting and log already exists
    def reinit(self, item):
        if os.path.exists('%s/%s.log' % (self.root, item)):
            if self.reinitialize:
                # Only print the removal mess
                if 'sv' in item:
                    if not any('sv' in item for item in self.metrics):
                        print('Deleting singular value logs...')
                else:
                    print('{} exists, deleting...'.format(
                        '%s_%s.log' % (self.root, item)))
                os.remove('%s/%s.log' % (self.root, item))

    # Log in plaintext; this is designed for being read in MATLAB(sorry not sorry)
    def log(self, itr, **kwargs):
        for arg in kwargs:
            if arg not in self.metrics:
                if self.reinitialize:
                    self.reinit(arg)
                self.metrics += [arg]
            if self.logstyle == 'pickle':
                print('Pickle not currently supported...')
            elif self.logstyle == 'mat':
                print('.mat logstyle not currently supported...')
            else:
                with open('%s/%s.log' % (self.root, arg), 'a') as f:
                    str_ = str(itr) + ":"+str(self.logstyle)+str(kwargs[arg])
                    f.write(str_)
                    f.write("\n")


# Write some metadata to the logs directory
def write_metadata(logs_root, experiment_name, config, state_dict):
    with open(('%s/%s/metalog.txt' %
               (logs_root, experiment_name)), 'w') as writefile:
        writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
        writefile.write('config: %s\n' % str(config))
        writefile.write('state: %s\n' % str(state_dict))


def progress(items, desc='', total=None, min_delay=0.1, displaytype='s1k'):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or items.get_dataset_size
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            if n > 0:

                if displaytype == 's1k':  # minutes/seconds for 1000 iters
                    next_1000 = n + (1000 - n % 1000)
                    t_done = t_now - t_start
                    t_1k = t_done / n * next_1000
                    outlist = list(divmod(t_done, 60)) + \
                        list(divmod(t_1k - t_done, 60))
                    print("(TE/ET1k: %d:%02d / %d:%02d)" %
                          tuple(outlist), end=" ")
                else:  # displaytype == 'eta':
                    t_done = t_now - t_start
                    t_total = t_done / n * total
                    outlist = list(divmod(t_done, 60)) + \
                        list(divmod(t_total - t_done, 60))
                    print("(TE/ETA: %d:%02d / %d:%02d)" %
                          tuple(outlist), end=" ")

            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))


# Sample function for use with inception metrics
def sample(G, z_, y_, config):

    z_.sample_()
    y_ = y_.sample_()
    if config['parallel']:
        G_z = nn.parallel.data_parallel(G, (z_, G.shared(y_)))
    else:
        G_z = G(z_, G.shared(y_))
    return G_z, y_


# Sample function for sample sheets
def sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number, z_=None):
    # Prepare sample directory
    if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
        os.mkdir('%s/%s' % (samples_root, experiment_name))
    if not os.path.isdir('%s/%s/%d' % (samples_root, experiment_name, folder_number)):
        os.mkdir('%s/%s/%d' % (samples_root, experiment_name, folder_number))
    # loop over total number of sheets
    for i in range(num_classes // classes_per_sheet):
        ims = []

        import mindspore.numpy as mnp
        y = mnp.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet)
        for j in range(samples_per_class):
            j += 1
            shape_ = (classes_per_sheet, G.dim_z)
            stdnormal = mindspore.ops.StandardNormal()
            z_ = stdnormal(shape_)
            if parallel:
                o = nn.parallel.data_parallel(
                    G, (z_[:classes_per_sheet], G.shared(y)))
            else:
                o = G(z_[:classes_per_sheet], G.shared(y))
            try:
                ims += [o.data.cpu()]
                stack_ = mindspore.ops.Stack(1)
                out_ims = stack_([ims]).view(-1, ims[0].shape[1], ims[0].shape[2],
                                             ims[0].shape[3]).data.float().cpu()
                image_filename = '%s/%s/%d/samples%d.jpg' % (samples_root, experiment_name,
                                                             folder_number, i)
                img = Image.fromarray(out_ims)
                img.save(image_filename)
            except Exception as e:
                print(str(e))


def interp(x0, x1, num_midpoints):
    linespace_ = mindspore.ops.LinSpace()
    start_ = mindspore.Tensor(0.0, mindspore.float32)
    stop_ = mindspore.Tensor(1.0, mindspore.float32)

    try:
        lerp = linespace_(start_, stop_, num_midpoints + 2).astype(x0.dtype)
        return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))
    except Exception as e:
        print(str(e))


def print_grad_norms(net):
    norm_ = mindspore.nn.Norm()
    gradsums = [[float(norm_(param.grad).item()),
                 float(norm_(param).item()), param.shape]
                for param in net.parameters()]
    order = np.argsort([item[0] for item in gradsums])
    print(['%3.3e,%3.3e, %s' % (gradsums[item_index][0],
                                gradsums[item_index][1],
                                str(gradsums[item_index][2]))
           for item_index in order])


def get_SVs(net, prefix):
    #d = net.state_dict()
    d = net.parameters_dict()
    return {('%s_%s' % (prefix, key)).replace('.', '_'):
            float(d[key].item())
            for key in d if 'sv' in key}


# Name an experiment based on its config
def name_from_config(config):
    name = '_'.join([
        item for item in [
            'Big%s' % config['which_train_fn'],
            config['dataset'],
            config['model'] if config['model'] != 'BigGAN' else None,
            'seed%d' % config['seed'],
            'Gch%d' % config['G_ch'],
            'Dch%d' % config['D_ch'],
            'Gd%d' % config['G_depth'] if config['G_depth'] > 1 else None,
            'Dd%d' % config['D_depth'] if config['D_depth'] > 1 else None,
            'bs%d' % config['batch_size'],
            'Gfp16' if config['G_fp16'] else None,
            'Dfp16' if config['D_fp16'] else None,
            'nDs%d' % config['num_D_steps'] if config['num_D_steps'] > 1 else None,
            'nDa%d' % config['num_D_accumulations'] if config['num_D_accumulations'] > 1 else None,
            'nGa%d' % config['num_G_accumulations'] if config['num_G_accumulations'] > 1 else None,
            'Glr%2.1e' % config['G_lr'],
            'Dlr%2.1e' % config['D_lr'],
            config['name_suffix'] if config['name_suffix'] else None,
        ]
        if item is not None])
    # dogball
    if config['hashname']:
        return hashname(name)
    else:
        return name


# A simple function to produce a unique experiment name from the animal hashes.
def hashname(name):
    h = hash(name)
    a = h % len(animal_hash.a)
    h = h // len(animal_hash.a)
    b = h % len(animal_hash.b)
    h = h // len(animal_hash.c)
    c = h % len(animal_hash.c)
    return animal_hash.a[a] + animal_hash.b[b] + animal_hash.c[c]


def query_gpu(indices):
    os.system('nvidia-smi -i 0 --query-gpu=memory.free --format=csv')


def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))


class Distribution(mindspore.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']

        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            print("get self.mean", self.mean)
            print("get self.var", self.var)

        elif self.dist_type == 'categorical':
            #print("get self.num_categories", self.num_categories)
            variable = random.randint(0, self.num_categories)
            return variable

    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, fp16=False, z_var=1.0):
    standnormal = mindspore.ops.StandardNormal(seed=2)
    out = standnormal((G_batch_size, dim_z))
    z_ = Distribution(mindspore.Tensor(out))
    z_.init_distribution('normal', mean=0, var=z_var)
    if fp16:
        z_ = z_.half()

    zeros = mindspore.ops.Zeros()
    y_ = zeros(G_batch_size, mindspore.float32)
    y_ = Distribution(y_)
    y_.init_distribution('categorical', num_categories=nclasses)
    return z_, y_


def initiate_standing_stats(net):
    for module in net.modules():
        if hasattr(module, 'accumulate_standing'):
            module.reset_stats()
            module.accumulate_standing = True


class Adam16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

    # Safety modification to make sure we floatify our state
    def load_state_dict(self, state_dict):
        super(Adam16, self).load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
                self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
                self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                    state['fp32_p'] = p.data.float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], state['fp32_p'])
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1
                state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
                p.data = state['fp32_p'].half()

        return loss


class GenWithLossCell(nn.Cell):
    """GenWithLossCell"""

    def __init__(self, netG, netD, auto_prefix=True):
        super(GenWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.loss_fn = nn.BCELoss(reduction="mean")

    def construct(self, latent_code, label):
        fake_data = self.netG(latent_code)
        ones = mindspore.ops.OnesLike()(fake_data)
        loss_G = self.loss_fn(fake_data, ones)

        return loss_G


class DisWithLossCell(nn.Cell):
    """DisWithLossCell"""

    def __init__(self, netG, netD, auto_prefix=True):
        super(DisWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.loss_fn = nn.BCELoss(reduction="mean")

    def construct(self, real_data, label):
        """construct"""
        real_out = self.netD(real_data, label)
        ones = mindspore.ops.OnesLike()(real_out)
        real_loss = self.loss_fn(real_out, ones)
        loss_D = real_loss
        return loss_D


class TrainOneStepCell(nn.Cell):
    def __init__(self,
                 netG,
                 netD,
                 optimizerG: nn.Optimizer,
                 optimizerD: nn.Optimizer,
                 sens=1.0
                 ):

        super(TrainOneStepCell, self).__init__()
        self.netG = netG
        self.netG.set_grad()
        self.netG.add_flags(defer_inline=True)

        self.netD = netD
        self.netD.set_grad()
        self.netD.add_flags(defer_inline=True)

        self.weights_G = optimizerG.parameters
        self.optimizerG = optimizerG
        self.weights_D = optimizerD.parameters
        self.optimizerD = optimizerD
        self.grad = mindspore.ops.GradOperation(
            get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.grad_reducer_D = F.identity
        self.parallel_mode = mindspore.parallel._utils._get_parallel_mode()
        if self.parallel_mode in (mindspore.context.ParallelMode.DATA_PARALLEL,
                                  mindspore.context.ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = mindspore.context.get_auto_parallel_context(
                "gradients_mean")
            degree = mindspore.context.get_auto_parallel_context("device_num")
            self.grad_reducer_G = mindspore.nn.DistributedGradReducer(
                self.weights_G, mean, degree)
            self.grad_reducer_D = mindspore.nn.DistributedGradReducer(
                self.weights_D, mean, degree)

    def trainD(self, real_data, label, loss):
        """trainD"""
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.netD, self.weights_D)(real_data, label, sens)
        grads = self.grad_reducer_D(grads)
        self.optimizerD(grads)
        return loss

    def trainG(self, latent_code, label, loss):
        """trainG"""
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.netG, self.weights_G)(latent_code, label, sens)
        grads = self.grad_reducer_G(grads)
        self.optimizerG(grads)
        return loss

    def construct(self, real_data, latent_code, label):
        """construct"""
        loss_G = self.netG(latent_code, label)
        loss_D = self.netD(real_data, label)
        d_out = self.trainD(real_data, label, loss_D)
        g_out = self.trainG(latent_code, label, loss_G)
        return d_out, g_out
