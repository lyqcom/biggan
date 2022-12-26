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

import math
import time

from numpy import dtype
import mindspore
import mindspore.nn as nn
from mindspore.common import initializer as init
# pylint: disable=R1705
# pylint: disable=W0702
# pylint: disable=C0121
# pylint: disable=C0411


def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
                 'upsample': [True] * 6,
                 'resolution': [8, 16, 32, 64, 128, 256],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 9)}}
    arch[128] = {'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample': [True] * 5,
                 'resolution': [8, 16, 32, 64, 128],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 8)}}
    arch[64] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)}}
    arch[32] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)}}

    return arch


def D_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [3] + [ch*item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [3] + [ch*item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [3] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    return arch


def init_weights(net, init_type='normal', init_gain=0.02):
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(
                    init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(
                    init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'KaimingUniform':
                cell.weight.set_data(init.initializer(
                    init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(
                    init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.GroupNorm):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


class Generator(mindspore.nn.Cell):
    def __init__(self, input_dim, output_dim=1, dim_z=128, resolution=128, class_num=10, **kwargs):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_z = dim_z
        self.resolution = resolution
        self.class_num = class_num
        self.concat = mindspore.ops.operations.Concat(1)
        self.dense = nn.Dense(128, 1024).to_float(mindspore.float16)
        self.batchnorm2d = nn.BatchNorm2d(3)
        self.relu = nn.ReLU().to_float(mindspore.float32)
        self.dense_2 = nn.Dense(
            1024, 128 * (self.resolution // 4) * (self.resolution // 4)).to_float(mindspore.float16)
        self.batchnorm2d_2 = nn.BatchNorm2d(
            128 * (self.resolution // 4) * (self.resolution // 4))
        self.batchnorm2d_2 = nn.BatchNorm2d(3)

        self.resblock_1 = nn.SequentialCell(
            nn.Conv2dTranspose(128, 64, 4, 2, padding=0,
                               has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2dTranspose(64, self.output_dim, 4, 2,
                               padding=0, has_bias=True, pad_mode='same'),
            nn.Tanh(),
        )
        self.resblock_2 = nn.SequentialCell(
            nn.Conv2d(self.output_dim, 32, self.output_dim+1, self.output_dim-1, padding=0,
                      has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.output_dim, self.output_dim+1, self.output_dim-1,
                      padding=0, has_bias=True, pad_mode='same'),
            nn.Tanh(),
        )
        self.resblock_3 = nn.SequentialCell(
            nn.Conv2d(self.output_dim, 16, self.output_dim+1, self.output_dim-1, padding=0,
                      has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, self.output_dim, self.output_dim+1, self.output_dim-1,
                      padding=0, has_bias=True, pad_mode='same'),
            nn.Tanh(),
        )
        init_weights(self.resblock_1, 'xavier', math.sqrt(5))
        self.tanh = nn.Tanh()

    def construct(self, input_param):
        """construct"""
        x = self.dense(input_param)
        x = self.batchnorm2d(x)
        x = self.relu(x)
        x = self.dense_2(x)
        x = self.batchnorm2d_2(x)
        x = self.relu(x)
        x = x.view(-1, 128, (self.resolution // 4), (self.resolution // 4))
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.tanh(x)
        return x


class Discriminator(mindspore.nn.Cell):
    def __init__(self, batch_size, input_dim=1, output_dim=1, resolution=128, class_num=10, **kwargs):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resolution = resolution
        self.class_num = class_num
        self.concat = mindspore.ops.Concat(1)
        self.ExpandDims = mindspore.ops.ExpandDims()

        self.conv = nn.SequentialCell(
            nn.Conv2d(3, 64, 4, 2,
                      padding=0, has_bias=True, pad_mode='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, padding=0,
                      has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.resblock_1 = nn.SequentialCell(
            nn.Conv2d(self.resolution, 64, 4, 2,
                      padding=0, has_bias=True, pad_mode='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, padding=0,
                      has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.resblock_2 = nn.SequentialCell(
            nn.Conv2d(self.resolution, 64, 4, 2,
                      padding=0, has_bias=True, pad_mode='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, 2, padding=0,
                      has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.resblock_3 = nn.SequentialCell(
            nn.Conv2d(self.resolution//2, 64, 4, 2,
                      padding=0, has_bias=True, pad_mode='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 4, 2, padding=0,
                      has_bias=True, pad_mode='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        self.embed = nn.SequentialCell(
            nn.Dense(self.resolution // 4, 16).to_float(mindspore.float16),
            nn.LeakyReLU(0.2),
            nn.Dense(16, self.output_dim).to_float(mindspore.float16),
            nn.Sigmoid().to_float(mindspore.float32),
        )

        init_weights(self.conv, 'xavier', math.sqrt(5))

    def construct(self, input_param, label):
        x = self.conv(input_param)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = x.view(-1, self.resolution // 4)
        x = self.embed(x)
        return x


class G_D(mindspore.nn.Cell):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def construct(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
                  split_D=False):
        G_z = self.G(z, gy)
        if self.G.fp16 and not self.D.fp16:
            G_z = G_z.float()
        if self.D.fp16 and not self.G.fp16:
            G_z = G_z.half()
        if split_D:
            D_fake = self.D(G_z, gy)
            if x is not None:
                D_real = self.D(x, dy)
                return D_fake, D_real
            else:
                if return_G_z:
                    return D_fake, G_z
                else:
                    return D_fake
        else:
            if x is not None:
                x = mindspore.ops.expand_dims(x, 0)
            D_input = G_z
            D_class = gy
            D_out = self.D(D_input, D_class)
            if x is not None:
                flatten = mindspore.ops.Flatten()
                D_out = flatten(D_out)[:, :16]
                split_ = mindspore.ops.Split(1, 2)
                o_ = split_(D_out)
                return o_
            else:
                if return_G_z:
                    return D_out, G_z
                else:
                    return D_out
