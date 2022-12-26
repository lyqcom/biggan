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

import numpy as np
import mindspore
# pylint: disable=R1705
# pylint: disable=C0330
# pylint: disable=R1719
# pylint: disable=C0326


def proj(x, y):
    mm_ = mindspore.ops.MatMul()
    return mm_(y, x.t()) * y / mm_(y, y.t())


def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        matmul_ = mindspore.nn.MatMul()
        v = matmul_(u, W)
        # Run Gram-Schmidt to subtract components of all other singular vectors
        normalize_ = mindspore.ops.L2Normalize(epsilon=eps)
        v = normalize_(gram_schmidt(v, vs))
        vs += [v]
        matmul_ = mindspore.nn.MatMul()
        u = matmul_(v, W.t())
        u = normalize_(gram_schmidt(u, us))
        # Add to the list
        us += [u]
        if update:
            u_[i][:] = u
        squeeze_ = mindspore.ops.Squeeze()
        svs += [squeeze_(matmul_(matmul_(v, W.t()), u.t()))]

    return svs, us, vs


class identity(mindspore.nn.Cell):
    def construct(self, input_):
        return input_


class SN():
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps

    def register_buffer(self, elem1, elem2):
        pass

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        print("get self.weight.size(0):", self.weight.shape[0])
        W_mat = self.weight.view(self.weight.shape[0], -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        svs = []
        for _ in range(self.num_itrs):
            svs, dummy_us, dummy_vs = power_iteration(
                W_mat, self.u, update=self.training, eps=self.eps)
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(mindspore.nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        mindspore.nn.Conv2d.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, pad_mode="pad", padding=padding, dilation=dilation, group=groups,
                                     has_bias=bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def construct(self, x):
        conv2d_ = mindspore.ops.Conv2D(out_channel=self.out_channels, pad=self.padding,
                                       stride=self.stride, dilation=self.dilation, group=self.groups)
        return conv2d_(x, self.W_())


# Linear layer with spectral norm
class SNLinear(mindspore.nn.Dense, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        mindspore.nn.Dense.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def construct(self, x):
        linear_ = mindspore.nn.Dense(self.weight.shape[1], 100)
        return linear_(x)


class SNEmbedding(mindspore.nn.Embedding, SN):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None,
                 num_svs=1, num_itrs=1, eps=1e-12):
        mindspore.nn.Embedding.__init__(self, vocab_size=num_embeddings, embedding_size=embedding_dim,
                                        padding_idx=padding_idx, dtype=mindspore.float32)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def construct(self, x):
        embedding_ = mindspore.nn.Embedding(x.shape[1]+x.shape[0], x.shape[0])
        return embedding_(x)


class Attention(mindspore.nn.Cell):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        self.gamma = mindspore.Parameter(
            mindspore.Tensor(0.), requires_grad=True)

    def construct(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = mindspore.nn.MaxPool2d(self.phi(x), [2, 2])
        g = mindspore.nn.MaxPool2d(self.g(x), [2, 2])
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)

        softmax_ = mindspore.ops.Softmax()
        batmatmul_ = mindspore.ops.BatchMatMul()
        beta = softmax_(batmatmul_(theta.transpose(1, 2), phi))
        o = self.o(batmatmul_(g, beta.transpose(1, 2)).view(-1,
                   self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    else:
        groups = 16
    group_normal_ = mindspore.nn.GroupNorm(num_groups=groups)
    return group_normal_(x)


class ccbn(mindspore.nn.Cell):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False, norm_style='bn',):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style

        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps,
                               momentum=self.momentum, affine=False)

    def construct(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.mybn or self.cross_replica:
            return self.bn(x, gain=gain, bias=bias)
        else:
            out = 0.0
            if self.norm_style == 'bn':
                batch_norm_ = mindspore.ops.BatchNorm(
                    is_training=self.training, epsilon=self.eps, momentum=0.1)
                out = batch_norm_(
                    input_x=x, mean=self.stored_mean, variance=self.stored_var)
            elif self.norm_style == 'in':
                instance_norm_ = mindspore.nn.InstanceNorm2d(
                    eps=self.eps, momentum=0.1)
                out = instance_norm_(x)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s += ' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


class bn(mindspore.nn.Cell):
    def __init__(self, output_size,  eps=1e-5, momentum=0.1, cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.output_size = output_size
        self.gain = mindspore.Parameter(mindspore.Tensor(
            np.ones(output_size), mindspore.float32), requires_grad=True)
        self.bias = mindspore.Parameter(mindspore.Tensor(
            np.zeros(output_size), mindspore.float32), requires_grad=True)
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        self.register_buffer = mindspore.Parameter(mindspore.Tensor(np.zeros(
            output_size), mindspore.float32), name="stored_mean", requires_grad=True)

    def construct(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                                self.bias, self.training, self.momentum, self.eps)


class GBlock(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels,
                 which_conv=mindspore.nn.Conv2d, which_bn=bn, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def construct(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class DBlock(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (
            in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def construct(self, x):
        if self.preactivation:
            h_ = mindspore.nn.ReLU()
            h = h_(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)
