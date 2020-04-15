#!/usr/bin/python
# -*- coding: utf-8 -*-

import io
import numpy as np
import queue
import collections
from PIL import Image
import os
import torch
import torch.nn as nn
import base64

import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm
try:
    from torch.nn.parallel._functions import ReduceAddCoalesced, \
        Broadcast
except ImportError:
    ReduceAddCoalesced = Broadcast = None

from flask import Flask, request, Response
import json

app = flask.Flask(__name__)


@app.route('/api/process', methods=['POST'])
def process():
    opt = {  # num classes in coco model
        'label_nc': 184,
        'crop_size': 256,
        'load_size': 256,
        'aspect_ratio': 1.0,
        'isTrain': False,
        'checkpoints_dir': '/content/drive/My Drive/be proj/coco_pretrained',
        'which_epoch': 'latest',
        'use_gpu': False,
        'no_instance': False,
        'gpu_ids': [],
        }
        
    model = Pix2PixModel(opt)
    model.eval()

    encodedImg = np.array(json.loads(request.form.get('data')))
    img = np.reshape(encodedImg, (1, 1, 256, 256))

    generated = model(img, mode='inference')
    print ('generated_image:', generated.shape)

    return str(generated.shape)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        ):

        assert ReduceAddCoalesced is not None, \
            'Can not use Synchronized Batch Normalization without CUDA support.'

        super(_SynchronizedBatchNorm, self).__init__(num_features,
                eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):

        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.

        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
                )

        # Resize the input to (B, C, -1).

        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.

        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.

        if self._parallel_id == 0:
            (mean, inv_std) = \
                self._sync_master.run_master(_ChildMessage(input_sum,
                    input_ssum, sum_size))
        else:
            (mean, inv_std) = \
                self._slave_pipe.run_slave(_ChildMessage(input_sum,
                    input_ssum, sum_size))

        # Compute the output.

        if self.affine:

            # MJY:: Fuse the multiplication for speed.

            output = (input - _unsqueeze_ft(mean)) \
                * _unsqueeze_ft(inv_std * self.weight) \
                + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) \
                * _unsqueeze_ft(inv_std)

        # Reshape it.

        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.

        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)

        intermediates = sorted(intermediates, key=lambda i: \
                               i[1].sum.get_device())

        to_reduce = [(i[1])[:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        (sum_, ssum) = ReduceAddCoalesced.apply(target_gpus[0], 2,
                *to_reduce)
        (mean, inv_std) = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for (i, rec) in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i
                           * 2 + 2])))

        return outputs

    def _compute_mean_std(
        self,
        sum_,
        ssum,
        size,
        ):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""

        assert size > 1, \
            'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        if hasattr(torch, 'no_grad'):
            torch.no_grad()
            self.running_mean = (1 - self.momentum) * self.running_mean \
                + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var \
                + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean \
                + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var \
                + self.momentum * unbias_var.data

        return (mean, bias_var.clamp(self.eps) ** -0.5)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):

    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape::
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class Pix2PixModel(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = (torch.cuda.FloatTensor if opt['use_gpu'
                            ] else torch.FloatTensor)
        self.ByteTensor = (torch.cuda.ByteTensor if opt['use_gpu'
                           ] else torch.ByteTensor)

        self.netG = self.initialize_networks(opt)

    def forward(self, label_tensor, mode='inference'):

        data = {'label': label_tensor, 'instance': label_tensor}
        input_semantics = self.preprocess_input(data)

        if mode == 'inference':
            torch.no_grad()
            fake_image = self.generate_fake(input_semantics)
            return fake_image
        else:
            raise ValueError('|mode| is invalid')

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge = edge.bool()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:,
                :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:
                , :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:,
                :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:
                , :, :-1, :])
        return edge.float()

    def preprocess_input(self, data):
        data['label'] = data['label'].long()

    # move to GPU and change data types

        if self.opt['use_gpu']:
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()

    # create one-hot label map

        label_map = data['label']
        (bs, _, h, w) = label_map.size()
        input_label = self.FloatTensor(bs, self.opt['label_nc'] - 1, h,
                w).zero_()

    # one whole label map -> to one label map per class

        input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists

        if not self.opt['no_instance']:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics,
                    instance_edge_map), dim=1)

        return input_semantics

    def generate_fake(self, input_semantics):
        fake_image = self.netG(input_semantics)
        return fake_image

    def create_network(self, cls, opt):
        net = cls(opt)
        if self.opt['use_gpu']:
            net.cuda()

        gain = 0.02

        def init_weights(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                    or classname.find('Linear') != -1):

                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    # Applies fn recursively to every submodule (as returned by .children()) as well as self

        net.apply(init_weights)

        return net

    def load_network(
        self,
        net,
        label,
        epoch,
        opt,
        ):

        save_filename = '%s_net_%s.pth' % (epoch, label)
        save_path = os.path.join(opt['checkpoints_dir'], save_filename)
        weights = torch.load(save_path)
        net.load_state_dict(weights)
        return net

    def initialize_networks(self, opt):
        netG = self.create_network(SPADEGenerator, opt)

        if not opt['isTrain']:
            netG = self.load_network(netG, 'G', opt['which_epoch'], opt)

    # self.print_network(netG)

        return netG


class SPADEGenerator(nn.Module):

    def __init__(self, opt):
        super().__init__()

    # nf: # of gen filters in first conv layer

        nf = 64

        (self.sw, self.sh) = \
            self.compute_latent_vector_size(opt['crop_size'],
                opt['aspect_ratio'])

        self.fc = nn.Conv2d(opt['label_nc'], 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(opt, 16 * nf, 16 * nf)

        self.G_middle_0 = SPADEResnetBlock(opt, 16 * nf, 16 * nf)
        self.G_middle_1 = SPADEResnetBlock(opt, 16 * nf, 16 * nf)

        self.up_0 = SPADEResnetBlock(opt, 16 * nf, 8 * nf)
        self.up_1 = SPADEResnetBlock(opt, 8 * nf, 4 * nf)
        self.up_2 = SPADEResnetBlock(opt, 4 * nf, 2 * nf)
        self.up_3 = SPADEResnetBlock(opt, 2 * nf, 1 * nf)

        self.conv_img = nn.Conv2d(1 * nf, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, crop_size, aspect_ratio):
        num_up_layers = 5

        sw = crop_size // 2 ** num_up_layers
        sh = round(sw / aspect_ratio)

        return (sw, sh)

    def forward(self, seg):

    # we downsample segmap and run convolution

        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


import torch.nn.utils.spectral_norm as spectral_norm


# label_nc: the #channels of the input semantic map, hence the input dim of SPADE
# label_nc: also equivalent to the # of input label classes

class SPADEResnetBlock(nn.Module):

    def __init__(
        self,
        opt,
        fin,
        fout,
        ):

        super().__init__()

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle,
                                    kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout,
                                    kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout,
                    kernel_size=1, bias=False))

    # define normalization layers

        self.norm_0 = SPADE(opt, fin)
        self.norm_1 = SPADE(opt, fmiddle)
        if self.learned_shortcut:
            self.norm_s = SPADE(opt, fin)

  # note the resnet block with SPADE also takes in |seg|,
  # the semantic segmentation map as input

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.relu(self.norm_0(x, seg)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg)))

        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def relu(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):

    def __init__(self, opt, norm_nc):
        super().__init__()

        self.param_free_norm = SynchronizedBatchNorm2d(norm_nc,
                affine=False)

    # number of internal filters for generating scale/bias

        nhidden = 128

    # size of kernels

        kernal_size = 3

    # padding size

        padding = kernal_size // 2

        self.mlp_shared = nn.Sequential(nn.Conv2d(opt['label_nc'],
                nhidden, kernel_size=kernal_size, padding=padding),
                nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc,
                                   kernel_size=kernal_size,
                                   padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc,
                                  kernel_size=kernal_size,
                                  padding=padding)

    def forward(self, x, segmap):

    # Part 1. generate parameter-free normalized activations

        normalized = self.param_free_norm(x)

    # Part 2. produce scaling and bias conditioned on semantic map
    # resize input segmentation map to match x.size() using nearest interpolation
    # N, C, H, W = x.size()

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest'
                               )
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

    # apply scale and bias

        out = normalized * (1 + gamma) + beta

        return out


from torch.nn.parallel.data_parallel import DataParallel


class SyncMaster(object):

    """An abstract `SyncMaster` object.
    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """
        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """

        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.
        Args:
            identifier: an identifier, usually is the device id.
        Returns: a `SlavePipe` object which can be used to communicate with the master device.
        """

        if self._activated:
            assert self._queue.empty(), \
                'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).
        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.
        Returns: the message to be sent back to the master device.
        """

        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, \
            'The first result should belongs to the master.'

        for (i, res) in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.
    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """

    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for (i, module) in enumerate(modules):
        for (j, m) in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):

    """
    Data Parallel with a replication callback.
    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback,
                        self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.
    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate


