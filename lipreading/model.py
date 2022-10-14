import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.losses.combine_margin_loss import CombineMarginLinear
from lipreading.models.densetcn import DenseTemporalConvNet
from lipreading.models.memory import Memory
from lipreading.models.resnet import BasicBlock, ResNet
from lipreading.models.resnet1D import BasicBlock1D, ResNet1D
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.swish import Swish
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths=None, B=None, average_dim=1):
    if lengths is None:
        return torch.mean(x, average_dim)
    else:
        return torch.stack([
            torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)
        ], 0)


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self,
                 input_size,
                 num_channels,
                 num_classes,
                 tcn_options,
                 dropout,
                 relu_type,
                 dwpw=False,
                 linear_config=None):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size,
                                                    num_channels,
                                                    tcn_options,
                                                    dropout=dropout,
                                                    relu_type=relu_type,
                                                    dwpw=dwpw)
        if linear_config.get('linear_type',
                             'Linear') in ['CombineMarginLinear']:
            linear_config.pop('linear_type')
            self.tcn_output = CombineMarginLinear(num_channels[-1],
                                                  num_classes, **linear_config)
        else:
            self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B, targets):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func(out, lengths, B)
        if isinstance(self.tcn_output, CombineMarginLinear):
            return self.tcn_output(out, targets)
        else:
            return self.tcn_output(out)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """
    def __init__(self,
                 input_size,
                 num_channels,
                 num_classes,
                 tcn_options,
                 dropout,
                 relu_type,
                 dwpw=False,
                 linear_config=None):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size,
                                         num_channels,
                                         dropout=dropout,
                                         tcn_options=tcn_options,
                                         relu_type=relu_type,
                                         dwpw=dwpw)
        if linear_config.get('linear_type',
                             'Linear') in ['CombineMarginLinear']:
            linear_config.pop('linear_type')
            self.tcn_output = CombineMarginLinear(num_channels[-1],
                                                  num_classes, **linear_config)
        else:
            self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B, targets):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        if isinstance(self.tcn_output, CombineMarginLinear):
            return self.tcn_output(x, targets)
        else:
            return self.tcn_output(x)


class DenseTCN(nn.Module):
    def __init__(self,
                 block_config,
                 growth_rate_set,
                 input_size,
                 reduced_size,
                 num_classes,
                 kernel_size_set,
                 dilation_size_set,
                 dropout,
                 relu_type,
                 squeeze_excitation=False,
                 linear_config=None):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1] * growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(
            block_config,
            growth_rate_set,
            input_size,
            reduced_size,
            kernel_size_set,
            dilation_size_set,
            dropout=dropout,
            relu_type=relu_type,
            squeeze_excitation=squeeze_excitation,
        )
        if linear_config.get('linear_type',
                             'Linear') in ['CombineMarginLinear']:
            linear_config.pop('linear_type')
            self.tcn_output = CombineMarginLinear(num_features, num_classes,
                                                  **linear_config)
        else:
            self.tcn_output = nn.Linear(num_features, num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B, targets):
        # B, C, T
        x = self.tcn_trunk(x.transpose(1, 2))
        # B, C_new, T
        x = self.consensus_func(x.transpose(1, 2), lengths, B)
        if isinstance(self.tcn_output, CombineMarginLinear):
            return self.tcn_output(x, targets)
        else:
            return self.tcn_output(x)


class Lipreading(nn.Module):
    def __init__(self,
                 modality='video',
                 hidden_dim=256,
                 backbone_type='resnet',
                 num_classes=500,
                 relu_type='prelu',
                 tcn_options={},
                 densetcn_options={},
                 width_mult=1.0,
                 use_boundary=False,
                 extract_feats=False,
                 linear_config=None,
                 predict_future=-1,
                 frontend_type='3D',
                 use_memory=False,
                 membanks_size=1024,
                 predict_residual=False,
                 predict_type=0,
                 block_size=4,
                 memory_type='memdpc',
                 memory_options={}):
        super(Lipreading, self).__init__()
        if linear_config is None:
            linear_config = {'linear_type': 'Linear'}
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.use_boundary = use_boundary
        self.linear_config = linear_config
        self.predict_future = predict_future
        self.frontend_type = frontend_type
        self.use_memory = use_memory
        self.membanks_size = membanks_size
        self.predict_residual = predict_residual
        self.predict_type = predict_type
        self.block_size = block_size
        self.memory_type = memory_type

        if self.modality == 'audio':
            self.frontend_nout = 1
            self.backend_out = 512
            self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2],
                                  relu_type=relu_type)
        elif self.modality == 'video':
            if self.backbone_type == 'resnet':
                self.frontend_nout = 64
                self.backend_out = 512
                self.trunk = ResNet(BasicBlock, [2, 2, 2, 2],
                                    relu_type=relu_type)
            elif self.backbone_type == 'shufflenet':
                assert width_mult in [0.5, 1.0, 1.5,
                                      2.0], "Width multiplier not correct"
                shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
                self.trunk = nn.Sequential(shufflenet.features,
                                           shufflenet.conv_last,
                                           shufflenet.globalpool)
                self.frontend_nout = 24
                self.backend_out = 1024 if width_mult != 2.0 else 2048
                self.stage_out_channels = shufflenet.stage_out_channels[-1]

            # -- frontend3D
            if relu_type == 'relu':
                frontend_relu = nn.ReLU(True)
            elif relu_type == 'prelu':
                frontend_relu = nn.PReLU(self.frontend_nout)
            elif relu_type == 'swish':
                frontend_relu = Swish()
            else:
                raise NotImplementedError(f'{relu_type} is not supported.')
            if self.frontend_type == '3D':
                self.frontend3D = nn.Sequential(
                    nn.Conv3d(1,
                              self.frontend_nout,
                              kernel_size=(5, 7, 7),
                              stride=(1, 2, 2),
                              padding=(2, 3, 3),
                              bias=False), nn.BatchNorm3d(self.frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d(kernel_size=(1, 3, 3),
                                 stride=(1, 2, 2),
                                 padding=(0, 1, 1)))
            elif self.frontend_type == '2D':
                self.frontend3D = nn.Sequential(
                    nn.Conv3d(1,
                              self.frontend_nout,
                              kernel_size=(1, 7, 7),
                              stride=(1, 2, 2),
                              padding=(0, 3, 3),
                              bias=False), nn.BatchNorm3d(self.frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d(kernel_size=(1, 3, 3),
                                 stride=(1, 2, 2),
                                 padding=(0, 1, 1)))
        else:
            raise NotImplementedError

        if self.predict_future > 0:
            if not self.use_memory:
                # input_size = B * T * self.backend_out
                self.network_pred = nn.Sequential(
                    nn.Linear(self.backend_out, self.backend_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.backend_out, self.backend_out))
            else:
                if self.memory_type == 'memdpc':
                    self.register_parameter('membanks', nn.Parameter(
                        torch.randn(self.membanks_size, self.backend_out)))
                    # input_size = B * T * self.backend_out
                    self.network_pred = nn.Sequential(
                        nn.Linear(self.backend_out, self.backend_out),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.backend_out, self.membanks_size))
                    print('MEM Bank has size %dx%d' %
                          (self.membanks_size, self.backend_out))
                elif self.memory_type == 'mvm':
                    self.memory = Memory(radius=memory_options['radius'],
                                         n_slot=memory_options['slot'],
                                         n_head=memory_options['head'])
                else:
                    raise RuntimeError(f'{self.memory_type} is not supported.')

        if tcn_options:
            tcn_class = TCN if len(
                tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
            self.tcn = tcn_class(
                input_size=self.backend_out,
                num_channels=[
                    hidden_dim * len(tcn_options['kernel_size']) *
                    tcn_options['width_mult']
                ] * tcn_options['num_layers'],
                num_classes=num_classes,
                tcn_options=tcn_options,
                dropout=tcn_options['dropout'],
                relu_type=relu_type,
                dwpw=tcn_options['dwpw'],
                linear_config=linear_config)
        elif densetcn_options:
            self.tcn = DenseTCN(
                block_config=densetcn_options['block_config'],
                growth_rate_set=densetcn_options['growth_rate_set'],
                input_size=self.backend_out
                if not self.use_boundary else self.backend_out + 1,
                reduced_size=densetcn_options['reduced_size'],
                num_classes=num_classes,
                kernel_size_set=densetcn_options['kernel_size_set'],
                dilation_size_set=densetcn_options['dilation_size_set'],
                dropout=densetcn_options['dropout'],
                relu_type=relu_type,
                squeeze_excitation=densetcn_options['squeeze_excitation'],
                linear_config=linear_config)
        else:
            raise NotImplementedError

        # -- initialize
        self._initialize_weights_randomly()

    def forward(self, x, lengths, boundaries=None, targets=None):
        if self.modality == 'video':
            B, C, T, H, W = x.size()
            x = self.frontend3D(x)
            Tnew = x.shape[2]  # outpuT should be B x C2 x Tnew x H x W
            x = threeD_to_2D_tensor(x)
            x = self.trunk(x)
            if self.backbone_type == 'shufflenet':
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.size(1))
            if self.predict_future > 0:
                if self.predict_type == 1:
                    time_chunks = torch.split(x, self.block_size, dim=1)
                    stride = 1
                    context_block_number = 2
                    predict_times = len(time_chunks) - context_block_number
                    feature_context = feature_target = None
                    for i in range(0, predict_times, stride):
                        # print(i, predict_times, len(time_chunks))
                        if feature_context is None:
                            feature_context = _average_batch(torch.cat(
                                time_chunks[i:i + 2], 1),
                                                             average_dim=1)
                            feature_target = _average_batch(time_chunks[i + 2])
                        else:
                            feature_context = torch.cat((_average_batch(
                                torch.cat(time_chunks[i:i + 2], 1),
                                average_dim=1), feature_context),
                                                        dim=0)
                            feature_target = torch.cat((_average_batch(
                                time_chunks[i + 2],
                                average_dim=1), feature_target),
                                                       dim=0)
                else:
                    # batch * x.size(1)
                    context_lengths = [_ // 2 for _ in lengths]
                    feature_context = _average_batch(x.transpose(1, 2),
                                                     context_lengths, B)
                    feature_target = torch.stack([
                        x[index, int(length * 0.75), :]
                        for index, length in enumerate(lengths)
                    ], 0)

                if not self.use_memory:
                    feature_predict = self.network_pred(feature_context)
                else:
                    target_recon_loss = contrastive_loss = None
                    if self.memory_type == 'memdpc':
                        predict_logits = self.network_pred(feature_context)
                        scores = F.softmax(predict_logits, dim=1)  # B,MEM,H,W
                        feature_predict = torch.einsum('bm,mc->bc', scores,
                                                       self.membanks)
                    else:
                        # input query, recon_target
                        feature_predict, feature_target_recon, target_recon_loss, contrastive_loss = self.memory(
                            feature_context.view(-1, 1,
                                                 feature_context.shape[1]),
                            feature_target.view(-1, 1, feature_target.shape[1]),
                            inference=False)
                        feature_predict = feature_predict.view(-1, feature_predict.shape[2])
                if self.predict_residual:
                    feature_target = feature_target - feature_predict

        elif self.modality == 'audio':
            B, C, T = x.size()
            x = self.trunk(x)
            x = x.transpose(1, 2)
            lengths = [_ // 640 for _ in lengths]

        # -- duration
        if self.use_boundary:
            x = torch.cat([x, boundaries], dim=-1)

        if self.extract_feats:
            return x
        else:
            if self.predict_future <= 0:
                return self.tcn(x, lengths, B, targets)
            else:
                return self.tcn(x, lengths, B, targets), feature_predict, feature_target, target_recon_loss, contrastive_loss

        # return x if self.extract_feats else self.tcn(x, lengths, B, targets)

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:

            def f(n):
                return math.sqrt(2.0 / float(n))
        else:

            def f(n):
                return 2.0 / float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(
                    m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(
                    m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
