import dataclasses
from collections import OrderedDict
from typing import List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels.models.senet import SENet, Bottleneck, SEResNetBottleneck, SEModule

EPSILON = 1e-10


@dataclasses.dataclass
class CnnBlockParams:
    in_feature: int
    middle_feature: int
    out_feature: int
    dropout_rate: int
    kernel_sizes: Union[Tuple[int, int], int]
    strides: Union[Tuple[int, int], int]
    padding: Union[Tuple[int, int], int]
    pool_kernel_size: int
    pool_padding: int
    pool_stride: int
    dilation: int = 1
    n_dropouts: int = 1
    concat_pool: bool = True


@dataclasses.dataclass
class GruParams:
    seq_len: int
    input_size: int
    hidden_size: int
    gru_dropout: int
    dense_output: int
    dropout: int
    num_layers: int = 2


class StackedGRU(nn.Module):

    def __init__(self, params: GruParams):
        super().__init__()
        self.grus = nn.GRU(params.input_size, hidden_size=params.hidden_size, num_layers=params.num_layers,
                           bidirectional=True,
                           dropout=params.gru_dropout, batch_first=True)
        self.hidden_size = params.hidden_size * 2
        self.classify = nn.Sequential(
            nn.Linear(self.hidden_size * 3, params.dense_output),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.dense_output, 1)
        )

    def forward(self, x):
        self.grus.flatten_parameters()
        x, _ = self.grus(x)
        max_pooled = F.max_pool2d(x, kernel_size=(x.shape[1], 1)).squeeze()
        avg_pooled = F.avg_pool2d(x, kernel_size=(x.shape[1], 1)).squeeze()
        last_hidden = x[:, -1, :].squeeze()
        x = torch.cat([max_pooled, avg_pooled, last_hidden], dim=1)
        assert len(x.shape) == 2
        x = self.classify(x)
        return x


class StackAttentionGRU(nn.Module):

    def __init__(self, params: GruParams):
        super().__init__()
        self.grus = nn.GRU(params.input_size, hidden_size=params.hidden_size, num_layers=params.num_layers,
                           bidirectional=True,
                           dropout=params.gru_dropout, batch_first=True)
        self.hidden_size = params.hidden_size * 2
        self.attention = Attention(self.hidden_size, params.seq_len)
        self.classify = nn.Sequential(
            nn.Linear(self.hidden_size, params.dense_output),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.dense_output, 1)
        )

    def forward(self, x):
        self.grus.flatten_parameters()
        if len(x.shape) > 3:
            x = x.view((x.shape[0], x.shape[1], -1))
        x, _ = self.grus(x)
        x = self.attention(x)
        x = self.classify(x)
        return x


class StackAttentionLSTM(nn.Module):

    def __init__(self, params: GruParams):
        super().__init__()
        self.grus = nn.LSTM(params.input_size, hidden_size=params.hidden_size, num_layers=params.num_layers,
                            bidirectional=True,
                            dropout=params.gru_dropout, batch_first=True)
        self.hidden_size = params.hidden_size * 2
        self.attention = Attention(self.hidden_size, params.seq_len)
        self.classify = nn.Sequential(
            nn.Linear(self.hidden_size, params.dense_output),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.dense_output, 1)
        )

    def forward(self, x):
        self.grus.flatten_parameters()
        x, _ = self.grus(x)
        x = self.attention(x)
        x = self.classify(x)
        return x


@dataclasses.dataclass
class HierarchicalGruParams:
    seq_len: int
    chunk_len: int
    input_size: int
    hidden_sizes: (int, int)
    gru_dropout: int
    dense_output: int
    dropout: int
    num_layers: int = 1
    pool_size: int = 0
    pool_stride: int = 0
    attention_type: str = "self"
    context_size: (int, int) = (0, 0)
    use_hidden: bool = False


class HierarchicalAttentionGru(nn.Module):
    def __init__(self, params: HierarchicalGruParams):
        super().__init__()
        self.params = params
        self.chunk_grus = nn.GRU(params.input_size, hidden_size=params.hidden_sizes[0], num_layers=params.num_layers,
                                 bidirectional=True,
                                 dropout=params.gru_dropout, batch_first=True)
        self.chunk_hidden_size = params.hidden_sizes[0] * 2
        if params.attention_type == "self":
            self.chunk_attention = Attention(self.chunk_hidden_size, params.chunk_len)
        else:
            self.chunk_attention = ContextAttention(self.chunk_hidden_size, params.context_size[0], params.chunk_len)

        self.seq_grus = nn.GRU(input_size=self.chunk_hidden_size, hidden_size=params.hidden_sizes[1],
                               num_layers=params.num_layers,
                               bidirectional=True,
                               dropout=params.gru_dropout, batch_first=True)
        self.seq_hidden_size = params.hidden_sizes[1] * 2

        if params.attention_type == "self":
            self.seq_attention = Attention(self.seq_hidden_size, params.seq_len)
        else:
            self.seq_attention = ContextAttention(self.seq_hidden_size, params.context_size[1], params.seq_len)

        self.classify = nn.Sequential(
            nn.Linear(self.seq_hidden_size, params.dense_output),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.dense_output, 1)
        )
        self.use_hidden = params.use_hidden

    def forward(self, x):
        # x = x.cuda()
        # x = x.transpose(1, -1)
        # print(x.shape)
        if self.params.pool_size:
            x = torch.cat([F.max_pool2d(x, (self.params.pool_size, 1), stride=(self.params.pool_stride, 1),
                                        padding=(self.params.pool_stride, 0)),
                           F.avg_pool2d(x, (self.params.pool_stride, 1), stride=(self.params.pool_stride, 1),
                                        padding=(self.params.pool_stride, 0))], dim=2)
        src_shape = x.shape
        x = x.contiguous().view((-1, self.params.chunk_len, src_shape[-1]))
        # print(x.shape)
        self.chunk_grus.flatten_parameters()

        x, _ = self.chunk_grus(x)

        x = self.chunk_attention(x)

        x = x.view((src_shape[0], -1, x.shape[-1]))
        self.seq_grus.flatten_parameters()

        x, _ = self.seq_grus(x)
        x = self.seq_attention(x)

        x = self.classify(x)
        return x


class Attention(nn.Module):

    def __init__(self, input_feature, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_feature = input_feature
        self.layer = nn.Sequential(
            nn.Linear(input_feature, 1, bias=True),
            nn.Tanh()
        )

        for module in self.layer.modules():
            self.init_weights(module)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()

    def forward(self, x):
        # print("attention_input:" + str(x.shape))
        attended = x.contiguous().view((-1, self.input_feature))
        attended = self.layer(attended).contiguous().view((-1, self.seq_len))

        # attended = attended / torch.sum(attended + EPSILON, dim=1).view((-1, 1))
        attended = F.softmax(attended, dim=1).view((x.shape[0], self.seq_len, 1))
        weighted_input = x * attended
        return torch.sum(weighted_input, dim=1)


class ContextAttention(nn.Module):

    def __init__(self, input_feature, context_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_feature = input_feature
        self.context_size = context_size
        self.layer = nn.Sequential(
            nn.Linear(input_feature, self.context_size, bias=True),
            nn.Tanh()
        )
        self.chunk_vector = nn.Parameter(torch.empty((self.context_size, 1)).uniform_(0, 1), requires_grad=True)

        for module in self.layer.modules():
            self.init_weights(module)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()

    def forward(self, x):
        # print("attention_input:" + str(x.shape))
        attended = x.contiguous().view((-1, self.input_feature))
        # print(attended.shape)
        attended = self.layer(attended).contiguous().view((-1, self.seq_len, self.context_size))
        # print(attended.shape)
        # attended = attended / torch.sum(attended + EPSILON, dim=1).view((-1, 1))
        attended = torch.matmul(attended, self.chunk_vector)
        # print(attended.shape)
        attended = F.softmax(attended, dim=1).view((x.shape[0], self.seq_len, 1))
        weighted_input = x * attended
        return torch.sum(weighted_input, dim=1)


class SEResNet1DBottleneck(SEResNetBottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, kernel_size=3):
        super().__init__(inplanes, planes, groups, reduction, stride, downsample)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SE1DModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SE1DModule(SEModule):

    def __init__(self, channels, reduction):
        super(SE1DModule, self).__init__(channels, reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()


@dataclasses.dataclass
class SENetParams:
    input_height: int
    block: Bottleneck
    layers: List[int]
    groups: int
    reduction: int
    dropout_p: int = 0.2
    inplanes: int = 128
    downsample_kernel_size: int = 3
    downsample_padding: int = 1
    last_pool_size: int = 7
    block_stride: int = 2
    should_transpose: bool = True
    first_kernel_size: int = 3
    first_stride: int = 2
    n_first_conv: int = 1


#
# SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
#                   dropout_p=None, inplanes=64,


class SignalSENet(SENet):
    def __init__(self, params: SENetParams):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """

        self.input_height = params.input_height
        block = params.block
        layers = params.layers
        groups = params.groups
        reduction = params.reduction
        dropout_p = params.dropout_p
        inplanes = params.inplanes
        downsample_kernel_size = params.downsample_kernel_size
        downsample_padding = params.downsample_padding
        super().__init__(block, layers, groups, reduction)
        self.params = params

        self.inplanes = inplanes
        # if input_3x3:
        #     layer0_modules = [
        #         ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
        #                             bias=False)),
        #         ('bn1', nn.BatchNorm2d(64)),
        #         ('relu1', nn.ReLU(inplace=True)),
        #         ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
        #                             bias=False)),
        #         ('bn2', nn.BatchNorm2d(64)),
        #         ('relu2', nn.ReLU(inplace=True)),
        #         ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
        #                             bias=False)),
        #         ('bn3', nn.BatchNorm2d(inplanes)),
        #         ('relu3', nn.ReLU(inplace=True)),
        #     ]
        # else:
        layer0_modules = [
            ('conv1', nn.Conv2d(in_channels=1,
                                out_channels=inplanes, kernel_size=(self.input_height, params.first_kernel_size),
                                stride=(1, params.first_stride),
                                padding=0, bias=False)),
            ('bn1', nn.BatchNorm2d(inplanes)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer0_modules.append(('pool1', nn.MaxPool2d((1, params.first_kernel_size),
                                                     stride=(1, params.first_stride),
                                                     ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer0_optional = []
        if params.n_first_conv > 1:
            for i in range(params.n_first_conv - 1):
                self.layer0_optional.extend([
                    ('conv1_{}'.format(i), nn.Conv1d(in_channels=inplanes,
                                                     out_channels=inplanes,
                                                     kernel_size=params.first_kernel_size,
                                                     stride=params.first_stride,
                                                     padding=0, bias=False)),
                    ('bn1_{}'.format(i), nn.BatchNorm1d(inplanes)),
                    ('relu1_{}'.format(i), nn.ReLU(inplace=True)),
                    ('pool1_{}'.format(i),
                     nn.MaxPool1d(params.first_kernel_size, stride=params.first_stride,
                                  ceil_mode=True))
                ]
                )
            self.layer0_optional = nn.Sequential(OrderedDict(self.layer0_optional))

        self.layer1 = self._make_layer(
            block,
            planes=32,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=32,
            blocks=layers[1],
            stride=params.block_stride,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=64,
            blocks=layers[2],
            stride=params.block_stride,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=128,
            blocks=layers[3],
            stride=params.block_stride,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        # self.avg_pool = nn.AvgPool1d(params.last_pool_size, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(128 * block.expansion, 1)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1,
                    downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        # print(x.shape)
        if self.params.should_transpose:
            x = x.transpose(1, 2)
        x = x.view((x.shape[0], 1, self.input_height, -1))
        x = self.layer0(x)
        # print(x.shape)
        # print(x.shape)
        x = x.squeeze(dim=2)
        if self.params.n_first_conv > 1:
            x = self.layer0_optional(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.logits(x)
    #     return x


@dataclasses.dataclass
class CnnBlockMultipleDropoutParams(CnnBlockParams):
    pass


def multiple_dropouts(x, dropout_rate, n_dropouts):
    x = torch.mean(
        torch.stack([F.dropout(x, dropout_rate) for i in range(n_dropouts)], dim=-1),
        dim=-1)
    return x


class CnnMultipleDropoutBlock(nn.Module):

    def __init__(self, params: CnnBlockMultipleDropoutParams):
        super().__init__()
        # self.params = params
        self.pool_stride = params.pool_stride
        self.n_dropouts = params.n_dropouts
        self.dropout_rate = params.dropout_rate
        self.in_feature = params.in_feature
        self.in_feature = params.in_feature
        self.out_feature = params.out_feature
        self.padding = params.padding
        self.kernel_size = params.kernel_sizes
        self.dilation = params.dilation
        self.strides = params.strides
        self.pool_kernel_size = params.pool_kernel_size

        self.block_1_1, self.block_1_2 = self._block()
        self.block_2_1, self.block_2_2 = self._block()

    def _block(self):
        return (nn.Sequential(
            nn.Conv1d(in_channels=self.in_feature, out_channels=self.out_feature, padding=self.padding,
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=self.strides),
            nn.BatchNorm1d(self.in_feature),
            nn.ReLU()),
                nn.Sequential(
                    nn.Conv1d(in_channels=self.out_feature, out_channels=self.out_feature, padding=self.padding,
                              stride=self.strides, kernel_size=self.kernel_size),
                    nn.BatchNorm1d(self.out_feature)
                ))

    def forward(self, x):
        x = [self.block_1_1(x), self.block_2_1(x)]
        x = [multiple_dropouts(each_x, dropout_rate=self.dropout_rate, n_dropouts=self.n_dropouts)
             for each_x in x]
        x = torch.cat([
            self.block_1_2(x[0]),
            self.block_2_2(x[1])
        ], dim=1)
        x = F.glu(x, dim=1)

        if self.pool_kernel_size > 0:
            x = torch.cat([F.max_pool1d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride),
                           F.avg_pool1d(x, kernel_size=self.pool_kernel_size,
                                        stride=self.pool_stride)], dim=1)
        return x

    @staticmethod
    def pad_to_even(x):
        if x.shape[2] % 2:
            x = F.pad(x, (1, 0))
        return x


class CnnBasicBlock(nn.Module):

    def __init__(self, params: CnnBlockParams):
        super().__init__()
        # self.params = params
        self.pool_stride = params.pool_stride
        # self.n_dropouts = params.n_dropouts
        self.dropout_rate = params.dropout_rate
        self.in_feature = params.in_feature
        self.in_feature = params.in_feature
        self.middle_feature = params.middle_feature
        self.out_feature = params.out_feature
        self.padding = params.padding
        self.kernel_size = params.kernel_sizes
        self.dilation = params.dilation
        self.strides = params.strides
        self.pool_kernel_size = params.pool_kernel_size
        self.n_dropout = params.n_dropouts

        self.block1 = self._block(in_channels=self.in_feature, out_channels=self.middle_feature, padding=self.padding,
                                  kernel_size=self.kernel_size, dilation=self.dilation, stride=self.strides)
        self.block_2 = self._block(in_channels=self.middle_feature, out_channels=self.out_feature, padding=self.padding,
                                   kernel_size=self.kernel_size, dilation=self.dilation, stride=self.strides)
        self.concat_pool = params.concat_pool

    def _block(self, in_channels, out_channels, padding,
               kernel_size, dilation, stride):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, padding=padding,
                      kernel_size=kernel_size, dilation=dilation, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())

    def forward(self, x):
        # print(x)
        x = self.block1(x)
        if self.n_dropout > 1:
            x = multiple_dropouts(x, self.dropout_rate, self.n_dropout)
        else:
            x = F.dropout(x, self.dropout_rate)
        x = self.block_2(x)
        x = F.relu(x)

        if self.pool_kernel_size > 0:
            if self.concat_pool:
                x = torch.cat([F.max_pool1d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride),
                               F.avg_pool1d(x, kernel_size=self.pool_kernel_size,
                                            stride=self.pool_stride)], dim=1)
            else:
                x = F.max_pool1d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride)

        return x


@dataclasses.dataclass
class BasicCnnAttentionParams(nn.Module):
    num_classes: int
    cnn_blocks: List[CnnBlockParams]
    last_chunk_len: int
    context_size: int
    last_dropout_rate: int
    last_n_dropouts: int = 1
    concat_pool: bool = False


@dataclasses.dataclass
class CnnAttentionMultipleDropoutParams(BasicCnnAttentionParams):
    num_classes: int
    cnn_blocks: List[CnnBlockMultipleDropoutParams]
    last_chunk_len: int
    context_size: int
    concat_pool: bool = False


class CnnAttentionMultipleDropoutModule(nn.Module):

    def __init__(self, params: CnnAttentionMultipleDropoutParams):
        super().__init__()
        self.n_dropouts = params.last_n_dropouts
        self.dropout_rate = params.last_dropout_rate
        # self.params = params
        first_params = params.cnn_blocks[0]
        self.first_conv_1 = nn.Sequential(nn.Conv2d(in_channels=first_params.in_feature,
                                                    out_channels=first_params.out_feature,
                                                    padding=first_params.padding,
                                                    stride=first_params.strides,
                                                    kernel_size=first_params.kernel_sizes),
                                          nn.BatchNorm2d(first_params.out_feature)
                                          )
        self.first_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=first_params.in_feature,
                      out_channels=first_params.out_feature,
                      padding=first_params.padding,
                      stride=first_params.strides,
                      kernel_size=first_params.kernel_sizes),
            nn.BatchNorm2d(first_params.out_feature)
        )

        self.first_activation = nn.GLU(dim=1)
        self.first_pools = [nn.MaxPool1d(kernel_size=first_params.pool_kernel_size,
                                         stride=first_params.pool_stride,
                                         padding=first_params.pool_padding),
                            nn.AvgPool1d(kernel_size=first_params.pool_kernel_size,
                                         stride=first_params.pool_stride,
                                         padding=first_params.pool_padding)
                            ]
        layers = []
        for i, block_params in enumerate(params.cnn_blocks[1:]):
            layers.append(CnnMultipleDropoutBlock(block_params))
        self.middle_layer = nn.Sequential(*layers)

        self.attention = ContextAttention(input_feature=params.cnn_blocks[-1].out_feature,
                                          seq_len=params.last_chunk_len,
                                          context_size=params.context_size)

        self.output_layer = nn.Linear(params.cnn_blocks[-1].out_feature, 1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = torch.cat([self.first_conv_1(x), self.first_conv_2(x)], dim=1)
        x = self.first_activation(x)
        x = x.squeeze(dim=-1)

        x = torch.cat([
            self.first_pools[0](x),
            self.first_pools[1](x)
        ], dim=1)
        x = self.middle_layer(x).transpose(-1, -2)
        # print(x.shape)
        x = self.attention(x).view(x.shape[0], -1)
        x = multiple_dropouts(x, n_dropouts=self.n_dropouts, dropout_rate=self.dropout_rate)
        return self.output_layer(x)


class BasicCnnAttentionModule(nn.Module):

    def __init__(self, params: BasicCnnAttentionParams):
        super().__init__()
        self.dropout_rate = params.last_dropout_rate
        self.last_n_dropouts = params.last_n_dropouts
        self.concat_pool = params.concat_pool
        # self.params = params
        first_params = params.cnn_blocks[0]
        self.first_conv_1 = nn.Sequential(
            # TODO enable switch
            nn.BatchNorm2d(first_params.in_feature),
            nn.Conv2d(in_channels=first_params.in_feature,
                      out_channels=first_params.out_feature,
                      padding=first_params.padding,
                      stride=first_params.strides,
                      kernel_size=first_params.kernel_sizes),
            nn.BatchNorm2d(first_params.out_feature),
            nn.LeakyReLU()
        )

        self.first_pools = [nn.MaxPool1d(kernel_size=first_params.pool_kernel_size,
                                         stride=first_params.pool_stride,
                                         padding=first_params.pool_padding),
                            nn.AvgPool1d(kernel_size=first_params.pool_kernel_size,
                                         stride=first_params.pool_stride,
                                         padding=first_params.pool_padding)
                            ]
        layers = []
        for i, block_params in enumerate(params.cnn_blocks[1:]):
            layers.append(CnnBasicBlock(block_params))
        self.middle_layer = nn.Sequential(*layers)

        self.attention = ContextAttention(input_feature=params.cnn_blocks[-1].out_feature,
                                          seq_len=params.last_chunk_len,
                                          context_size=params.context_size)

        self.output_layer = nn.Linear(params.cnn_blocks[-1].out_feature, 1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.first_conv_1(x)
        x = x.squeeze(dim=-1)
        # print(x)
        if self.concat_pool:
            x = torch.cat([
                self.first_pools[0](x),
                self.first_pools[1](x)
            ], dim=1)
        else:
            x = self.first_pools[0](x)
        x = self.middle_layer(x).transpose(-1, -2)
        # print(x.shape)
        x = self.attention(x).view(x.shape[0], -1)
        if self.last_n_dropouts > 1:
            x = multiple_dropouts(x, self.dropout_rate, self.last_n_dropouts)
        else:
            x = F.dropout(x, self.dropout_rate)
        return self.output_layer(x)
