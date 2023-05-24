#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
######## Important: In compliance with the Apache License, we note that this file has     ########
########            been derived and modified from the original nnUNet work.              ########
######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########


from copy import deepcopy
from nnunet.network_architecture.custom_modules.helperModules import Identity
from torch import nn, cat
import torch


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(ConvDropoutNormReLU, self).__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.conv = network_props['conv_op'](input_channels, output_channels, kernel_size,
                                             padding=[(i - 1) // 2 for i in kernel_size],
                                             **network_props['conv_op_kwargs'])

        # maybe dropout
        if network_props['dropout_op'] is not None:
            self.do = network_props['dropout_op'](**network_props['dropout_op_kwargs'])
        else:
            self.do = Identity()

        if network_props['norm_op'] is not None:
            self.norm = network_props['norm_op'](output_channels, **network_props['norm_op_kwargs'])
        else:
            self.norm = Identity()

        self.nonlin = network_props['nonlin'](**network_props['nonlin_kwargs'])

        self.all = nn.Sequential(self.conv, self.do, self.norm, self.nonlin)

    def forward(self, x):
        return self.all(x)


class StackedConvLayers(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_convs, first_stride=None):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(StackedConvLayers, self).__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.
        network_props_first = deepcopy(network_props)

        if first_stride is not None:
            network_props_first['conv_op_kwargs']['stride'] = first_stride

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(input_channels, output_channels, kernel_size, network_props_first),
            *[ConvDropoutNormReLU(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_convs - 1)]
        )

    def forward(self, x):
        return self.convs(x)


class BasicResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = stride
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.conv1 = props['conv_op'](in_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()

        self.conv2 = props['conv_op'](out_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        if (self.stride is not None and any((i != 1 for i in self.stride))) or (in_planes != out_planes):
            stride_here = stride if stride is not None else 1
            self.downsample_skip = nn.Sequential(props['conv_op'](in_planes, out_planes, 1, stride_here, bias=False),
                                                 props['norm_op'](out_planes, **props['norm_op_kwargs']))
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.dropout(self.conv1(x))
        out = self.nonlin1(self.norm1(out))

        out = self.norm2(self.conv2(out))

        residual = self.downsample_skip(residual)

        out += residual

        return self.nonlin2(out)


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
            raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = stride
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.conv1 = props['conv_op'](in_planes, self.bottleneck_planes, [1 for _ in kernel_size], padding=[0 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](self.bottleneck_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](self.bottleneck_planes, self.bottleneck_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](self.bottleneck_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv3 = props['conv_op'](self.bottleneck_planes, out_planes, [1 for _ in kernel_size], padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm3 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])

        if (self.stride is not None and any((i != 1 for i in self.stride))) or (in_planes != out_planes):
            stride_here = stride if stride is not None else 1
            self.downsample_skip = nn.Sequential(props['conv_op'](in_planes, out_planes, 1, stride_here, bias=False),
                                                 props['norm_op'](out_planes, **props['norm_op_kwargs']))
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.nonlin1(self.norm1(self.conv1(x)))
        out = self.nonlin2(self.norm2(self.conv2(out)))

        out = self.norm3(self.conv3(out))

        residual = self.downsample_skip(residual)

        out += residual

        return self.nonlin3(out)


class ResidualLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=BasicResidualBlock):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )
        
        #Start of Squeeze and Excited block
        num_channels = input_channels
        reduction_ratio = 2
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        #Squeeze and excited block
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))
        
        return self.convs(output_tensor)

        #return self.convs(x)


############### Defining Dense Layer/Block Here #####################


class BasicDenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = growth_rate
        self.in_planes = in_planes

        self.conv1 = props['conv_op'](in_planes, growth_rate, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm1 = props['norm_op'](growth_rate, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()

    def forward(self, x):

        out = self.dropout(self.conv1(x))
        out = self.nonlin1(self.norm1(out))

        return out


class DenseLayer(nn.Module):
    """
    Here the dense layer is defined in which we have multiple sub dense blocks contained (In the paper the term block and layer
    are swapped. Here however we are following the original nnUNet terminology convention.)
    """

    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None,
                 block=BasicDenseBlock, growth_rate=10):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.kernel_size = kernel_size
        network_props['conv_op_kwargs']['stride'] = 1

        if first_stride is not None:
            kwargs_conv1 = deepcopy(network_props['conv_op_kwargs'])
            kwargs_conv1['stride'] = first_stride
        else:
            kwargs_conv1 = network_props['conv_op_kwargs']

        self.convs = nn.ModuleList([block(input_channels + i*growth_rate, growth_rate, kernel_size, network_props)
                                    for i in range(num_blocks)])

        self.final_one_conv = nn.Sequential(network_props['conv_op'](input_channels + num_blocks * growth_rate,
                                                                     output_channels, kernel_size=1),
                                            network_props['norm_op'](output_channels, **network_props['norm_op_kwargs']),
                                            network_props['nonlin'](**network_props['nonlin_kwargs']))

        # this final conv will mean that the output is the desired number of channels
        self.final_pooling_conv = nn.Sequential(network_props['conv_op'](output_channels, output_channels, kernel_size,
                                                                         padding=[(i - 1) // 2 for i in kernel_size],
                                                                         **kwargs_conv1),
                                                network_props['norm_op'](output_channels, **network_props['norm_op_kwargs']),
                                                network_props['nonlin'](**network_props['nonlin_kwargs']))

    def forward(self, x):

        for block in self.convs:
            out = block(x)
            x = cat([x, out], 1)  # 1 = channel axis
        return self.final_pooling_conv(self.final_one_conv(x))


############### Defining Inception Layer/Block Here #####################


class BasicInceptionBlock(nn.Module):
    """ Here we define the inception block.
        This code is inspired from the adaptation of inception block from the authors at PyTorch.
        (https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py)
        """

    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        super().__init__()

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes

        self.branch1x1 = props['conv_op'](in_planes, int(out_planes/4), kernel_size=1)

        self.branch5x5_1 = props['conv_op'](in_planes, int(out_planes/8), kernel_size=1)
        self.branch5x5_2 = props['conv_op'](int(out_planes/8), int(out_planes/4), kernel_size=5, padding=2)

        self.branch3x3dbl_1 = props['conv_op'](in_planes, int(out_planes/8), kernel_size=1)
        self.branch3x3dbl_2 = props['conv_op'](int(out_planes/8), int(out_planes/4), kernel_size=3, padding=1)
        self.branch3x3dbl_3 = props['conv_op'](int(out_planes/4), int(out_planes/4), kernel_size=3, padding=1)

        self.branch_pool = props['conv_op'](in_planes, int(out_planes/4), kernel_size=1)

        self.avg_pool = props['avg_pool'](kernel_size=3, stride=1, padding=1)

        self.norm = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin = props['nonlin'](**props['nonlin_kwargs'])

        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()

    def forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        concat_out = cat(outputs, 1)  # this should have same x,y,z as input
        return self.nonlin(self.norm(concat_out))


class InceptionLayer(nn.Module):
    """
    Here the inception layer is defined in which we have multiple sub inception blocks contained (In the paper the term block and layer
    are swapped. Here however we are following the original nnUNet terminology convention.)
    """
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None,
                 block=BasicInceptionBlock):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.kernel_size = kernel_size
        network_props['conv_op_kwargs']['stride'] = 1

        if first_stride is not None:
            kwargs_conv1 = deepcopy(network_props['conv_op_kwargs'])
            kwargs_conv1['stride'] = first_stride
        else:
            kwargs_conv1 = network_props['conv_op_kwargs']

        self.convs = nn.Sequential(block(input_channels, output_channels, kernel_size, network_props),
                                   *[block(output_channels, output_channels, kernel_size,
                                           network_props) for _ in range(num_blocks - 1)])

        # this final conv will mean that the output is the desired number of channels
        self.final_pooling_conv = nn.Sequential(network_props['conv_op'](output_channels, output_channels, kernel_size,
                                                                         padding=[(i - 1) // 2 for i in kernel_size],
                                                                         **kwargs_conv1),
                                                network_props['norm_op'](output_channels, **network_props['norm_op_kwargs']),
                                                network_props['nonlin'](**network_props['nonlin_kwargs']))

    def forward(self, x):
        return self.final_pooling_conv(self.convs(x))


