import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import MLP, Permute
import math
import numpy as np
from functools import partial

from .fpn import FPN

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with 'pip install thop' to measure FLOPs.")





# Simplified 3D Residual Block
class ResidualBlockSimplified(nn.Module):
    """The simplified Basic Residual block of ResNet."""
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y += X
        return F.relu(Y)


# ResNet Bottleneck for 3D convolution
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


# ResNet_FPN
class ResNet_FPN_64(nn.Module):
    """ A smaller backbone for 64^3 inputs. """
    # block: the type of ResNet layer
    # layers: the depth of each size of layers, i.e. the num of layers before the next
    def __init__(self, block, layers, input_dim=4, use_fpn=True):
        super(ResNet_FPN_64, self).__init__()
        self.in_planes = 16
        self.out_channels = 64
        self.conv1 = nn.Conv3d(input_dim, 16, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  16, layers[0], stride=1)
        self.layer2 = self._make_layer(block,  32, layers[1], stride=2)
        self.layer3 = self._make_layer(block,  64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        # Top layer
        self.toplayer = nn.Conv3d(
            512, 64, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv3d(
            256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(
            128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(
            64, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, X, Y, Z = y.size()
        return F.interpolate(x, size=(X, Y, Z), mode='trilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        # c1 = F.max_pool3d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5


class ResNet_FPN_256(nn.Module):
    # block: the type of ResNet layer
    # layers: the depth of each size of layers, i.e. the num of layers before the next

    '''
    Args:
        layers: list of int. Its size could be variable. The length will be the ouput
                length. The value is the depth of layers at that level
        is_max_pool: If it is False, the network will not use downsample

    Returns (of self.forward function):
        A feature list. Its size is equal to the size of self.layers.
    '''

    def __init__(self, block, layers, input_dim=4, is_max_pool=False):
        super(ResNet_FPN_256, self).__init__()
        self.in_planes = 64
        self.out_channels = 256
        self.conv1 = nn.Conv3d(input_dim, self.in_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)

        # Bottom-up layers
        self.layers = nn.ModuleList()
        self.start_deep = self.in_planes
        self.is_max_pool = is_max_pool
        for i in range(len(layers)):
            self.layers.append(self._make_layer(block, self.start_deep * (2**i), layers[i],
                                                stride=1 if i == 0 else 2))

        # Smooth layers
        self.smooths = nn.ModuleList()
        for i in range(len(layers)-1):
            self.smooths.append(nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1))

        # Lateral layers
        self.latlayers = nn.ModuleList()
        for i in range(len(layers)-1, -1, -1):
            self.latlayers.append(
                nn.Conv3d(block.expansion * self.start_deep * (2**i), self.out_channels, 
                          kernel_size=1, stride=1, padding=0)
            )

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, X, Y, Z = y.size()
        return F.interpolate(x, size=(X, Y, Z), mode='nearest') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        if self.is_max_pool:
            c1 = F.max_pool3d(c1, kernel_size=3, stride=2, padding=1)
        c_out = [c1]
        for i in range(len(self.layers)):
            c_out.append(self.layers[i](c_out[i]))

        # Top-down
        p5 = self.latlayers[0](c_out[-1])
        p_out = [p5]
        for i in range(len(self.latlayers)-1):
            p_out.append(self._upsample_add(p_out[i], self.latlayers[i+1](c_out[-2-i])))

        # Smooth
        for i in range(len(self.smooths)):
            p_out[i+1] = self.smooths[i](p_out[i+1])

        p_out.reverse()
        return p_out


# Simplified ResNet (for debug)
class ResNetSimplified_64(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.residuals = nn.ModuleList()
        for i in range(num_residuals):
            self.residuals.append(ResidualBlockSimplified(out_channels))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        for i in range(len(self.residuals)):
            Y = self.residuals[i](Y)
        return (Y,)


class ResNetSimplified_256(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.residuals = nn.ModuleList()
        for i in range(num_residuals):
            self.residuals.append(ResidualBlockSimplified(out_channels))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.pool1(Y)
        for i in range(len(self.residuals)):
            Y = self.residuals[i](Y)
        return (Y,)


# VGG_FPN
vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    "AF": [64, 128, "F", 256, 256, "M", "F", 512, 512, "M", "F", 512, 512, "M", "F"],
    "DF":  [64, 64, 128, 128, "F", 256, 256, 256, "M", "F", 512, 512, 512, "M", "F", 512, 512, 512, "M", "F"],
    "EF": [64, 64, 128, 128, "F", 256, 256, 256, 256, "M", "F", 512, 512, 512, 512, "M", "F", 512, 512, 512, 512, "M", "F"],
}


class VGG_FPN(nn.Module):
    def __init__(self, cfg: str = "EF", in_channels: int = 4, batch_norm: bool = True, input_size: int = 256,
                 conv_at_start: bool=False):
        """ VGG-FPN backbone.
            Args:
                cfg (str): Config name of the VGG-FPN.
                in_channels (int): Number of input channels.
                batch_norm (bool): Use batch normalization.
                feature_size (int): The largest side length of input grid. If the input_size>=200, the network will downsmaple it. 
                conv_at_start (bool): Use conv layer at the start of the network before first downsampling.
        """
        super().__init__()
        self.out_channels = 256
        self.input_size = input_size
        in_channels = 1
        _in_channels = in_channels if not conv_at_start else 32
        self.layers = self.make_layers(vgg_cfgs[cfg], _in_channels, batch_norm, input_size)
        self.fpn_neck = FPN([128, 256, 512, 512], self.out_channels, 4)

        self.conv_at_start = conv_at_start
        self.starting_layers = None
        self.ds_layers = None
        if self.conv_at_start:
            self.starting_layers = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

            self.ds_layers = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
    
    def make_layers(self, cfg: List[Union[str, int]], in_channels, batch_norm, input_size) -> nn.Sequential:
        layers: List[nn.Module] = []
        curr_layer: List[nn.Module] = []
        _in_channels = in_channels
        if input_size >= 160:
            layers += [nn.Conv2d(_in_channels, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), 
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            layers += [nn.Conv2d(_in_channels, 64, kernel_size=7, stride=1, padding=3),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
        _in_channels = 64
        for v in cfg:
            if v == "M":
                curr_layer += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            elif v == "F":
                layers += [nn.Sequential(*curr_layer)]
                curr_layer = []
            else:
                v = cast(int, v)
                conv3d = nn.Conv2d(_in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    curr_layer += [conv3d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    curr_layer += [conv3d, nn.ReLU(inplace=True)]
                _in_channels = v

        return nn.Sequential(*layers)
    
    def forward(self, X):
        features = []

        X_ds = None
        if self.conv_at_start:
            X = self.starting_layers(X)
            X_ds = self.ds_layers(X)

        for i in range(len(self.layers)):
            X = self.layers[i](X)
            features.append(X)

        if self.conv_at_start:
            features[-4] = features[-4] + X_ds

        return self.fpn_neck(features[-4:])


class Conv2DBase_model(nn.Module):
    """
    2D Base model - Only 2D projection without 3D fusion
    Projects 3D volumes onto 2D planes along each axis and applies 2D convolutions
    """
    def __init__(self, config='EF', in_channels=4, batch_norm=True, input_size=256, pe_type=None):
        """
        Args:
            config: VGG_FPN 설정 (예: 'EF')
            in_channels: 입력 채널 수 (원래는 4)
            batch_norm: 배치 정규화 사용 여부
            input_size: x, y, z의 크기 (160 또는 200로 가정)
            pe_type: 포지셔널 인코딩 유형 ('transformer' 또는 None/'none')
        """
        super(Conv2DBase_model, self).__init__()
        self.input_size = input_size
        self.out_channels = 256
        self.pe_type = pe_type
        
        # FLOPs 측정을 위한 플래그 (한 번만 측정)
        self._flops_measured = False

        # 2D feature extractor for each axis branch
        self.model_x = VGG_FPN(config, in_channels, batch_norm, input_size)
        self.model_y = VGG_FPN(config, in_channels, batch_norm, input_size)
        self.model_z = VGG_FPN(config, in_channels, batch_norm, input_size)

        # Scale lengths for FPN (2D feature size에 맞춤)
        # VGG_FPN은 입력 크기를 4로 나눈 크기의 feature를 생성
        if input_size == 160:
            self.scale_lengths = [40, 20, 10, 5]
        elif input_size == 200:
            self.scale_lengths = [50, 25, 13, 7]
        else:
            base_2d_size = input_size // 4
            self.scale_lengths = [base_2d_size // (2**i) for i in range(4)]

        # PE 설정 (classification.py와 유사)
        if self.pe_type == 'transformer':
            self._create_transformer_pe_layers()
            self._create_transformer_pe_after_layers()

    def _create_transformer_pe_layers(self):
        """Create transformer-based positional encoding layers (same as classification.py)"""
        self.pe_transformer = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True),
                num_layers=2
            ),
            nn.Linear(64, self.input_size),
            nn.Sigmoid()
        )

    def _create_transformer_pe_after_layers(self):
        """Create transformer-based positional encoding layers for after reconstruction"""
        # 모든 스케일에서 동일한 input_size 크기 input 사용
        self.pe_after_transformers = nn.ModuleDict()
        
        for i, scale_len in enumerate(self.scale_lengths):
            transformer_name = f'pe_after_{i}'
            
            # 모든 transformer가 input_size 크기 input을 받도록 수정
            self.pe_after_transformers[transformer_name] = nn.Sequential(
                nn.Linear(self.input_size, 64),  # Project input_size input to transformer dimension
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True),
                    num_layers=2
                ),
                nn.Linear(64, self.input_size),  # Project back to input_size dimension
                nn.Sigmoid()
            )

    def _apply_pe_before_projection(self, X):
        """Apply positional encoding before 3D → 2D projection"""
        B, C, D, H, W = X.shape
        device = X.device
        
        # If no PE type specified, return original input
        if self.pe_type is None or self.pe_type == 'none':
            return X, X, X

        # Transformer PE 활성화
        if self.pe_type == 'transformer':
            # Transformer PE (same as classification.py)
            X_flat = X.squeeze(1)  # [B, D, H, W]

            # Extract features for each axis (same as classification.py)
            x_vec = X_flat.sum(dim=(2, 3))  # [B, D] - sum over H, W
            y_vec = X_flat.sum(dim=(1, 3))  # [B, H] - sum over D, W
            z_vec = X_flat.sum(dim=(1, 2))  # [B, W] - sum over D, H

            # Stack for transformer input [B, 3, input_size]
            transformer_input = torch.stack([x_vec, y_vec, z_vec], dim=1)

            # 두 번째 PE에서 사용할 수 있도록 캐시
            self._cached_original_transformer_input = transformer_input.clone()

            transformer_weights = self.pe_transformer(transformer_input)  # [B, 3, input_size]

            # Extract weights for each axis
            weight_x = transformer_weights[:, 0, :D]  # [B, D]
            weight_y = transformer_weights[:, 1, :H]  # [B, H]
            weight_z = transformer_weights[:, 2, :W]  # [B, W]

            # Apply weights to input
            X_x = X * (1.0 + weight_x.view(B, 1, D, 1, 1).expand_as(X) * 0.1)
            X_y = X * (1.0 + weight_y.view(B, 1, 1, H, 1).expand_as(X) * 0.1)
            X_z = X * (1.0 + weight_z.view(B, 1, 1, 1, W).expand_as(X) * 0.1)

            return X_x, X_y, X_z

        return X, X, X

    def _apply_pe_after_reconstruction(self, x, axis, scale_idx):
        """Apply positional encoding after 2D → 3D reconstruction"""

        # If no PE type specified, return original input
        if self.pe_type is None or self.pe_type == 'none':
            return x

        if self.pe_type == 'transformer':
            # Transformer PE (data-driven like classification.py)
            
            # Cache transformer results for efficiency
            if not hasattr(self, '_cached_after_transformer_weights') or \
               self._cached_after_transformer_weights is None or \
               self._cached_after_transformer_weights.get(f'scale_{scale_idx}') is None:
                
                transformer_name = f'pe_after_{scale_idx}'
                if transformer_name not in self.pe_after_transformers:
                    return x
                
                transformer = self.pe_after_transformers[transformer_name]
                
                # Extract features for all 3 axes (like classification.py)
                x_flat = x.squeeze(1) if x.dim() == 5 else x  # Remove channel dimension if present
                
                # Get current scale length
                scale_len = self.scale_lengths[scale_idx]
                
                # 첫 번째 PE에서 캐시된 원본 transformer input 사용
                if hasattr(self, '_cached_original_transformer_input') and self._cached_original_transformer_input is not None:
                    transformer_input = self._cached_original_transformer_input
                else:
                    # 폴백: 현재 스케일 데이터에서 생성 (원래 방식)
                    batch_size = x.shape[0]
                    
                    if len(x_flat.shape) == 4:  # [B, D, H, W]
                        x_vec_orig = x_flat.sum(dim=(2, 3))  # [B, D]
                        y_vec_orig = x_flat.sum(dim=(1, 3))  # [B, H]
                        z_vec_orig = x_flat.sum(dim=(1, 2))  # [B, W]
                        
                    elif len(x_flat.shape) == 5:  # [B, C, D, H, W]
                        x_vec_orig = x_flat.sum(dim=(1, 3, 4))  # [B, D]
                        y_vec_orig = x_flat.sum(dim=(1, 2, 4))  # [B, H]
                        z_vec_orig = x_flat.sum(dim=(1, 2, 3))  # [B, W]
                        
                    else:  # [B, H, W] or other shapes
                        x_vec_orig = x_flat.sum(dim=tuple(range(1, x_flat.dim()-1)))
                        y_vec_orig = x_vec_orig
                        z_vec_orig = x_vec_orig
                    
                    # input_size 크기로 맞춤 (패딩 또는 반복)
                    x_vec = torch.zeros(batch_size, self.input_size, device=x.device)
                    y_vec = torch.zeros(batch_size, self.input_size, device=x.device)
                    z_vec = torch.zeros(batch_size, self.input_size, device=x.device)
                    
                    orig_size = x_vec_orig.shape[1]
                    if orig_size <= self.input_size:
                        x_vec[:, :orig_size] = x_vec_orig
                        y_vec[:, :orig_size] = y_vec_orig
                        z_vec[:, :orig_size] = z_vec_orig
                        
                        if orig_size < self.input_size:
                            x_vec[:, orig_size:] = x_vec_orig[:, -1:].expand(-1, self.input_size - orig_size)
                            y_vec[:, orig_size:] = y_vec_orig[:, -1:].expand(-1, self.input_size - orig_size)
                            z_vec[:, orig_size:] = z_vec_orig[:, -1:].expand(-1, self.input_size - orig_size)
                    else:
                        x_vec = x_vec_orig[:, :self.input_size]
                        y_vec = y_vec_orig[:, :self.input_size]
                        z_vec = z_vec_orig[:, :self.input_size]
                    
                    transformer_input = torch.stack([x_vec, y_vec, z_vec], dim=1)
                
                transformer_weights = transformer(transformer_input)  # [B, 3, input_size]
                
                # 현재 스케일에 맞게 자르기
                transformer_weights = transformer_weights[:, :, :scale_len]  # [B, 3, scale_len]
                
                # Cache results
                if not hasattr(self, '_cached_after_transformer_weights'):
                    self._cached_after_transformer_weights = {}
                self._cached_after_transformer_weights[f'scale_{scale_idx}'] = transformer_weights
            else:
                transformer_weights = self._cached_after_transformer_weights[f'scale_{scale_idx}']
            
            # Apply axis-specific weights
            if axis == 'x':
                axis_idx = 0
            elif axis == 'y':
                axis_idx = 1
            else:  # z
                axis_idx = 2
            
            pos_weights = transformer_weights[:, axis_idx, :]  # [B, scale_len]
            
            # Reshape and expand to match x dimensions
            if axis == 'x':
                pos_weights = pos_weights.view(pos_weights.shape[0], 1, -1, 1, 1).expand_as(x)
            elif axis == 'y':
                pos_weights = pos_weights.view(pos_weights.shape[0], 1, 1, -1, 1).expand_as(x)
            else:
                pos_weights = pos_weights.view(pos_weights.shape[0], 1, 1, 1, -1).expand_as(x)
            
            return x * pos_weights
        
        return x

    def _measure_flops_once(self, X):
        """한 번만 실행하여 FLOPs를 측정하고 결과를 출력하는 메서드"""
        if not THOP_AVAILABLE:
            return
        
        # Wrapper 클래스 정의
        class FunctionWrapper(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func
            
            def forward(self, *args, **kwargs):
                return self.func(*args, **kwargs)
        
        # FLOPs 측정을 위한 변수 초기화
        total_flops = 0
        step_flops = {}
        
        B, C, D, H, W = X.shape
        
        try:
            # 1. 첫번째 PE 적용 측정
            pe_wrapper = FunctionWrapper(self._apply_pe_before_projection)
            flops, _ = profile(pe_wrapper, inputs=(X,), verbose=False)
            step_flops['pe_before_projection'] = flops
            total_flops += flops
            
            # 2. 3D → 2D 프로젝션 측정을 위한 임시 함수
            def projection_func(x_x, x_y, x_z):
                proj_x = torch.mean(x_x, dim=2)  # [B, C, H, W]
                proj_y = torch.mean(x_y, dim=3)  # [B, C, D, W]
                proj_z = torch.mean(x_z, dim=4)  # [B, C, D, H]
                return proj_x, proj_y, proj_z
            
            X_x, X_y, X_z = self._apply_pe_before_projection(X)
            proj_wrapper = FunctionWrapper(projection_func)
            flops, _ = profile(proj_wrapper, inputs=(X_x, X_y, X_z), verbose=False)
            step_flops['3d_to_2d_projection'] = flops
            total_flops += flops
            
            # 3. 각 축별 2D feature extraction 측정
            proj_x = torch.mean(X_x, dim=2)
            proj_y = torch.mean(X_y, dim=3)
            proj_z = torch.mean(X_z, dim=4)
            
            flops_x, _ = profile(self.model_x, inputs=(proj_x,), verbose=False)
            step_flops['feature_extraction_x'] = flops_x
            total_flops += flops_x
            
            flops_y, _ = profile(self.model_y, inputs=(proj_y,), verbose=False)
            step_flops['feature_extraction_y'] = flops_y
            total_flops += flops_y
            
            flops_z, _ = profile(self.model_z, inputs=(proj_z,), verbose=False)
            step_flops['feature_extraction_z'] = flops_z
            total_flops += flops_z
            
            # 4. 2D → 3D 복원 측정을 위한 임시 함수
            def reconstruction_func(features_x, features_y, features_z):
                combined_features = []
                
                for i, (feat_x, feat_y, feat_z) in enumerate(zip(features_x, features_y, features_z)):
                    # Reconstruction: repeat along missing dimensions
                    out_x_rep = feat_x.unsqueeze(2).repeat(1, 1, self.scale_lengths[i], 1, 1)
                    out_y_rep = feat_y.unsqueeze(3).repeat(1, 1, 1, self.scale_lengths[i], 1)
                    out_z_rep = feat_z.unsqueeze(4).repeat(1, 1, 1, 1, self.scale_lengths[i])
                    
                    # Apply second PE
                    if self.pe_type is not None:
                        out_x_final = self._apply_pe_after_reconstruction(out_x_rep, 'x', i)
                        out_y_final = self._apply_pe_after_reconstruction(out_y_rep, 'y', i)
                        out_z_final = self._apply_pe_after_reconstruction(out_z_rep, 'z', i)
                    else:
                        out_x_final = out_x_rep
                        out_y_final = out_y_rep
                        out_z_final = out_z_rep
                    
                    # Average the three reconstructions
                    combined = (out_x_final + out_y_final + out_z_final) / 3.0
                    combined_features.append(combined)
                
                return combined_features
            
            features_x = self.model_x(proj_x)
            features_y = self.model_y(proj_y)
            features_z = self.model_z(proj_z)
            
            recon_wrapper = FunctionWrapper(reconstruction_func)
            flops, _ = profile(recon_wrapper, inputs=(features_x, features_y, features_z), verbose=False)
            step_flops['2d_to_3d_reconstruction'] = flops
            total_flops += flops
            
        except Exception as e:
            print(f"FLOPs 측정 중 오류 발생: {e}")
            return
        
        # 결과 출력
        print(f"\n=== Backbone FLOPs (Conv2DBase) ===")
        print(f"Total Backbone FLOPs: {total_flops/1e9:.2f} G")
        print(f"==================================\n")

    def forward(self, X):
        # X: [B, 1, D, H, W]
        B, C, D, H, W = X.shape
        device = X.device

        # Clear transformer cache for after reconstruction
        if hasattr(self, '_cached_after_transformer_weights'):
            self._cached_after_transformer_weights = {}
        
        # Clear cached original transformer input
        if hasattr(self, '_cached_original_transformer_input'):
            self._cached_original_transformer_input = None

        # FLOPs 측정 (한 번만 실행, training 모드에서만)
        if THOP_AVAILABLE and not self._flops_measured and self.training:
            self._measure_flops_once(X)
            self._flops_measured = True

        # 1. 첫번째 PE 적용 (3D → 2D 전)
        X_x, X_y, X_z = self._apply_pe_before_projection(X)

        # 2. 3D → 2D projection
        # X-axis projection: average over depth (dim=2)
        proj_x = torch.mean(X_x, dim=2)  # [B, C, H, W]
        
        # Y-axis projection: average over height (dim=3)  
        proj_y = torch.mean(X_y, dim=3)  # [B, C, D, W]
        
        # Z-axis projection: average over width (dim=4)
        proj_z = torch.mean(X_z, dim=4)  # [B, C, D, H]

        # 3. 2D CNN 처리 (각 축별로)
        features_x = self.model_x(proj_x)  # FPN features for x-axis
        features_y = self.model_y(proj_y)  # FPN features for y-axis
        features_z = self.model_z(proj_z)  # FPN features for z-axis

        # 4. 2D → 3D reconstruction with PE
        combined_features = []
        
        for i, (feat_x, feat_y, feat_z) in enumerate(zip(features_x, features_y, features_z)):
            # Reconstruction: repeat along missing dimensions
            out_x_rep = feat_x.unsqueeze(2).repeat(1, 1, self.scale_lengths[i], 1, 1)
            out_y_rep = feat_y.unsqueeze(3).repeat(1, 1, 1, self.scale_lengths[i], 1)
            out_z_rep = feat_z.unsqueeze(4).repeat(1, 1, 1, 1, self.scale_lengths[i])
            
            # Apply second PE
            if self.pe_type is not None:
                out_x_final = self._apply_pe_after_reconstruction(out_x_rep, 'x', i)
                out_y_final = self._apply_pe_after_reconstruction(out_y_rep, 'y', i)
                out_z_final = self._apply_pe_after_reconstruction(out_z_rep, 'z', i)
            else:
                out_x_final = out_x_rep
                out_y_final = out_y_rep
                out_z_final = out_z_rep
            
            # Average the three reconstructions
            combined = (out_x_final + out_y_final + out_z_final) / 3.0
            combined_features.append(combined)

        return combined_features


class Total_model(nn.Module):
    def __init__(self, config='EF', in_channels=4, batch_norm=True, input_size=256, pe_type=None, ratio_3d=0.5):
        """
        Args:
            config: VGG_FPN 설정 (예: 'EF')
            in_channels: 입력 채널 수 (원래는 4)
            batch_norm: 배치 정규화 사용 여부
            input_size: x, y, z의 크기 (160 또는 200로 가정)
            pe_type: 포지셔널 인코딩 유형 ('transformer' 또는 None/'none')
            ratio_3d: 3D 해상도 비율 (0.5 = 1/2, 0.25 = 1/4)
        """
        super(Total_model, self).__init__()
        self.input_size = input_size
        self.out_channels = 256
        self.pe_type = pe_type
        self.ratio_3d = ratio_3d
        
        # FLOPs 측정을 위한 플래그 (한 번만 측정)
        self._flops_measured = False

        # 2D feature extractor for each axis branch
        self.model_x = VGG_FPN(config, in_channels, batch_norm, input_size)
        self.model_y = VGG_FPN(config, in_channels, batch_norm, input_size)
        self.model_z = VGG_FPN(config, in_channels, batch_norm, input_size)

        self.conv_at_start = False
        self.starting_layers = None
        self.ds_layers = None

        # Low-resolution branch: 연속적인 3D conv 블록 (피라미드)
        # ratio_3d에 따라 layer 수 조절
        self.lowres_conv1 = nn.Sequential(
            nn.Conv3d(1, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.lowres_conv2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.lowres_conv3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.lowres_conv4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # 1/2 해상도일 때 추가 layer (5개 layer)
        if ratio_3d == 0.5:
            self.lowres_conv5 = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True)
            )
        
        # Initialize positional encoding based on pe_type
        if self.pe_type is not None and self.pe_type != 'none':
            if self.pe_type == 'transformer':
                self._create_transformer_pe_layers()

        # 스케일별 길이 설정 (2D feature size에 맞춤)
        # VGG_FPN은 입력 크기를 4로 나눈 크기의 feature를 생성
        if input_size == 160:
            self.scale_lengths = [40, 20, 10, 5]
        elif input_size == 200:
            self.scale_lengths = [50, 25, 13, 7]
        else:
            base_2d_size = input_size // 4
            self.scale_lengths = [base_2d_size // (2**i) for i in range(4)]
        
        # Initialize second positional encoding for 2D → 3D reconstruction
        if self.pe_type is not None and self.pe_type != 'none':
            if self.pe_type == 'transformer':
                self._create_transformer_pe_after_layers()
        
        # 개별 branch 정규화를 위한 1x1x1 fusion layer (각 스케일별)
        self.fuse_conv_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=1, bias=False),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # 결합 후 전체 feature 재정규화를 위한 fusion layer
        self.fuse_conv_final = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
    def _create_transformer_pe_layers(self):
        """Create transformer-based positional encoding layers (same as classification.py)"""
        # Single transformer that processes all 3 axes like classification.py
        self.pe_transformer = nn.Sequential(
            nn.Linear(self.input_size, 64),  # Project input features to transformer dimension
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True),
                num_layers=2
            ),
            nn.Linear(64, self.input_size),  # Project back to original dimension
            nn.Sigmoid()
        )
        
    def _create_transformer_pe_after_layers(self):
        """Create transformer-based positional encoding layers for after reconstruction"""
        # Create transformer for each scale (processes all 3 axes like classification.py)
        # 모든 스케일에서 동일한 input_size 크기 input 사용
        self.pe_after_transformers = nn.ModuleDict()
        
        for i, scale_len in enumerate(self.scale_lengths):
            transformer_name = f'pe_after_{i}'
            
            # Create transformer that processes all 3 axes at once (like classification.py)
            # 모든 transformer가 input_size 크기 input을 받도록 수정
            self.pe_after_transformers[transformer_name] = nn.Sequential(
                nn.Linear(self.input_size, 64),  # Project input_size input to transformer dimension
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True),
                    num_layers=2
                ),
                nn.Linear(64, self.input_size),  # Project back to input_size dimension
                nn.Sigmoid()
            )

    
    def _apply_pe_before_projection(self, X):
        """Apply positional encoding before 3D → 2D projection"""
        B, C, D, H, W = X.shape
        device = X.device
        
        # If no PE type specified, return original input
        if self.pe_type is None or self.pe_type == 'none':
            return X, X, X
        
        # Transformer PE 활성화
        if self.pe_type == 'transformer':
            # Transformer PE (same as classification.py)
            X_flat = X.squeeze(1)  # [B, D, H, W]
            
            # Extract features for each axis (same as classification.py)
            x_vec = X_flat.sum(dim=(2, 3))  # [B, D] - sum over H, W
            y_vec = X_flat.sum(dim=(1, 3))  # [B, H] - sum over D, W  
            z_vec = X_flat.sum(dim=(1, 2))  # [B, W] - sum over D, H
            
            # Stack for transformer input [B, 3, input_size]
            transformer_input = torch.stack([x_vec, y_vec, z_vec], dim=1)
            
            # 두 번째 PE에서 사용할 수 있도록 캐시
            self._cached_original_transformer_input = transformer_input.clone()
            
            transformer_weights = self.pe_transformer(transformer_input)  # [B, 3, input_size]
            
            # Extract weights for each axis
            weight_x = transformer_weights[:, 0, :D]  # [B, D]
            weight_y = transformer_weights[:, 1, :H]  # [B, H]
            weight_z = transformer_weights[:, 2, :W]  # [B, W]
            
            # Apply weights to input 
            X_x = X * (1.0 + weight_x.view(B, 1, D, 1, 1).expand_as(X) * 0.1)
            X_y = X * (1.0 + weight_y.view(B, 1, 1, H, 1).expand_as(X) * 0.1)
            X_z = X * (1.0 + weight_z.view(B, 1, 1, 1, W).expand_as(X) * 0.1)
            
            return X_x, X_y, X_z
        
        return X, X, X

    def _apply_pe_after_reconstruction(self, x, axis, scale_idx):
        """Apply positional encoding after 2D → 3D reconstruction"""

        # If no PE type specified, return original input
        if self.pe_type is None or self.pe_type == 'none':
            return x

        
        if self.pe_type == 'transformer':
            # Transformer PE (data-driven like classification.py)
            
            # Cache transformer results for efficiency
            if not hasattr(self, '_cached_after_transformer_weights') or \
               self._cached_after_transformer_weights is None or \
               self._cached_after_transformer_weights.get(f'scale_{scale_idx}') is None:
                
                transformer_name = f'pe_after_{scale_idx}'
                if transformer_name not in self.pe_after_transformers:
                    return x
                
                transformer = self.pe_after_transformers[transformer_name]
                
                # Extract features for all 3 axes (like classification.py)
                x_flat = x.squeeze(1) if x.dim() == 5 else x  # Remove channel dimension if present
                
                # Get current scale length
                scale_len = self.scale_lengths[scale_idx]
                
                # 첫 번째 PE에서 캐시된 원본 transformer input 사용
                if hasattr(self, '_cached_original_transformer_input') and self._cached_original_transformer_input is not None:
                    transformer_input = self._cached_original_transformer_input
                else:
                    # 폴백: 현재 스케일 데이터에서 생성 (원래 방식)
                    batch_size = x.shape[0]
                    
                    if len(x_flat.shape) == 4:  # [B, D, H, W]
                        x_vec_orig = x_flat.sum(dim=(2, 3))  # [B, D]
                        y_vec_orig = x_flat.sum(dim=(1, 3))  # [B, H]
                        z_vec_orig = x_flat.sum(dim=(1, 2))  # [B, W]
                        
                    elif len(x_flat.shape) == 5:  # [B, C, D, H, W]
                        x_vec_orig = x_flat.sum(dim=(1, 3, 4))  # [B, D]
                        y_vec_orig = x_flat.sum(dim=(1, 2, 4))  # [B, H]
                        z_vec_orig = x_flat.sum(dim=(1, 2, 3))  # [B, W]
                        
                    else:  # [B, H, W] or other shapes
                        x_vec_orig = x_flat.sum(dim=tuple(range(1, x_flat.dim()-1)))
                        y_vec_orig = x_vec_orig
                        z_vec_orig = x_vec_orig
                    
                    # input_size 크기로 맞춤 (패딩 또는 반복)
                    x_vec = torch.zeros(batch_size, self.input_size, device=x.device)
                    y_vec = torch.zeros(batch_size, self.input_size, device=x.device)
                    z_vec = torch.zeros(batch_size, self.input_size, device=x.device)
                    
                    orig_size = x_vec_orig.shape[1]
                    if orig_size <= self.input_size:
                        x_vec[:, :orig_size] = x_vec_orig
                        y_vec[:, :orig_size] = y_vec_orig
                        z_vec[:, :orig_size] = z_vec_orig
                        
                        if orig_size < self.input_size:
                            x_vec[:, orig_size:] = x_vec_orig[:, -1:].expand(-1, self.input_size - orig_size)
                            y_vec[:, orig_size:] = y_vec_orig[:, -1:].expand(-1, self.input_size - orig_size)
                            z_vec[:, orig_size:] = z_vec_orig[:, -1:].expand(-1, self.input_size - orig_size)
                    else:
                        x_vec = x_vec_orig[:, :self.input_size]
                        y_vec = y_vec_orig[:, :self.input_size]
                        z_vec = z_vec_orig[:, :self.input_size]
                    
                    transformer_input = torch.stack([x_vec, y_vec, z_vec], dim=1)
                
                transformer_weights = transformer(transformer_input)  # [B, 3, input_size]
                
                # 현재 스케일에 맞게 자르기
                transformer_weights = transformer_weights[:, :, :scale_len]  # [B, 3, scale_len]
                
                # Cache results
                if not hasattr(self, '_cached_after_transformer_weights'):
                    self._cached_after_transformer_weights = {}
                self._cached_after_transformer_weights[f'scale_{scale_idx}'] = transformer_weights
            else:
                transformer_weights = self._cached_after_transformer_weights[f'scale_{scale_idx}']
            
            # Extract weight for the current axis
            if axis == 'x':
                axis_idx = 0
            elif axis == 'y':
                axis_idx = 1
            else:  # z
                axis_idx = 2
            
            pos_weights = transformer_weights[:, axis_idx, :]  # [B, scale_len]
            
            # Reshape and expand to match x dimensions
            if axis == 'x':
                pos_weights = pos_weights.view(pos_weights.shape[0], 1, -1, 1, 1).expand_as(x)
            elif axis == 'y':
                pos_weights = pos_weights.view(pos_weights.shape[0], 1, 1, -1, 1).expand_as(x)
            else:
                pos_weights = pos_weights.view(pos_weights.shape[0], 1, 1, 1, -1).expand_as(x)
            
            return x * pos_weights
        
        return x

    def _measure_flops_once(self, X):
        """한 번만 실행하여 FLOPs를 측정하고 결과를 출력하는 메서드"""
        if not THOP_AVAILABLE:
            return
        
        # Wrapper 클래스 정의
        class FunctionWrapper(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func
            
            def forward(self, *args, **kwargs):
                return self.func(*args, **kwargs)
        
        # FLOPs 측정을 위한 변수 초기화
        total_flops = 0
        step_flops = {}
        
        B, C, D, H, W = X.shape
        input_size = X.shape[-3]
        base_size = int(input_size * self.ratio_3d)
        
        try:
            # 1. 첫번째 PE 적용 측정
            pe_wrapper = FunctionWrapper(self._apply_pe_before_projection)
            flops, _ = profile(pe_wrapper, inputs=(X,), verbose=False)
            step_flops['pe_before_projection'] = flops
            total_flops += flops
            
            # 2. 3D → 2D 프로젝션 측정을 위한 임시 함수
            def projection_func(x_x, x_y, x_z):
                x_mean_x = x_x.mean(dim=-3)  # D 축 평균
                x_mean_y = x_y.mean(dim=-2)  # H 축 평균  
                x_mean_z = x_z.mean(dim=-1)  # W 축 평균
                return x_mean_x, x_mean_y, x_mean_z
            
            X_x, X_y, X_z = self._apply_pe_before_projection(X)
            proj_wrapper = FunctionWrapper(projection_func)
            flops, _ = profile(proj_wrapper, inputs=(X_x, X_y, X_z), verbose=False)
            step_flops['3d_to_2d_projection'] = flops
            total_flops += flops
            
            # 3. 각 축별 2D feature extraction 측정
            X_mean_x = X_x.mean(dim=-3)
            X_mean_y = X_y.mean(dim=-2)
            X_mean_z = X_z.mean(dim=-1)
            
            flops_x, _ = profile(self.model_x, inputs=(X_mean_x,), verbose=False)
            step_flops['feature_extraction_x'] = flops_x
            total_flops += flops_x
            
            flops_y, _ = profile(self.model_y, inputs=(X_mean_y,), verbose=False)
            step_flops['feature_extraction_y'] = flops_y
            total_flops += flops_y
            
            flops_z, _ = profile(self.model_z, inputs=(X_mean_z,), verbose=False)
            step_flops['feature_extraction_z'] = flops_z
            total_flops += flops_z
            
            # 4. 2D → 3D 복원 측정을 위한 임시 함수
            def reconstruction_func(outputs_x, outputs_y, outputs_z):
                final_outputs = []
                for i, (out_x, out_y, out_z) in enumerate(zip(outputs_x, outputs_y, outputs_z)):
                    scale_length = self.scale_lengths[i]
                    
                    out_x_rep = out_x.unsqueeze(2).repeat(1, 1, scale_length, 1, 1)
                    out_x_final = self._apply_pe_after_reconstruction(out_x_rep, 'x', i)
                    
                    out_y_rep = out_y.unsqueeze(3).repeat(1, 1, 1, scale_length, 1)
                    out_y_final = self._apply_pe_after_reconstruction(out_y_rep, 'y', i)
                    
                    out_z_rep = out_z.unsqueeze(4).repeat(1, 1, 1, 1, scale_length)
                    out_z_final = self._apply_pe_after_reconstruction(out_z_rep, 'z', i)
                    
                    combined = (out_x_final + out_y_final + out_z_final) / 3.0
                    final_outputs.append(combined)
                return final_outputs
            
            outputs_x = self.model_x(X_mean_x)
            outputs_y = self.model_y(X_mean_y)
            outputs_z = self.model_z(X_mean_z)
            
            recon_wrapper = FunctionWrapper(reconstruction_func)
            flops, _ = profile(recon_wrapper, inputs=(outputs_x, outputs_y, outputs_z), verbose=False)
            step_flops['2d_to_3d_reconstruction'] = flops
            total_flops += flops
            
            # 5. Branch normalization 측정
            def branch_norm_func(final_outputs):
                for i in range(len(final_outputs)):
                    final_outputs[i] = self.fuse_conv_branch[i](final_outputs[i])
                return final_outputs
            
            final_outputs = reconstruction_func(outputs_x, outputs_y, outputs_z)
            branch_norm_wrapper = FunctionWrapper(branch_norm_func)
            flops, _ = profile(branch_norm_wrapper, inputs=(final_outputs,), verbose=False)
            step_flops['branch_normalization'] = flops
            total_flops += flops
            
            # 6. Low-resolution branch 측정 (개별 단계별로)
            # 6-1. Interpolation 측정
            def interpolate_func(x, target_size=base_size):
                return F.interpolate(x, size=(target_size, target_size, target_size), mode='trilinear', align_corners=False)
            
            interpolate_wrapper = FunctionWrapper(lambda x: interpolate_func(x, base_size))
            flops_interp, _ = profile(interpolate_wrapper, inputs=(X,), verbose=False)
            step_flops['lowres_interpolate'] = flops_interp
            total_flops += flops_interp
            
            # 6-2. 각 conv layer 개별 측정
            lowres_base = F.interpolate(X, size=(base_size, base_size, base_size), mode='trilinear', align_corners=False)
            
            flops_conv1, _ = profile(self.lowres_conv1, inputs=(lowres_base,), verbose=False)
            step_flops['lowres_conv1'] = flops_conv1
            total_flops += flops_conv1
            
            feat1 = self.lowres_conv1(lowres_base)
            flops_conv2, _ = profile(self.lowres_conv2, inputs=(feat1,), verbose=False)
            step_flops['lowres_conv2'] = flops_conv2
            total_flops += flops_conv2
            
            feat2 = self.lowres_conv2(feat1)
            flops_conv3, _ = profile(self.lowres_conv3, inputs=(feat2,), verbose=False)
            step_flops['lowres_conv3'] = flops_conv3
            total_flops += flops_conv3
            
            feat3 = self.lowres_conv3(feat2)
            flops_conv4, _ = profile(self.lowres_conv4, inputs=(feat3,), verbose=False)
            step_flops['lowres_conv4'] = flops_conv4
            total_flops += flops_conv4
            
            feat4 = self.lowres_conv4(feat3)
            if self.ratio_3d == 0.5:
                flops_conv5, _ = profile(self.lowres_conv5, inputs=(feat4,), verbose=False)
                step_flops['lowres_conv5'] = flops_conv5
                total_flops += flops_conv5
                feat5 = self.lowres_conv5(feat4)
                feats = (feat1, feat2, feat3, feat4, feat5)
            else:
                feats = (feat1, feat2, feat3, feat4)
            
            # 전체 lowres_branch FLOPs 합계
            lowres_total = sum([step_flops.get(f'lowres_{k}', 0) for k in ['interpolate', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']])
            step_flops['lowres_branch'] = lowres_total
            
            # 7. Resize and combine 측정
            def resize_and_combine_func(feat1, feat2, feat3, feat4, final_outputs):
                feat1_resized = F.interpolate(feat1, size=(self.scale_lengths[0], self.scale_lengths[0], self.scale_lengths[0]), mode='trilinear', align_corners=False)
                feat2_resized = F.interpolate(feat2, size=(self.scale_lengths[1], self.scale_lengths[1], self.scale_lengths[1]), mode='trilinear', align_corners=False)
                feat3_resized = F.interpolate(feat3, size=(self.scale_lengths[2], self.scale_lengths[2], self.scale_lengths[2]), mode='trilinear', align_corners=False)
                feat4_resized = F.interpolate(feat4, size=(self.scale_lengths[3], self.scale_lengths[3], self.scale_lengths[3]), mode='trilinear', align_corners=False)
                
                final_outputs[0] = final_outputs[0] + feat1_resized
                final_outputs[1] = final_outputs[1] + feat2_resized
                final_outputs[2] = final_outputs[2] + feat3_resized
                final_outputs[3] = final_outputs[3] + feat4_resized
                return final_outputs
            
            # feats에서 feat1, feat2, feat3, feat4 사용 (위에서 이미 계산됨)
            feat1, feat2, feat3, feat4 = feats[:4]
            final_outputs = branch_norm_func(reconstruction_func(outputs_x, outputs_y, outputs_z))
            
            resize_wrapper = FunctionWrapper(resize_and_combine_func)
            flops, _ = profile(resize_wrapper, inputs=(feat1, feat2, feat3, feat4, final_outputs), verbose=False)
            step_flops['resize_and_combine'] = flops
            total_flops += flops
            
            # 8. Final fusion 측정
            def final_fusion_func(final_outputs):
                for i in range(len(final_outputs)):
                    final_outputs[i] = self.fuse_conv_final(final_outputs[i])
                return final_outputs
            
            final_fusion_wrapper = FunctionWrapper(final_fusion_func)
            flops, _ = profile(final_fusion_wrapper, inputs=(final_outputs,), verbose=False)
            step_flops['final_fusion'] = flops
            total_flops += flops
            
        except Exception as e:
            print(f"FLOPs 측정 중 오류 발생: {e}")
            return
        
        # 결과 출력
        print(f"\n=== Backbone FLOPs (Hybrid) ===")
        print(f"Total Backbone FLOPs: {total_flops/1e9:.2f} G")
        print(f"===============================\n")

    def forward(self, X):
        # X: [B, 1, D, H, W]
        B, C, D, H, W = X.shape
        device = X.device

        # Clear transformer cache for after reconstruction
        if hasattr(self, '_cached_after_transformer_weights'):
            self._cached_after_transformer_weights = {}
        
        # Clear cached original transformer input
        if hasattr(self, '_cached_original_transformer_input'):
            self._cached_original_transformer_input = None

        # FLOPs 측정 (한 번만 실행, training 모드에서만)
        if THOP_AVAILABLE and not self._flops_measured and self.training:
            self._measure_flops_once(X)
            self._flops_measured = True

        # 1. 첫번째 PE 적용 (3D → 2D 전)
        X_x, X_y, X_z = self._apply_pe_before_projection(X)

        # 2. 3D → 2D 프로젝션 (평균값)
        X_mean_x = X_x.mean(dim=-3)  # D 축 평균 → [B, 1, H, W]
        X_mean_y = X_y.mean(dim=-2)  # H 축 평균 → [B, 1, D, W]
        X_mean_z = X_z.mean(dim=-1)  # W 축 평균 → [B, 1, D, H]

        # 3. 각 축별 2D feature extraction
        outputs_x = self.model_x(X_mean_x)  # 튜플, 각 원소: [B, 256, F, F]
        outputs_y = self.model_y(X_mean_y)
        outputs_z = self.model_z(X_mean_z)

        # 4. 2D 결과를 3D로 복원: 각 스케일별로 누락된 축을 repeat한 후, 두번째 PE 적용
        final_outputs = []
        for i, (out_x, out_y, out_z) in enumerate(zip(outputs_x, outputs_y, outputs_z)):
            scale_length = self.scale_lengths[i]

            # 올바른 3D 재구성 방법 사용 (Conv2DBase_model 스타일)
            # out_x: [B, 256, H', W'] → [B, 256, scale_length, H', W']
            out_x_rep = out_x.unsqueeze(2).repeat(1, 1, scale_length, 1, 1)
            # 두번째 positional encoding 적용
            out_x_final = self._apply_pe_after_reconstruction(out_x_rep, 'x', i)

            # out_y: [B, 256, D', W'] → [B, 256, D', scale_length, W']
            out_y_rep = out_y.unsqueeze(3).repeat(1, 1, 1, scale_length, 1)
            # 두번째 positional encoding 적용
            out_y_final = self._apply_pe_after_reconstruction(out_y_rep, 'y', i)

            # out_z: [B, 256, D', H'] → [B, 256, D', H', scale_length]
            out_z_rep = out_z.unsqueeze(4).repeat(1, 1, 1, 1, scale_length)
            # 두번째 positional encoding 적용
            out_z_final = self._apply_pe_after_reconstruction(out_z_rep, 'z', i)

            # 세 축 branch 결과 평균
            combined = (out_x_final + out_y_final + out_z_final) / 3.0
            final_outputs.append(combined)

        # 5. 각 branch의 feature에 대해 개별 정규화 (fuse_conv_branch)
        for i in range(len(final_outputs)):
            final_outputs[i] = self.fuse_conv_branch[i](final_outputs[i])

        # 6. Low-resolution feature branch (피라미드)
        # 동적으로 base_size 계산
        input_size = X.shape[-3]
        base_size = int(input_size * self.ratio_3d)

        lowres_base = F.interpolate(X, size=(base_size, base_size, base_size), mode='trilinear', align_corners=False)
        feat1 = self.lowres_conv1(lowres_base)
        feat2 = self.lowres_conv2(feat1)
        feat3 = self.lowres_conv3(feat2)
        feat4 = self.lowres_conv4(feat3)

        if self.ratio_3d == 0.5:
            # 1/2 해상도: 5개 layer 사용하되, 4개 피라미드 스케일에 맞춤
            feat5 = self.lowres_conv5(feat4)
            
            # low-res features를 2D feature 크기에 맞게 다운샘플링
            feat1_resized = F.interpolate(feat1, size=(self.scale_lengths[0], self.scale_lengths[0], self.scale_lengths[0]), mode='trilinear', align_corners=False)
            feat2_resized = F.interpolate(feat2, size=(self.scale_lengths[1], self.scale_lengths[1], self.scale_lengths[1]), mode='trilinear', align_corners=False)
            feat3_resized = F.interpolate(feat3, size=(self.scale_lengths[2], self.scale_lengths[2], self.scale_lengths[2]), mode='trilinear', align_corners=False)
            feat4_resized = F.interpolate(feat4, size=(self.scale_lengths[3], self.scale_lengths[3], self.scale_lengths[3]), mode='trilinear', align_corners=False)
            
            # 7. 각 스케일별 low-res branch 피라미드 결합 (덧셈) - 4개 피라미드 스케일
            final_outputs[0] = final_outputs[0] + feat1_resized
            final_outputs[1] = final_outputs[1] + feat2_resized
            final_outputs[2] = final_outputs[2] + feat3_resized
            final_outputs[3] = final_outputs[3] + feat4_resized
        else:
            # 1/4 해상도: 4개 layer 사용 (기존 동작)
            
            # low-res features를 2D feature 크기에 맞게 다운샘플링
            feat1_resized = F.interpolate(feat1, size=(self.scale_lengths[0], self.scale_lengths[0], self.scale_lengths[0]), mode='trilinear', align_corners=False)
            feat2_resized = F.interpolate(feat2, size=(self.scale_lengths[1], self.scale_lengths[1], self.scale_lengths[1]), mode='trilinear', align_corners=False)
            feat3_resized = F.interpolate(feat3, size=(self.scale_lengths[2], self.scale_lengths[2], self.scale_lengths[2]), mode='trilinear', align_corners=False)
            feat4_resized = F.interpolate(feat4, size=(self.scale_lengths[3], self.scale_lengths[3], self.scale_lengths[3]), mode='trilinear', align_corners=False)
            
            # 7. 각 스케일별 low-res branch 피라미드 결합 (덧셈) - 4개 layer
            final_outputs[0] = final_outputs[0] + feat1_resized
            final_outputs[1] = final_outputs[1] + feat2_resized
            final_outputs[2] = final_outputs[2] + feat3_resized
            final_outputs[3] = final_outputs[3] + feat4_resized

        # 8. 결합 후 전체 feature 재정규화를 위한 fusion layer
        for i in range(len(final_outputs)):
            final_outputs[i] = self.fuse_conv_final(final_outputs[i])

        return final_outputs


# Swin Transformer FPN

def shifted_window_attention( # changed to 3D
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int] = [128, 128, 128],
    num_heads: int = 4,
    shift_size: List[int] = [64, 64, 64],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias for 3D.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, H, W, D, C]): The input tensor or 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
    Returns:
        Tensor[B, H, W, D, C]: The output tensor after shifted window attention.
    """
    B, H, W, D, C = input.shape
    # pad feature maps to multiples of window size -> 3D
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_d = (window_size[2] - D % window_size[2]) % window_size[2]
    x = F.pad(input, (0, 0, 0, pad_d, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, pad_D, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window -> 3D
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0
    if window_size[2] >= pad_D:
        shift_size[2] = 0

    # cyclic shift -> 3D
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # partition windows -> 3D
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1]) * (pad_D // window_size[2])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], pad_D // window_size[2], window_size[2], C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B * num_windows, window_size[0] * window_size[1] * window_size[2], C)  # B*nW, Ws*Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask #TODO: change to 3d
        attn_mask = x.new_zeros((pad_H, pad_W, pad_D))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        d_slices = ((0, -window_size[2]), (-window_size[2], -shift_size[2]), (-shift_size[2], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1], d[0] : d[1]] = count
                    count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], pad_D // window_size[2], window_size[2])
        attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(num_windows, window_size[0] * window_size[1] * window_size[2])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows -> 3D
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], pad_D // window_size[2], window_size[0], window_size[1], window_size[2], C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, pad_H, pad_W, pad_D, C)

    # reverse cyclic shift -> 3D
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # unpad features -> 3D
    x = x[:, :H, :W, :D, :].contiguous()
    return x


def _get_relative_position_bias( # changed to 3D
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1] * window_size[2]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


class ShiftedWindowAttention(nn.Module): # changed to 3D
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 3")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias -> 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1 * 2*Wd-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window -> 3D
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij"))  # 3, Wh, Ww, Wd
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[2] - 1) * (2 * self.window_size[1] - 1) # problematic
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1) # problematic
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wd*Wh*Ww*Wd
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, D, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, D, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )


class SwinTransformerBlock(nn.Module): # changed to 3D
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module): # changed to 3D
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm, expand_dim: bool = True):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, dim*2 if expand_dim else dim, bias=False)
        self.norm = norm_layer(8 * dim)
    
    def _patch_merging_pad(self, x: torch.Tensor) -> torch.Tensor: # changed to 3D
        H, W, D, _ = x.shape[-4:]
        x = F.pad(x, (0, 0, 0, D % 2, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x1 = x[..., 1::2, 0::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x2 = x[..., 0::2, 1::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x3 = x[..., 1::2, 1::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x4 = x[..., 0::2, 0::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x5 = x[..., 1::2, 0::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x6 = x[..., 0::2, 1::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x7 = x[..., 1::2, 1::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # ... H/2 W/2 D/2 8*C
        return x

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, D, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, D/2, C]
        """
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 D/2 2*C
        return x


class SwinTransformer_FPN(nn.Module): # TODO: change to 3D
    """
    Implements the 3D Swin Transformer FPN. 
    Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-5),
        block: Optional[Callable[..., nn.Module]] = SwinTransformerBlock,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        expand_dim: bool = True,
        out_channels: int = 256,
        input_dim: int = 4
    ):
        super().__init__()
        self.out_channels = out_channels
        # split image into non-overlapping patches
        self.patch_partition = nn.Sequential(
            nn.Conv3d(input_dim, embed_dim, kernel_size=(patch_size[0], patch_size[1], patch_size[2]), 
                      stride=(patch_size[0], patch_size[1], patch_size[2])),
            Permute([0, 2, 3, 4, 1]),
            norm_layer(embed_dim),
        )

        self.stages = nn.ModuleList()
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        fpn_in_channels = []

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = nn.ModuleList()
            dim = embed_dim * 2**i_stage if expand_dim else embed_dim
            fpn_in_channels.append(dim)

            # add patch merging layer
            if i_stage > 0:
                input_dim = fpn_in_channels[-2] if len(fpn_in_channels) > 1 else embed_dim
                stage.append(downsample_layer(input_dim, norm_layer, expand_dim))
            
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            self.stages.append(nn.Sequential(*stage))

        self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features : List[torch.Tensor] = []
        x = self.patch_partition(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous()) # [N, C, H, W, D]

        features = self.fpn_neck(features)
        return features 
