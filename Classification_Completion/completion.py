import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
from sklearn.metrics import f1_score
import open3d as o3d
try:
    from pytorch3d.loss import chamfer_distance
except ImportError:
    chamfer_distance = None
    print("Warning: pytorch3d is not installed. Chamfer distance will be unavailable.")

# Try to import thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop is not installed. FLOPs calculation will be skipped.")
    print("To install thop, run: pip install thop")

def seed_everything(seed=None):
    import time
    if seed is None:
        seed = int(time.time() * 1000) % 2**32  # Use current time as seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    torch.backends.cudnn.benchmark = False      # Disable benchmark to save memory
    print(f"Random seed set to: {seed}")
    return seed

class Transformer1D(nn.Module):
    def __init__(self, seq_len=3, input_dim=128, output_dim=128, d_model=128, nhead=8, num_layers=2):
        super(Transformer1D, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.transformer(x)
        weights = self.output_projection(x)
        weights = self.sigmoid(weights)
        return weights

class VoxelCompletionDataset(Dataset):
    def __init__(self, data_dir, partial_ratio=0.5):
        self.data_dir = data_dir
        self.partial_ratio = partial_ratio
        
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt') and not f.startswith('metadata')]
        
        meta_file = os.path.join(data_dir, "metadata.txt")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                lines = f.readlines()
                metadata = {}
                for line in lines:
                    key, value = line.strip().split(': ')
                    metadata[key] = value
                
                self.resolution = int(metadata.get('resolution', 128))
                self.class_name = metadata.get('class', 'unknown')
                self.phase = metadata.get('phase', 'unknown')
        else:
            self.resolution = 128
            self.class_name = 'unknown'
            self.phase = 'unknown'
        
        print(f"Dataset: {data_dir}, Files: {len(self.file_list)}, Resolution: {self.resolution}")
    
    def __len__(self):
        return len(self.file_list)
    
    def create_partial_input(self, voxel_grid):
        partial_grid = voxel_grid.clone()
        slice_idx = random.randint(0, self.resolution - 1)
        slice_dim = random.randint(0, 2)
        
        if slice_dim == 0:
            partial_grid[slice_idx:int(slice_idx + self.resolution * self.partial_ratio), :, :] = 0
        elif slice_dim == 1:
            partial_grid[:, slice_idx:int(slice_idx + self.resolution * self.partial_ratio), :] = 0
        else:
            partial_grid[:, :, slice_idx:int(slice_idx + self.resolution * self.partial_ratio)] = 0
        
        return partial_grid
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path)
        
        complete_grid = data['occupancy_grid']
        partial_grid = self.create_partial_input(complete_grid)
        
        return {
            "partial": partial_grid.unsqueeze(0),
            "complete": complete_grid.unsqueeze(0),
            "filename": data.get('filename', self.file_list[idx])
        }

def load_datasets(train_data, test_data, batch_size=16, partial_ratio=0.5):
    print("Loading datasets...")
    
    train_dataset = VoxelCompletionDataset(train_data, partial_ratio=partial_ratio)
    test_dataset = VoxelCompletionDataset(test_data, partial_ratio=partial_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    flops_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    
    return train_dataset, test_dataset, train_loader, test_loader, flops_loader

# 3D Convolutional Encoder-Decoder model for shape completion
class Conv3DCompletionNet(nn.Module):
    """
    3D Convolutional neural network for shape completion
    Using an encoder-decoder architecture without skip connections
    """
    def __init__(self, resolution=128, in_channels=1, out_channels=1):
        super(Conv3DCompletionNet, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )  # 64
        
        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )  # 32
        
        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )  # 16
        
        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )  # 8
        
        self.encoder5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )  # 4
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # 2 -> 4
        
        # Decoder without skip connections
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )  # 8
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )  # 16
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )  # 32
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )  # 64
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )  # 128

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Bottleneck
        bottleneck = self.bottleneck(e5)

        # Decoder without skip connections
        d5 = self.decoder5(bottleneck)
        d4 = self.decoder4(d5)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        # Ensure output is in valid range for BCELoss
        d1 = torch.clamp(d1, 0.0, 1.0)
        
        return d1

# 2D Projection-based Completion model
class Conv2DCompletionNet(nn.Module):
    """
    2D Projection-based neural network for shape completion
    Projects 3D volumes onto 2D planes along each axis and applies 2D convolutions
    """
    def __init__(self, resolution=128, in_channels=1, out_channels=1, use_pe=False, pe_type='transformer'):
        super(Conv2DCompletionNet, self).__init__()
        self.resolution = resolution
        self.use_pe = use_pe
        self.pe_type = pe_type

        if self.use_pe:
            if self.pe_type == 'transformer':
                self._create_transformer_pe_layers(resolution)
        
        input_channels = in_channels
        
        self.encoder_x = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.decoder_x = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.encoder_y = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.decoder_y = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.encoder_z = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.decoder_z = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def _create_transformer_pe_layers(self, resolution):
        self.pe_transformer_pre = Transformer1D(seq_len=3, input_dim=resolution, output_dim=resolution)
        self.pe_transformer_post = Transformer1D(seq_len=3, input_dim=resolution, output_dim=resolution)
        
        self._cached_transformer_weights_pre = None
        self._cached_transformer_weights_post = None

    def _apply_axis_specific_pe(self, x, axis):
        if not self.use_pe:
            return x
            
        if self.pe_type == 'transformer':
            if not hasattr(self, '_cached_transformer_weights_post'):
                self._cached_transformer_weights_post = self._apply_transformer_pe_post(x)
            
            weights = self._cached_transformer_weights_post
            batch_size = x.size(0)
            
            if axis == 'x':
                weight = weights['weight_x']
                weight_reshaped = weight.unsqueeze(1).unsqueeze(3).unsqueeze(4)
                x_with_pe = x * weight_reshaped
            elif axis == 'y':
                weight = weights['weight_y']
                weight_reshaped = weight.unsqueeze(1).unsqueeze(2).unsqueeze(4)
                x_with_pe = x * weight_reshaped
            else:
                weight = weights['weight_z']
                weight_reshaped = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x_with_pe = x * weight_reshaped
            
            return torch.clamp(x_with_pe, 0.0, 1.0)

        return x

    def _apply_positional_encoding(self, x, axis):
        if not self.use_pe:
            return x
            
        if self.pe_type == 'transformer':
            if not hasattr(self, '_cached_transformer_weights_pre'):
                self._cached_transformer_weights_pre = self._apply_transformer_pe(x)
            
            weights = self._cached_transformer_weights_pre
            batch_size = x.size(0)
            
            if axis == 0:
                weight = weights['weight_x']
                weight_reshaped = weight.unsqueeze(1).unsqueeze(3).unsqueeze(4)
                x_with_pe = x * (1.0 + weight_reshaped * 0.1)
            elif axis == 1:
                weight = weights['weight_y']
                weight_reshaped = weight.unsqueeze(1).unsqueeze(2).unsqueeze(4)
                x_with_pe = x * (1.0 + weight_reshaped * 0.1)
            else:
                weight = weights['weight_z']
                weight_reshaped = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                x_with_pe = x * (1.0 + weight_reshaped * 0.1)
            
            return torch.clamp(x_with_pe, 0.0, 1.0)

        return x

    def _apply_transformer_pe(self, x):
        batch_size = x.size(0)
        volume = x.squeeze(1)
        
        x_vec = volume.sum(dim=(2, 3))
        y_vec = volume.sum(dim=(1, 3))
        z_vec = volume.sum(dim=(1, 2))
        
        transformer_input = torch.stack([x_vec, y_vec, z_vec], dim=1)
        transformer_weights = self.pe_transformer_pre(transformer_input)
        
        weight_x = transformer_weights[:, 0, :self.resolution]
        weight_y = transformer_weights[:, 1, :self.resolution]
        weight_z = transformer_weights[:, 2, :self.resolution]
        
        return {
            'weight_x': weight_x,
            'weight_y': weight_y, 
            'weight_z': weight_z
        }

    def _apply_transformer_pe_post(self, x):
        batch_size = x.size(0)
        volume = x.squeeze(1)
        
        x_vec = volume.sum(dim=(2, 3))
        y_vec = volume.sum(dim=(1, 3))
        z_vec = volume.sum(dim=(1, 2))
        
        transformer_input = torch.stack([x_vec, y_vec, z_vec], dim=1)
        transformer_weights = self.pe_transformer_post(transformer_input)
        
        weight_x = transformer_weights[:, 0, :self.resolution]
        weight_y = transformer_weights[:, 1, :self.resolution]
        weight_z = transformer_weights[:, 2, :self.resolution]
        
        return {
            'weight_x': weight_x,
            'weight_y': weight_y, 
            'weight_z': weight_z
        }

    def forward(self, x):
        batch_size = x.size(0)
        channels = x.size(1)
        depth = x.size(2)
        height = x.size(3)
        width = x.size(4)
        
        if hasattr(self, '_cached_transformer_weights_pre'):
            delattr(self, '_cached_transformer_weights_pre')
        if hasattr(self, '_cached_transformer_weights_post'):
            delattr(self, '_cached_transformer_weights_post')
        
        if self.use_pe:
            x_with_pe_x = self._apply_positional_encoding(x, axis=0)
            x_with_pe_y = self._apply_positional_encoding(x, axis=1)
            x_with_pe_z = self._apply_positional_encoding(x, axis=2)
            
            proj_x = torch.mean(x_with_pe_x, dim=2)
            proj_y = torch.mean(x_with_pe_y, dim=3)
            proj_z = torch.mean(x_with_pe_z, dim=4)
        else:
            proj_x = torch.mean(x, dim=2)
            proj_y = torch.mean(x, dim=3)
            proj_z = torch.mean(x, dim=4)
        
        feat_x = self.encoder_x(proj_x)
        out_x = self.decoder_x(feat_x)
        feat_y = self.encoder_y(proj_y)
        out_y = self.decoder_y(feat_y)
        feat_z = self.encoder_z(proj_z)
        out_z = self.decoder_z(feat_z)

        out_x_3d = out_x.unsqueeze(2).repeat(1, 1, depth, 1, 1)
        out_y_3d = out_y.unsqueeze(3).repeat(1, 1, 1, height, 1)
        out_z_3d = out_z.unsqueeze(4).repeat(1, 1, 1, 1, width)
        
        if self.use_pe:
            out_x_3d = self._apply_axis_specific_pe(out_x_3d, axis='x')
            out_y_3d = self._apply_axis_specific_pe(out_y_3d, axis='y')
            out_z_3d = self._apply_axis_specific_pe(out_z_3d, axis='z')
        
        final_output = (out_x_3d + out_y_3d + out_z_3d) / 3.0
        final_output = torch.clamp(final_output, 0.0, 1.0)
        
        return final_output

# Hybrid 2D+3D Completion model
class HybridCompletionNet(nn.Module):
    """
    Hybrid neural network that combines 2D projection and 3D convolution for shape completion
    Uses configurable resolution 3D processing and full-resolution 2D processing, then combines them
    """
    def __init__(self, resolution=128, in_channels=1, out_channels=1, use_pe=False, pe_type='transformer', ratio_3d=0.5):
        super(HybridCompletionNet, self).__init__()
        self.resolution = resolution
        self.ratio_3d = ratio_3d
        self.resolution_3d = int(resolution * ratio_3d)
        
        self.conv2d_branch = Conv2DCompletionNet(resolution=resolution, in_channels=in_channels, out_channels=out_channels, use_pe=use_pe, pe_type=pe_type)
        
        # Create 3D convolution branch with configurable resolution
        if ratio_3d == 0.5:
            # Half resolution (64x64x64) - use existing full 3D branch
            self.conv3d_branch = Conv3DCompletionNet(resolution=self.resolution_3d, in_channels=in_channels, out_channels=out_channels)
        elif ratio_3d == 0.25:
            # Quarter resolution (32x32x32) - use reduced 3D branch
            self.conv3d_encoder1 = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            )  # 32 -> 16
            
            self.conv3d_encoder2 = nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            )  # 16 -> 8
            
            self.conv3d_encoder3 = nn.Sequential(
                nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True)
            )  # 8 -> 4
            
            # Bottleneck
            self.conv3d_bottleneck = nn.Sequential(
                nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )  # 4 -> 2 -> 4
            
            # Decoder
            self.conv3d_decoder3 = nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            )  # 4 -> 8
            
            self.conv3d_decoder2 = nn.Sequential(
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            )  # 8 -> 16
            
            self.conv3d_decoder1 = nn.Sequential(
                nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )  # 16 -> 32
        else:
            raise ValueError(f"Unsupported 3D ratio: {ratio_3d}. Supported values: 0.5, 0.25")
        
        # Fusion module to combine 2D and 3D outputs
        self.fusion = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out_2d = self.conv2d_branch(x)
        
        x_downsampled = F.interpolate(x, size=(self.resolution_3d, self.resolution_3d, self.resolution_3d), mode='trilinear', align_corners=False)
        
        if self.ratio_3d == 0.5:
            # Use existing 3D branch
            out_3d_low = self.conv3d_branch(x_downsampled)
        elif self.ratio_3d == 0.25:
            # Use reduced 3D branch
            e1 = self.conv3d_encoder1(x_downsampled)
            e2 = self.conv3d_encoder2(e1)
            e3 = self.conv3d_encoder3(e2)
            
            # Bottleneck
            bottleneck = self.conv3d_bottleneck(e3)
            
            # Decoder
            d3 = self.conv3d_decoder3(bottleneck)
            d2 = self.conv3d_decoder2(d3)
            out_3d_low = self.conv3d_decoder1(d2)
            
            # Ensure output is in valid range for BCELoss
            out_3d_low = torch.clamp(out_3d_low, 0.0, 1.0)
        
        out_3d_upsampled = F.interpolate(out_3d_low, size=(self.resolution, self.resolution, self.resolution), mode='trilinear', align_corners=False)
        
        # Simple average instead of concatenation
        combined_features = (out_2d + out_3d_upsampled) / 2.0
        final_output = self.fusion(combined_features)
        
        return final_output

# Chamfer distance 계산 함수 추가
def compute_l2_chamfer(gt_points, pred_points, resolution=128):
    """
    두 점 집합 간의 L2 Chamfer distance를 계산합니다.
    
    Args:
        gt_points: 실제 점(ground truth) 좌표 (N, 3)
        pred_points: 예측된 점 좌표 (M, 3)
        resolution: 복셀 그리드 해상도 (정규화에 사용)
        
    Returns:
        chamfer_dist: Chamfer distance 값
    """
    if len(gt_points) == 0 or len(pred_points) == 0:
        return 1.0
    
    gt = torch.from_numpy(gt_points.astype(np.float32) / resolution).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    pred = torch.from_numpy(pred_points.astype(np.float32) / resolution).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    
    cd, _ = chamfer_distance(gt, pred)
    return cd.item()

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    iou_scores = []
    batch_count = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Training (Epoch {epoch+1})")
    for batch_idx, batch in enumerate(pbar):
        partial = batch["partial"].to(device)
        complete = batch["complete"].to(device)

        output = model(partial)                  # [B,1,D,H,W]

        loss = criterion(output, complete)

        with torch.no_grad():
            pred_binary = (output > 0.5).float()
            intersection = torch.sum(pred_binary * complete).item()
            union        = torch.sum((pred_binary + complete) > 0).item()
            iou = intersection / (union + 1e-6)
            iou_scores.append(iou)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * complete.size(0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "IoU": f"{iou:.4f}",
            "batch": f"{batch_idx+1}/{batch_count}"
        })
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_iou = np.mean(iou_scores)
    print(f"Training completed: Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")
    return epoch_loss, epoch_iou

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    iou_scores = []
    f1_scores = []
    chamfer_distances = []
    
    print(f"Starting evaluation on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            partial = batch["partial"].to(device)
            complete = batch["complete"].to(device)

            output = model(partial)                  # [B,1,D,H,W]
            loss = criterion(output, complete)

            pred_binary = (output > 0.5).float()
            intersection = torch.sum(pred_binary * complete, dim=(1,2,3,4)).cpu().numpy()
            union        = torch.sum((pred_binary + complete) > 0, dim=(1,2,3,4)).cpu().numpy()
            batch_ious   = intersection / (union + 1e-6)
            iou_scores.extend(batch_ious)

            pred_flat = pred_binary.cpu().view(-1).numpy()
            target_flat = complete.cpu().view(-1).numpy()
            f1 = f1_score(target_flat, pred_flat, zero_division=1)
            f1_scores.append(f1)
            
            for i in range(complete.size(0)):
                complete_volume = complete[i, 0].cpu().numpy()
                complete_points = np.array(np.where(complete_volume > 0.5)).T
                
                pred_volume = pred_binary[i, 0].cpu().numpy()
                pred_points = np.array(np.where(pred_volume > 0.5)).T
                
                if len(pred_points) < 10:
                    chamfer_distances.append(1.0)
                    continue
                
                if len(complete_points) == 0:
                    chamfer_distances.append(1.0)
                    continue
                
                max_points = 5000
                if len(complete_points) > max_points:
                    indices = np.random.choice(len(complete_points), max_points, replace=False)
                    complete_points = complete_points[indices]
                
                if len(pred_points) > max_points:
                    indices = np.random.choice(len(pred_points), max_points, replace=False)
                    pred_points = pred_points[indices]
                
                cd = compute_l2_chamfer(complete_points, pred_points, resolution=complete_volume.shape[0])
                chamfer_distances.append(cd)
            
            test_loss += loss.item() * complete.size(0)
    
    test_loss /= len(test_loader.dataset)
    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)
    avg_cd = np.mean(chamfer_distances)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average L2 Chamfer Distance: {avg_cd:.6f}")
    
    result = {
        'loss': test_loss,
        'iou': avg_iou,
        'f1': avg_f1,
        'chamfer_distance': avg_cd
    }
    
    return result

# Main function
def main():
    parser = argparse.ArgumentParser(description='3D Shape Completion with 3D/2D Convolution')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='models/completion_model.pth', help='Path to save/load model')
    parser.add_argument('--train_data', type=str, default='processed_data/completion_chair_train_res128', help='Training data directory')
    parser.add_argument('--test_data', type=str, default='processed_data/completion_chair_test_res128', help='Test data directory')
    parser.add_argument('--partial_ratio', type=float, default=0.3, help='Ratio of model to remove')
    parser.add_argument('--pe_type', type=str, choices=['transformer'], help='Positional encoding type (transformer-based, automatically enables PE)')
    parser.add_argument('--3dratio', type=float, default=0.5, 
                       help='3D resolution ratio for hybrid model (default: 0.5 for half resolution, 0.25 for quarter resolution)')
    args = parser.parse_args()

    current_seed = seed_everything()  # Use random seed based on current time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = 'models/completion_hybrid.pth'
    ratio_3d = args.__dict__.get('3dratio', 0.5)
    print(f"Using hybrid 2D+3D convolution model with 3D ratio {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)}).")

    if args.model_path != 'models/completion_model.pth':
        model_path = args.model_path

    print("Loading datasets...")
    train_dataset, test_dataset, train_loader, test_loader, flops_loader = load_datasets(
        train_data=args.train_data,
        test_data=args.test_data,
        batch_size=args.batch_size,
        partial_ratio=args.partial_ratio
    )
    
    use_pe = args.pe_type is not None
    pe_type = args.pe_type if args.pe_type is not None else 'transformer'

    if args.pe_type is not None:
        print(f"Positional encoding enabled with type: {pe_type}")

    model = HybridCompletionNet(resolution=train_dataset.resolution, use_pe=use_pe, pe_type=pe_type, ratio_3d=args.__dict__.get('3dratio', 0.5)).to(device)

    # Calculate FLOPs for the model
    print(f"\n{'='*50}")
    print(f"MODEL COMPLEXITY ANALYSIS")
    print(f"{'='*50}")
    
    # Create dummy input for FLOPs calculation (batch_size=1 for single iteration analysis)
    dummy_input = torch.randn(1, 1, train_dataset.resolution, train_dataset.resolution, train_dataset.resolution).to(device)
    
    if THOP_AVAILABLE:
        try:
            # Calculate FLOPs and parameters
            model_copy = model
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)

            # Convert to more readable units
            flops_g = flops / 1e9  # GFLOPs
            params_m = params / 1e6  # Millions of parameters

            print(f"Model Type: HYBRID")
            ratio_3d = args.__dict__.get('3dratio', 0.5)
            print(f"3D Resolution Ratio: {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)})")
            if use_pe:
                print(f"Positional Encoding: {pe_type}")
            print(f"Input Shape: {dummy_input.shape}")
            print(f"Resolution: {train_dataset.resolution}x{train_dataset.resolution}x{train_dataset.resolution}")
            print(f"FLOPs per iteration: {flops_g:.2f} GFLOPs ({flops:,} FLOPs)")
            print(f"Parameters: {params_m:.2f} M ({params:,} params)")
            print(f"{'='*50}\n")

        except Exception as e:
            print(f"Warning: Could not calculate FLOPs due to: {e}")
            print(f"Model Type: HYBRID")
            ratio_3d = args.__dict__.get('3dratio', 0.5)
            print(f"3D Resolution Ratio: {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)})")
            if use_pe:
                print(f"Positional Encoding: {pe_type}")
            print(f"Input Shape: {dummy_input.shape}")
            print(f"Resolution: {train_dataset.resolution}x{train_dataset.resolution}x{train_dataset.resolution}")

            # Calculate parameters manually if FLOPs calculation fails
            total_params = sum(p.numel() for p in model.parameters())
            params_m = total_params / 1e6
            print(f"Parameters: {params_m:.2f} M ({total_params:,} params)")
            print(f"{'='*50}\n")
    else:
        # Only show basic info if thop is not available
        print(f"Model Type: HYBRID")
        ratio_3d = args.__dict__.get('3dratio', 0.5)
        print(f"3D Resolution Ratio: {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)})")
        if use_pe:
            print(f"Positional Encoding: {pe_type}")
        print(f"Input Shape: {dummy_input.shape}")
        print(f"Resolution: {train_dataset.resolution}x{train_dataset.resolution}x{train_dataset.resolution}")
        
        # Calculate parameters manually
        total_params = sum(p.numel() for p in model.parameters())
        params_m = total_params / 1e6
        print(f"Parameters: {params_m:.2f} M ({total_params:,} params)")
        print(f"FLOPs calculation skipped (thop not installed)")
        print(f"{'='*50}\n")
    
    learning_rate = args.lr
    print(f"Using hybrid completion model.")

    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create detailed results file with header
    detailed_result_file = os.path.join(results_dir, "detailed_completion.txt")
    
    # Check if detailed file exists to determine if this is the first run
    file_exists = os.path.exists(detailed_result_file)
    
    if not file_exists:
        # Create detailed results file with header
        with open(detailed_result_file, 'w') as f:
            f.write(f"3D Shape Completion Training Results\n")
            f.write(f"{'='*50}\n\n")
    
    # Add model section to detailed file
    with open(detailed_result_file, 'a') as f:
        # Determine model name for display
        model_name = 'Hybrid 2D+3D CNN'

        # Add PE information if enabled
        if use_pe:
            model_name = f'{model_name} (PE: {pe_type})'
        
        f.write(f"Model: HYBRID ({model_name})\n")
        ratio_3d = args.__dict__.get('3dratio', 0.5)
        f.write(f"3D Resolution Ratio: {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)})\n")
        f.write(f"{'='*30}\n")
        f.write(f"Training Data: {args.train_data}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Partial Ratio: {args.partial_ratio}\n")
        f.write(f"Total Epochs: {args.epochs}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*30}\n\n")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    best_test_iou = 0.0
    best_epoch = 0
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_results = test(model, test_loader, criterion, device)
        
        is_best = test_results['iou'] > best_test_iou
        if is_best:
            best_test_iou = test_results['iou']
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved! Test IoU: {best_test_iou:.4f}")
        
        # Log detailed results only
        with open(detailed_result_file, 'a') as f:
            f.write(f"Epoch {epoch+1:3d}: Train(Loss={train_loss:.6f}, IoU={train_iou:.6f}) | ")
            f.write(f"Test(Loss={test_results['loss']:.6f}, IoU={test_results['iou']:.6f}, ")
            f.write(f"F1={test_results['f1']:.6f}, CD={test_results['chamfer_distance']:.6f})")
            if is_best:
                f.write(" ⭐ BEST")
            f.write("\n")
    
    print(f"\nTraining completed!")
    print(f"Best test IoU: {best_test_iou:.4f} at epoch {best_epoch+1}")
    print(f"Model saved to: {model_path}")
    print(f"Detailed results saved to: {detailed_result_file}")
    
    # Save final summary to detailed file
    with open(detailed_result_file, 'a') as f:
        f.write(f"\nFinal Results for HYBRID:\n")
        f.write(f"  Best Test IoU: {best_test_iou:.4f} (Epoch {best_epoch+1})\n")
        f.write(f"  Model saved to: {model_path}\n")
        f.write(f"  Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")

if __name__ == "__main__":
    main()