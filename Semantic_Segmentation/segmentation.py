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
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import logging

# Try to import thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

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

def get_unique_filename(filepath):
    """중복 파일명을 방지하기 위해 고유한 파일명을 생성하는 함수"""
    if not os.path.exists(filepath):
        return filepath
    
    # 파일명과 확장자 분리
    base_path, ext = os.path.splitext(filepath)
    counter = 1
    
    # _1, _2, _3... 형식으로 숫자를 붙여서 고유한 파일명 찾기
    while os.path.exists(f"{base_path}_{counter}{ext}"):
        counter += 1
    
    new_filepath = f"{base_path}_{counter}{ext}"
    logger.info(f"File already exists. Saving as: {new_filepath}")
    return new_filepath

def calculate_model_complexity(model, input_shape, device):
    """Calculate model FLOPs and parameters"""
    if not THOP_AVAILABLE:
        return None, None
    
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        return None, None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ScanNet 클래스 라벨 정의
CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')

VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

# 색상 매핑 (시각화용)
SCANNET_COLOR_MAP = {
    0: (0, 0, 0),           # background
    1: (174, 199, 232),     # wall
    2: (152, 223, 138),     # floor
    3: (31, 119, 180),      # cabinet
    4: (255, 187, 120),     # bed
    5: (188, 189, 34),      # chair
    6: (140, 86, 75),       # sofa
    7: (255, 152, 150),     # table
    8: (214, 39, 40),       # door
    9: (197, 176, 213),     # window
    10: (148, 103, 189),    # bookshelf
    11: (196, 156, 148),    # picture
    12: (23, 190, 207),     # counter
    13: (247, 182, 210),    # desk
    14: (219, 219, 141),    # curtain
    15: (255, 127, 14),     # refrigerator
    16: (158, 218, 229),    # shower curtain
    17: (44, 160, 44),      # toilet
    18: (112, 128, 144),    # sink
    19: (227, 119, 194),    # bathtub
    20: (82, 84, 163),      # otherfurniture
}

class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation"""
    def __init__(self, smooth=1.0, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        # inputs: (N, C, D, H, W) - logits
        # targets: (N, D, H, W) - class indices
        
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Create mask to ignore certain indices
        valid_mask = (targets != self.ignore_index)
        
        # Flatten
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # (N, C, DHW)
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes=inputs.size(1)).float()  # (N, D, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).contiguous()  # (N, C, D, H, W)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)  # (N, C, DHW)
        
        # Apply mask
        valid_mask = valid_mask.view(valid_mask.size(0), 1, -1).expand_as(targets_one_hot)  # (N, C, DHW)
        inputs = inputs * valid_mask.float()
        targets_one_hot = targets_one_hot * valid_mask.float()
        
        # Calculate Dice coefficient for each class
        intersection = (inputs * targets_one_hot).sum(dim=2)  # (N, C)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)  # (N, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (N, C)
        dice_loss = 1.0 - dice.mean()  # Average over batch and classes
        
        return dice_loss

class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""
    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None, ignore_index=-1):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return combined_loss

class VoxelSegmentationDataset(Dataset):
    def __init__(self, data_dir, phase='train'):
        self.data_dir = data_dir
        self.phase = phase
        
        # 파일 목록 가져오기
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt') and not f.startswith('metadata')]
        
        # 메타데이터 읽기
        meta_file = os.path.join(data_dir, "metadata.txt")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                lines = f.readlines()
                metadata = {}
                for line in lines:
                    if ': ' in line:
                        key, value = line.strip().split(': ', 1)
                        metadata[key] = value
                
                self.resolution = int(metadata.get('resolution', 128))
                self.num_classes = int(metadata.get('num_classes', 21))
                self.phase_name = metadata.get('phase', 'unknown')
        else:
            self.resolution = 128
            self.num_classes = 21
            self.phase_name = 'unknown'
        
        logger.info(f"Dataset: {data_dir}")
        logger.info(f"Phase: {self.phase_name}")
        logger.info(f"Files: {len(self.file_list)}")
        logger.info(f"Resolution: {self.resolution}")
        logger.info(f"Classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path, weights_only=False)  # PyTorch 2.6 호환성
        
        # 데이터 로드 - 두 가지 키 형식 지원
        # 새로운 형식: occupancy, colors, labels (prepare_segmentation.py에서 생성)
        # 기존 형식: occupancy_grid, color_grid, label_grid (이전 버전 호환성)
        if 'occupancy' in data:
            # 새로운 형식 사용
            occupancy_grid = data['occupancy']  # (128, 128, 128)
            color_grid = data['colors']         # (128, 128, 128, 3)
            label_grid = data['labels']         # (128, 128, 128)
        else:
            # 기존 형식 사용 (하위 호환성)
            occupancy_grid = data['occupancy_grid']  # (128, 128, 128)
            color_grid = data['color_grid']          # (128, 128, 128, 3)
            label_grid = data['label_grid']          # (128, 128, 128)
        
        # numpy array를 tensor로 변환 (필요한 경우)
        if isinstance(occupancy_grid, np.ndarray):
            occupancy_grid = torch.from_numpy(occupancy_grid)
        if isinstance(color_grid, np.ndarray):
            color_grid = torch.from_numpy(color_grid)
        if isinstance(label_grid, np.ndarray):
            label_grid = torch.from_numpy(label_grid)
        
        # ignore(-1)은 그대로 유지 (학습/평가에서 제외됨)
        # label_grid = torch.where(label_grid == -1, 0, label_grid)  # 이 라인 제거
        
        # 입력 데이터: occupancy + color (4 channels)
        # occupancy를 첫 번째 채널로, color를 나머지 3개 채널로 사용
        input_data = torch.zeros(4, self.resolution, self.resolution, self.resolution)
        input_data[0] = occupancy_grid.float()
        input_data[1:4] = color_grid.permute(3, 0, 1, 2).float()  # (3, 128, 128, 128)
        
        # 라벨 데이터 (ignore는 -1로 유지)
        target = label_grid.long()
        
        # 파일명 정보 - 여러 형식 지원
        filename = data.get('filename') or data.get('scene_name') or self.file_list[idx]
        
        return {
            "input": input_data,
            "target": target,
            "filename": filename
        }

def load_datasets(train_data, test_data, batch_size=8):
    logger.info("Loading segmentation datasets...")
    
    train_dataset = VoxelSegmentationDataset(train_data, phase='train')
    test_dataset = VoxelSegmentationDataset(test_data, phase='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True, drop_last=False)
    flops_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                           num_workers=2, pin_memory=True, drop_last=False)
    
    return train_dataset, test_dataset, train_loader, test_loader, flops_loader

class DoubleConv3D(nn.Module):
    """Double Convolution Block for 3D U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2D(nn.Module):
    """Double Convolution Block for 2D U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Conv2DSegmentationNet(nn.Module):
    """
    2D projection-based neural network for semantic segmentation using U-Net architecture
    """
    def __init__(self, in_channels=4, num_classes=21, resolution=128, use_pe=False, pe_type='transformer'):
        super(Conv2DSegmentationNet, self).__init__()
        self.num_classes = num_classes
        self.resolution = resolution
        self.use_pe = use_pe
        self.pe_type = pe_type if use_pe else None
        
        # Initialize positional encoding if enabled
        if self.use_pe:
            if pe_type == 'transformer':
                self._create_transformer_pe_layers(resolution)
        
        # X-axis projection U-Net - Reduced channels for memory efficiency
        self.encoder_x1 = DoubleConv2D(in_channels, 32)
        self.pool_x1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_x2 = DoubleConv2D(32, 64)
        self.pool_x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_x3 = DoubleConv2D(64, 128)
        self.pool_x3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_x4 = DoubleConv2D(128, 256)
        self.pool_x4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck_x = DoubleConv2D(256, 512)
        
        self.upconv_x4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_x4 = DoubleConv2D(512, 256)
        self.upconv_x3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_x3 = DoubleConv2D(256, 128)
        self.upconv_x2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_x2 = DoubleConv2D(128, 64)
        self.upconv_x1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_x1 = DoubleConv2D(64, 32)
        self.output_x = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Y-axis projection U-Net
        self.encoder_y1 = DoubleConv2D(in_channels, 32)
        self.pool_y1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_y2 = DoubleConv2D(32, 64)
        self.pool_y2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_y3 = DoubleConv2D(64, 128)
        self.pool_y3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_y4 = DoubleConv2D(128, 256)
        self.pool_y4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck_y = DoubleConv2D(256, 512)
        
        self.upconv_y4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_y4 = DoubleConv2D(512, 256)
        self.upconv_y3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_y3 = DoubleConv2D(256, 128)
        self.upconv_y2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_y2 = DoubleConv2D(128, 64)
        self.upconv_y1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_y1 = DoubleConv2D(64, 32)
        self.output_y = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Z-axis projection U-Net
        self.encoder_z1 = DoubleConv2D(in_channels, 32)
        self.pool_z1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_z2 = DoubleConv2D(32, 64)
        self.pool_z2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_z3 = DoubleConv2D(64, 128)
        self.pool_z3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_z4 = DoubleConv2D(128, 256)
        self.pool_z4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck_z = DoubleConv2D(256, 512)
        
        self.upconv_z4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_z4 = DoubleConv2D(512, 256)
        self.upconv_z3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_z3 = DoubleConv2D(256, 128)
        self.upconv_z2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_z2 = DoubleConv2D(128, 64)
        self.upconv_z1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_z1 = DoubleConv2D(64, 32)
        self.output_z = nn.Conv2d(32, num_classes, kernel_size=1)

        # Final classifier
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _create_transformer_pe_layers(self, resolution):
        """Create transformer-based positional encoding layers"""
        # Using 3 axes × 4 channels = 12 tokens with rich 128-dim embeddings
        seq_len = 3 * 4  # 3 axes × (occupancy + RGB) = 12 tokens
        self.pe_transformer_pre = Transformer1D(seq_len=seq_len, input_dim=resolution, output_dim=resolution)
        self.pe_transformer_post = Transformer1D(seq_len=seq_len, input_dim=resolution, output_dim=resolution)
        
        self._cached_transformer_weights_pre = None
        self._cached_transformer_weights_post = None

    def _apply_axis_specific_pe(self, x, axis):
        """Apply axis-specific positional encoding"""
        if not self.use_pe:
            return x
            
        if self.pe_type == 'transformer':
            if not hasattr(self, '_cached_transformer_weights_post'):
                self._cached_transformer_weights_post = self._apply_transformer_pe_post(x)
            
            weights = self._cached_transformer_weights_post
            batch_size = x.size(0)
            
            # Create a copy to avoid in-place operations
            x_modified = x.clone()
            
            if axis == 'x':
                # X축: 채널별로 다른 가중치 적용
                x_channels = weights['x_channels']  # (B, 4, 128)
                # occupancy 채널
                occ_weight = x_channels[:, 0].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                x_modified[:, 0:1] = x[:, 0:1] * (1.0 + occ_weight * 0.1)
                # RGB 채널들
                for c in range(3):
                    rgb_weight = x_channels[:, c+1].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * (1.0 + rgb_weight * 0.1)
            elif axis == 'y':
                # Y축: 채널별로 다른 가중치 적용
                y_channels = weights['y_channels']  # (B, 4, 128)
                # occupancy 채널
                occ_weight = y_channels[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                x_modified[:, 0:1] = x[:, 0:1] * (1.0 + occ_weight * 0.1)
                # RGB 채널들
                for c in range(3):
                    rgb_weight = y_channels[:, c+1].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * (1.0 + rgb_weight * 0.1)
            else:
                # Z축: 채널별로 다른 가중치 적용
                z_channels = weights['z_channels']  # (B, 4, 128)
                # occupancy 채널
                occ_weight = z_channels[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                x_modified[:, 0:1] = x[:, 0:1] * (1.0 + occ_weight * 0.1)
                # RGB 채널들
                for c in range(3):
                    rgb_weight = z_channels[:, c+1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * (1.0 + rgb_weight * 0.1)
            
            return torch.clamp(x_modified, 0.0, 1.0)

        return x

    def _apply_positional_encoding(self, x, axis):
        """Apply positional encoding during projection"""
        if not self.use_pe:
            return x
            
        if self.pe_type == 'transformer':
            if not hasattr(self, '_cached_transformer_weights_pre'):
                self._cached_transformer_weights_pre = self._apply_transformer_pe(x)
            
            weights = self._cached_transformer_weights_pre
            batch_size = x.size(0)
            
            # Create a copy to avoid in-place operations
            x_modified = x.clone()
            
            if axis == 0:
                # X축: 채널별로 다른 가중치 적용
                x_channels = weights['x_channels']  # (B, 4, 128)
                # occupancy 채널
                occ_weight = x_channels[:, 0].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                x_modified[:, 0:1] = x[:, 0:1] * occ_weight
                # RGB 채널들
                for c in range(3):
                    rgb_weight = x_channels[:, c+1].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * rgb_weight
            elif axis == 1:
                # Y축: 채널별로 다른 가중치 적용
                y_channels = weights['y_channels']  # (B, 4, 128)
                # occupancy 채널
                occ_weight = y_channels[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                x_modified[:, 0:1] = x[:, 0:1] * occ_weight
                # RGB 채널들
                for c in range(3):
                    rgb_weight = y_channels[:, c+1].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * rgb_weight
            else:
                # Z축: 채널별로 다른 가중치 적용
                z_channels = weights['z_channels']  # (B, 4, 128)
                # occupancy 채널
                occ_weight = z_channels[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                x_modified[:, 0:1] = x[:, 0:1] * occ_weight
                # RGB 채널들
                for c in range(3):
                    rgb_weight = z_channels[:, c+1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * rgb_weight
            
            return torch.clamp(x_modified, 0.0, 1.0)

        return x

    def _apply_transformer_pe(self, x):
        """Apply transformer-based positional encoding"""
        batch_size = x.size(0)
        
        # Separate occupancy and RGB channels
        occupancy = x[:, 0:1]  # (B, 1, D, H, W) - occupancy channel
        rgb = x[:, 1:4]        # (B, 3, D, H, W) - RGB channels
        
        # Create channel-wise projections for each axis (12 tokens total)
        tokens = []
        
        # X-axis projections (average over H,W dimensions)
        occ_x = occupancy.squeeze(1).mean(dim=(2, 3))  # (B, D=128) - occupancy along X
        r_x = rgb[:, 0].mean(dim=(2, 3))               # (B, D=128) - R channel along X  
        g_x = rgb[:, 1].mean(dim=(2, 3))               # (B, D=128) - G channel along X
        b_x = rgb[:, 2].mean(dim=(2, 3))               # (B, D=128) - B channel along X
        tokens.extend([occ_x, r_x, g_x, b_x])
        
        # Y-axis projections (average over D,W dimensions)  
        occ_y = occupancy.squeeze(1).mean(dim=(1, 3))  # (B, H=128) - occupancy along Y
        r_y = rgb[:, 0].mean(dim=(1, 3))               # (B, H=128) - R channel along Y
        g_y = rgb[:, 1].mean(dim=(1, 3))               # (B, H=128) - G channel along Y
        b_y = rgb[:, 2].mean(dim=(1, 3))               # (B, H=128) - B channel along Y
        tokens.extend([occ_y, r_y, g_y, b_y])
        
        # Z-axis projections (average over D,H dimensions)
        occ_z = occupancy.squeeze(1).mean(dim=(1, 2))  # (B, W=128) - occupancy along Z
        r_z = rgb[:, 0].mean(dim=(1, 2))               # (B, W=128) - R channel along Z
        g_z = rgb[:, 1].mean(dim=(1, 2))               # (B, W=128) - G channel along Z  
        b_z = rgb[:, 2].mean(dim=(1, 2))               # (B, W=128) - B channel along Z
        tokens.extend([occ_z, r_z, g_z, b_z])
        
        # Stack tokens: (B, 12, 128)
        transformer_input = torch.stack(tokens, dim=1)  # (B, 12, 128)
        
        transformer_weights = self.pe_transformer_pre(transformer_input)  # (B, 12, 128)
        
        # Extract channel-wise weights for each axis (no MLP combining)
        # X-axis: 4 channels separately
        x_channels = transformer_weights[:, 0:4]  # (B, 4, 128)
        
        # Y-axis: 4 channels separately  
        y_channels = transformer_weights[:, 4:8]  # (B, 4, 128)
        
        # Z-axis: 4 channels separately
        z_channels = transformer_weights[:, 8:12]  # (B, 4, 128)
        
        return {
            'x_channels': x_channels,
            'y_channels': y_channels, 
            'z_channels': z_channels
        }

    def _apply_transformer_pe_post(self, x):
        """Apply post-processing transformer-based positional encoding"""
        batch_size = x.size(0)
        
        # Separate occupancy and RGB channels
        occupancy = x[:, 0:1]  # (B, 1, D, H, W) - occupancy channel
        rgb = x[:, 1:4]        # (B, 3, D, H, W) - RGB channels
        
        # Create channel-wise projections for each axis (12 tokens total)
        tokens = []
        
        # X-axis projections (average over H,W dimensions)
        occ_x = occupancy.squeeze(1).mean(dim=(2, 3))  # (B, D=128) - occupancy along X
        r_x = rgb[:, 0].mean(dim=(2, 3))               # (B, D=128) - R channel along X  
        g_x = rgb[:, 1].mean(dim=(2, 3))               # (B, D=128) - G channel along X
        b_x = rgb[:, 2].mean(dim=(2, 3))               # (B, D=128) - B channel along X
        tokens.extend([occ_x, r_x, g_x, b_x])
        
        # Y-axis projections (average over D,W dimensions)  
        occ_y = occupancy.squeeze(1).mean(dim=(1, 3))  # (B, H=128) - occupancy along Y
        r_y = rgb[:, 0].mean(dim=(1, 3))               # (B, H=128) - R channel along Y
        g_y = rgb[:, 1].mean(dim=(1, 3))               # (B, H=128) - G channel along Y
        b_y = rgb[:, 2].mean(dim=(1, 3))               # (B, H=128) - B channel along Y
        tokens.extend([occ_y, r_y, g_y, b_y])
        
        # Z-axis projections (average over D,H dimensions)
        occ_z = occupancy.squeeze(1).mean(dim=(1, 2))  # (B, W=128) - occupancy along Z
        r_z = rgb[:, 0].mean(dim=(1, 2))               # (B, W=128) - R channel along Z
        g_z = rgb[:, 1].mean(dim=(1, 2))               # (B, W=128) - G channel along Z  
        b_z = rgb[:, 2].mean(dim=(1, 2))               # (B, W=128) - B channel along Z
        tokens.extend([occ_z, r_z, g_z, b_z])
        
        # Stack tokens: (B, 12, 128)
        transformer_input = torch.stack(tokens, dim=1)  # (B, 12, 128)
        
        transformer_weights = self.pe_transformer_post(transformer_input)  # (B, 12, 128)
        
        # Extract channel-wise weights for each axis (no MLP combining)
        # X-axis: 4 channels separately
        x_channels = transformer_weights[:, 0:4]  # (B, 4, 128)
        
        # Y-axis: 4 channels separately  
        y_channels = transformer_weights[:, 4:8]  # (B, 4, 128)
        
        # Z-axis: 4 channels separately
        z_channels = transformer_weights[:, 8:12]  # (B, 4, 128)
        
        return {
            'x_channels': x_channels,
            'y_channels': y_channels, 
            'z_channels': z_channels
        }

    def _apply_transformer_pe_optimized(self, x):
        """Memory-optimized transformer PE - calculate once, use multiple times"""
        batch_size = x.size(0)
        
        # Separate occupancy and RGB channels
        occupancy = x[:, 0:1]  # (B, 1, D, H, W) - occupancy channel
        rgb = x[:, 1:4]        # (B, 3, D, H, W) - RGB channels
        
        # Create channel-wise projections for each axis (12 tokens total)
        tokens = []
        
        # X-axis projections (average over H,W dimensions)
        occ_x = occupancy.squeeze(1).mean(dim=(2, 3))  # (B, D=128) - occupancy along X
        r_x = rgb[:, 0].mean(dim=(2, 3))               # (B, D=128) - R channel along X  
        g_x = rgb[:, 1].mean(dim=(2, 3))               # (B, D=128) - G channel along X
        b_x = rgb[:, 2].mean(dim=(2, 3))               # (B, D=128) - B channel along X
        tokens.extend([occ_x, r_x, g_x, b_x])
        
        # Y-axis projections (average over D,W dimensions)  
        occ_y = occupancy.squeeze(1).mean(dim=(1, 3))  # (B, H=128) - occupancy along Y
        r_y = rgb[:, 0].mean(dim=(1, 3))               # (B, H=128) - R channel along Y
        g_y = rgb[:, 1].mean(dim=(1, 3))               # (B, H=128) - G channel along Y
        b_y = rgb[:, 2].mean(dim=(1, 3))               # (B, H=128) - B channel along Y
        tokens.extend([occ_y, r_y, g_y, b_y])
        
        # Z-axis projections (average over D,H dimensions)
        occ_z = occupancy.squeeze(1).mean(dim=(1, 2))  # (B, W=128) - occupancy along Z
        r_z = rgb[:, 0].mean(dim=(1, 2))               # (B, W=128) - R channel along Z
        g_z = rgb[:, 1].mean(dim=(1, 2))               # (B, W=128) - G channel along Z  
        b_z = rgb[:, 2].mean(dim=(1, 2))               # (B, W=128) - B channel along Z
        tokens.extend([occ_z, r_z, g_z, b_z])
        
        # Stack tokens: (B, 12, 128)
        transformer_input = torch.stack(tokens, dim=1)  # (B, 12, 128)
        
        # Single transformer call for all weights
        transformer_weights = self.pe_transformer_pre(transformer_input)  # (B, 12, 128)
        
        # Extract all channel-wise weights at once
        return {
            # Pre-projection weights (for input processing)
            'x_channels_pre': transformer_weights[:, 0:4],    # (B, 4, 128)
            'y_channels_pre': transformer_weights[:, 4:8],    # (B, 4, 128)
            'z_channels_pre': transformer_weights[:, 8:12],   # (B, 4, 128)
            
            # Post-reconstruction weights (reuse same weights with different scaling)
            'x_channels_post': transformer_weights[:, 0:4] * 0.5,   # (B, 4, 128) - scaled for post-processing
            'y_channels_post': transformer_weights[:, 4:8] * 0.5,   # (B, 4, 128)
            'z_channels_post': transformer_weights[:, 8:12] * 0.5,  # (B, 4, 128)
        }

    def process_unet_2d(self, x, axis):
        """Process 2D U-Net for specific axis"""
        if axis == 'x':
            # Encoder
            enc1 = self.encoder_x1(x)
            x = self.pool_x1(enc1)
            enc2 = self.encoder_x2(x)
            x = self.pool_x2(enc2)
            enc3 = self.encoder_x3(x)
            x = self.pool_x3(enc3)
            enc4 = self.encoder_x4(x)
            x = self.pool_x4(enc4)
            
            # Bottleneck
            x = self.bottleneck_x(x)
            
            # Decoder
            x = self.upconv_x4(x)
            x = torch.cat([x, enc4], dim=1)
            x = self.decoder_x4(x)
            x = self.upconv_x3(x)
            x = torch.cat([x, enc3], dim=1)
            x = self.decoder_x3(x)
            x = self.upconv_x2(x)
            x = torch.cat([x, enc2], dim=1)
            x = self.decoder_x2(x)
            x = self.upconv_x1(x)
            x = torch.cat([x, enc1], dim=1)
            x = self.decoder_x1(x)
            return self.output_x(x)
            
        elif axis == 'y':
            # Encoder
            enc1 = self.encoder_y1(x)
            x = self.pool_y1(enc1)
            enc2 = self.encoder_y2(x)
            x = self.pool_y2(enc2)
            enc3 = self.encoder_y3(x)
            x = self.pool_y3(enc3)
            enc4 = self.encoder_y4(x)
            x = self.pool_y4(enc4)
            
            # Bottleneck
            x = self.bottleneck_y(x)
            
            # Decoder
            x = self.upconv_y4(x)
            x = torch.cat([x, enc4], dim=1)
            x = self.decoder_y4(x)
            x = self.upconv_y3(x)
            x = torch.cat([x, enc3], dim=1)
            x = self.decoder_y3(x)
            x = self.upconv_y2(x)
            x = torch.cat([x, enc2], dim=1)
            x = self.decoder_y2(x)
            x = self.upconv_y1(x)
            x = torch.cat([x, enc1], dim=1)
            x = self.decoder_y1(x)
            return self.output_y(x)
            
        else:  # axis == 'z'
            # Encoder
            enc1 = self.encoder_z1(x)
            x = self.pool_z1(enc1)
            enc2 = self.encoder_z2(x)
            x = self.pool_z2(enc2)
            enc3 = self.encoder_z3(x)
            x = self.pool_z3(enc3)
            enc4 = self.encoder_z4(x)
            x = self.pool_z4(enc4)
            
            # Bottleneck
            x = self.bottleneck_z(x)
            
            # Decoder
            x = self.upconv_z4(x)
            x = torch.cat([x, enc4], dim=1)
            x = self.decoder_z4(x)
            x = self.upconv_z3(x)
            x = torch.cat([x, enc3], dim=1)
            x = self.decoder_z3(x)
            x = self.upconv_z2(x)
            x = torch.cat([x, enc2], dim=1)
            x = self.decoder_z2(x)
            x = self.upconv_z1(x)
            x = torch.cat([x, enc1], dim=1)
            x = self.decoder_z1(x)
            return self.output_z(x)

    def forward(self, x):
        batch_size = x.size(0)
        channels = x.size(1)
        depth = x.size(2)
        height = x.size(3)
        width = x.size(4)
        
        # Calculate transformer weights once for all operations (memory optimization)
        if self.use_pe and self.pe_type == 'transformer':
            all_transformer_weights = self._apply_transformer_pe_optimized(x)
        
        # Apply positional encoding to input volume before projection
        if self.use_pe:
            if self.pe_type == 'transformer':
                # Use pre-calculated weights (no additional transformer calls)
                x_modified = x.clone()
                
                # X dimension
                x_channels = all_transformer_weights['x_channels_pre']  # (B, 4, 128)
                occ_weight = x_channels[:, 0].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                x_modified[:, 0:1] = x[:, 0:1] * (1.0 + occ_weight * 0.1)
                for c in range(3):
                    rgb_weight = x_channels[:, c+1].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * (1.0 + rgb_weight * 0.1)
                x = torch.clamp(x_modified, 0.0, 1.0)
                
                # Y dimension
                x_modified = x.clone()
                y_channels = all_transformer_weights['y_channels_pre']  # (B, 4, 128)
                occ_weight = y_channels[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                x_modified[:, 0:1] = x[:, 0:1] * (1.0 + occ_weight * 0.1)
                for c in range(3):
                    rgb_weight = y_channels[:, c+1].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * (1.0 + rgb_weight * 0.1)
                x = torch.clamp(x_modified, 0.0, 1.0)
                
                # Z dimension
                x_modified = x.clone()
                z_channels = all_transformer_weights['z_channels_pre']  # (B, 4, 128)
                occ_weight = z_channels[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                x_modified[:, 0:1] = x[:, 0:1] * (1.0 + occ_weight * 0.1)
                for c in range(3):
                    rgb_weight = z_channels[:, c+1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                    x_modified[:, c+1:c+2] = x[:, c+1:c+2] * (1.0 + rgb_weight * 0.1)
                x = torch.clamp(x_modified, 0.0, 1.0)
            else:
                # Use existing PE methods for non-transformer types
                x = self._apply_positional_encoding(x, 0)  # Apply to X dimension
                x = self._apply_positional_encoding(x, 1)  # Apply to Y dimension  
                x = self._apply_positional_encoding(x, 2)  # Apply to Z dimension
        
        # Project 3D volume onto 2D planes along each axis
        # X-axis projection: average along depth (dim=2)
        proj_x = torch.mean(x, dim=2)  # (batch, channels, height, width)
        
        # Y-axis projection: average along height (dim=3)
        proj_y = torch.mean(x, dim=3)  # (batch, channels, depth, width)
        
        # Z-axis projection: average along width (dim=4)
        proj_z = torch.mean(x, dim=4)  # (batch, channels, depth, height)
        
        # Process each projection with 2D U-Net
        out_x = self.process_unet_2d(proj_x, 'x')  # (batch, num_classes, height, width)
        out_y = self.process_unet_2d(proj_y, 'y')  # (batch, num_classes, depth, width)
        out_z = self.process_unet_2d(proj_z, 'z')  # (batch, num_classes, depth, height)
        
        # Reconstruct 3D segmentation from 2D projections
        # Expand 2D outputs back to 3D
        out_x_3d = out_x.unsqueeze(2).repeat(1, 1, depth, 1, 1)     # (batch, num_classes, depth, height, width)
        out_y_3d = out_y.unsqueeze(3).repeat(1, 1, 1, height, 1)    # (batch, num_classes, depth, height, width)
        out_z_3d = out_z.unsqueeze(4).repeat(1, 1, 1, 1, width)     # (batch, num_classes, depth, height, width)
        
        # Apply axis-specific positional encoding to reconstructed 3D volumes
        if self.use_pe:
            if self.pe_type == 'transformer':
                # Use pre-calculated weights (no additional transformer calls)
                out_x_3d_modified = out_x_3d.clone()
                x_channels_post = all_transformer_weights['x_channels_post']  # (B, 4, 128)
                # Apply post-processing weights (using num_classes channels)
                for cls in range(self.num_classes):
                    if cls < 4:  # Use channel-specific weights for first 4 classes
                        weight = x_channels_post[:, cls].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                    else:  # Use average for additional classes
                        weight = x_channels_post.mean(dim=1).unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, 128, 1, 1)
                    out_x_3d_modified[:, cls:cls+1] = out_x_3d[:, cls:cls+1] * (1.0 + weight * 0.1)
                out_x_3d = out_x_3d_modified
                
                out_y_3d_modified = out_y_3d.clone()
                y_channels_post = all_transformer_weights['y_channels_post']  # (B, 4, 128)
                for cls in range(self.num_classes):
                    if cls < 4:
                        weight = y_channels_post[:, cls].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                    else:
                        weight = y_channels_post.mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, 128, 1)
                    out_y_3d_modified[:, cls:cls+1] = out_y_3d[:, cls:cls+1] * (1.0 + weight * 0.1)
                out_y_3d = out_y_3d_modified
                
                out_z_3d_modified = out_z_3d.clone()
                z_channels_post = all_transformer_weights['z_channels_post']  # (B, 4, 128)
                for cls in range(self.num_classes):
                    if cls < 4:
                        weight = z_channels_post[:, cls].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                    else:
                        weight = z_channels_post.mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, 128)
                    out_z_3d_modified[:, cls:cls+1] = out_z_3d[:, cls:cls+1] * (1.0 + weight * 0.1)
                out_z_3d = out_z_3d_modified
                
            else:
                # Apply axis-specific PE for other types
                out_x_3d = self._apply_axis_specific_pe(out_x_3d, axis='x')
                out_y_3d = self._apply_axis_specific_pe(out_y_3d, axis='y')
                out_z_3d = self._apply_axis_specific_pe(out_z_3d, axis='z')
        
        # Combine the three projections (average)
        final_output = (out_x_3d + out_y_3d + out_z_3d) / 3.0
        
        return final_output

class HybridSegmentationNet(nn.Module):
    """
    Hybrid neural network that combines 2D projection and 3D convolution for semantic segmentation
    Uses configurable resolution 3D processing and full-resolution 2D processing, then combines them
    """
    def __init__(self, in_channels=4, num_classes=21, resolution=128, use_pe=False, pe_type='transformer', ratio_3d=0.5):
        super(HybridSegmentationNet, self).__init__()
        self.num_classes = num_classes
        self.resolution = resolution
        self.ratio_3d = ratio_3d
        self.resolution_3d = int(resolution * ratio_3d)
        self.use_pe = use_pe
        self.pe_type = pe_type if use_pe else None
        
        # 2D projection branch - with PE support
        self.conv2d_branch = Conv2DSegmentationNet(
            in_channels=in_channels, 
            num_classes=num_classes, 
            resolution=resolution,
            use_pe=use_pe,
            pe_type=pe_type
        )
        
        # 3D convolution branch with configurable resolution
        if ratio_3d == 0.5:
            # Half resolution (64x64x64) - 4 encoder + 4 decoder layers
            self.encoder3d_1 = DoubleConv3D(in_channels, 32)
            self.pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.encoder3d_2 = DoubleConv3D(32, 64)
            self.pool3d_2 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.encoder3d_3 = DoubleConv3D(64, 128)
            self.pool3d_3 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.encoder3d_4 = DoubleConv3D(128, 256)
            self.pool3d_4 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            # Bottleneck
            self.bottleneck3d = DoubleConv3D(256, 512)
            
            # Decoder
            self.upconv3d_4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
            self.decoder3d_4 = DoubleConv3D(512, 256)  # 256 + 256 from skip connection
            
            self.upconv3d_3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
            self.decoder3d_3 = DoubleConv3D(256, 128)   # 128 + 128 from skip connection
            
            self.upconv3d_2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
            self.decoder3d_2 = DoubleConv3D(128, 64)    # 64 + 64 from skip connection
            
            self.upconv3d_1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
            self.decoder3d_1 = DoubleConv3D(64, 32)    # 32 + 32 from connection
            
        elif ratio_3d == 0.25:
            # Quarter resolution (32x32x32) - 3 encoder + 3 decoder layers
            self.encoder3d_1 = DoubleConv3D(in_channels, 32)
            self.pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.encoder3d_2 = DoubleConv3D(32, 64)
            self.pool3d_2 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.encoder3d_3 = DoubleConv3D(64, 128)
            self.pool3d_3 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            # Bottleneck
            self.bottleneck3d = DoubleConv3D(128, 256)
            
            # Decoder
            self.upconv3d_3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
            self.decoder3d_3 = DoubleConv3D(256, 128)   # 128 + 128 from skip connection
            
            self.upconv3d_2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
            self.decoder3d_2 = DoubleConv3D(128, 64)    # 64 + 64 from skip connection
            
            self.upconv3d_1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
            self.decoder3d_1 = DoubleConv3D(64, 32)    # 32 + 32 from connection
            
        else:
            raise ValueError(f"Unsupported 3D ratio: {ratio_3d}. Supported values: 0.5, 0.25")
        
        # Output layer for 3D branch
        self.output3d_conv = nn.Conv3d(32, num_classes, kernel_size=1)
        
        # Fusion module to combine 2D and 3D outputs
        self.fusion = nn.Sequential(
            nn.Conv3d(num_classes, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        # 2D branch processing
        out_2d = self.conv2d_branch(x)
        
        # 3D branch processing with downsampling
        x_downsampled = F.interpolate(x, size=(self.resolution_3d, self.resolution_3d, self.resolution_3d), 
                                     mode='trilinear', align_corners=False)
        
        if self.ratio_3d == 0.5:
            # Half resolution processing (64x64x64) - 4 encoder + 4 decoder layers
            # 3D Encoder path with skip connections saved
            enc3d_1 = self.encoder3d_1(x_downsampled)  # 64x64x64 -> 32 channels
            x_3d = self.pool3d_1(enc3d_1)              # 32x32x32
            
            enc3d_2 = self.encoder3d_2(x_3d)           # 32x32x32 -> 64 channels
            x_3d = self.pool3d_2(enc3d_2)              # 16x16x16
            
            enc3d_3 = self.encoder3d_3(x_3d)           # 16x16x16 -> 128 channels
            x_3d = self.pool3d_3(enc3d_3)              # 8x8x8
            
            enc3d_4 = self.encoder3d_4(x_3d)           # 8x8x8 -> 256 channels
            x_3d = self.pool3d_4(enc3d_4)              # 4x4x4
            
            # Bottleneck
            x_3d = self.bottleneck3d(x_3d)             # 4x4x4 -> 512 channels
            
            # 3D Decoder path with skip connections
            x_3d = self.upconv3d_4(x_3d)               # 8x8x8 -> 256 channels
            x_3d = torch.cat([x_3d, enc3d_4], dim=1)  # Concatenate with skip connection -> 512 channels
            x_3d = self.decoder3d_4(x_3d)              # 512 -> 256 channels
            
            x_3d = self.upconv3d_3(x_3d)               # 16x16x16 -> 128 channels
            x_3d = torch.cat([x_3d, enc3d_3], dim=1)  # Concatenate with skip connection -> 256 channels
            x_3d = self.decoder3d_3(x_3d)              # 256 -> 128 channels
            
            x_3d = self.upconv3d_2(x_3d)               # 32x32x32 -> 64 channels
            x_3d = torch.cat([x_3d, enc3d_2], dim=1)  # Concatenate with skip connection -> 128 channels
            x_3d = self.decoder3d_2(x_3d)              # 128 -> 64 channels
            
            x_3d = self.upconv3d_1(x_3d)               # 64x64x64 -> 32 channels
            x_3d = torch.cat([x_3d, enc3d_1], dim=1)  # Concatenate with skip connection -> 64 channels
            x_3d = self.decoder3d_1(x_3d)              # 64 -> 32 channels
            
        elif self.ratio_3d == 0.25:
            # Quarter resolution processing (32x32x32) - 3 encoder + 3 decoder layers
            # 3D Encoder path with skip connections saved
            enc3d_1 = self.encoder3d_1(x_downsampled)  # 32x32x32 -> 32 channels
            x_3d = self.pool3d_1(enc3d_1)              # 16x16x16
            
            enc3d_2 = self.encoder3d_2(x_3d)           # 16x16x16 -> 64 channels
            x_3d = self.pool3d_2(enc3d_2)              # 8x8x8
            
            enc3d_3 = self.encoder3d_3(x_3d)           # 8x8x8 -> 128 channels
            x_3d = self.pool3d_3(enc3d_3)              # 4x4x4
            
            # Bottleneck
            x_3d = self.bottleneck3d(x_3d)             # 4x4x4 -> 256 channels
            
            # 3D Decoder path with skip connections
            x_3d = self.upconv3d_3(x_3d)               # 8x8x8 -> 128 channels
            x_3d = torch.cat([x_3d, enc3d_3], dim=1)  # Concatenate with skip connection -> 256 channels
            x_3d = self.decoder3d_3(x_3d)              # 256 -> 128 channels
            
            x_3d = self.upconv3d_2(x_3d)               # 16x16x16 -> 64 channels
            x_3d = torch.cat([x_3d, enc3d_2], dim=1)  # Concatenate with skip connection -> 128 channels
            x_3d = self.decoder3d_2(x_3d)              # 128 -> 64 channels
            
            x_3d = self.upconv3d_1(x_3d)               # 32x32x32 -> 32 channels
            x_3d = torch.cat([x_3d, enc3d_1], dim=1)  # Concatenate with skip connection -> 64 channels
            x_3d = self.decoder3d_1(x_3d)              # 64 -> 32 channels
        
        # Output from 3D branch
        out_3d_low = self.output3d_conv(x_3d)      # 32 -> num_classes channels
        
        # Upsample 3D output to full resolution
        out_3d_upsampled = F.interpolate(out_3d_low, size=(self.resolution, self.resolution, self.resolution), 
                                        mode='trilinear', align_corners=False)
        
        # Simple average instead of concatenation
        combined_features = (out_2d + out_3d_upsampled) / 2.0
        final_output = self.fusion(combined_features)
        
        return final_output

class Transformer1D(nn.Module):
    def __init__(self, seq_len=3, input_dim=128, output_dim=128, d_model=128, nhead=8, num_layers=2):
        super(Transformer1D, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.output_projection(x)
        return x

def compute_iou(pred, target, num_classes, ignore_index=-1):
    """Compute IoU for each class and mean IoU (ignoring ignore_index)"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # ignore 복셀 제외
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # Ignore classes not present
        else:
            ious.append((intersection / union).item())
    
    # Compute mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return mean_iou, ious

def compute_accuracy(pred, target, ignore_index=-1):
    """Compute accuracy (ignoring ignore_index)"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    # ignore 복셀 제외
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    if len(target) == 0:
        return 0.0
    
    correct = (pred == target).sum().item()
    total = len(target)
    return correct / total

def compute_f1_score(pred, target, num_classes, ignore_index=-1):
    """Compute F1-score for each class and macro-averaged F1-score (ignoring ignore_index)"""
    f1_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # ignore 복셀 제외
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # True Positive, False Positive, False Negative 계산
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        
        # Precision과 Recall 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
        
        # F1-score 계산
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1.item())
        else:
            f1_scores.append(float('nan'))  # 클래스가 없을 때
    
    # Macro-averaged F1 계산 (NaN 제외)
    valid_f1s = [f1 for f1 in f1_scores if not np.isnan(f1)]
    macro_f1 = np.mean(valid_f1s) if valid_f1s else 0.0
    
    return macro_f1, f1_scores

def compute_fwiou(pred, target, num_classes, ignore_index=-1):
    """Compute Frequency-Weighted IoU (FWIoU) (ignoring ignore_index)"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    # ignore 복셀 제외
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    if len(target) == 0:
        return 0.0
    
    total_pixels = len(target)
    fwiou = 0.0
    
    for cls in range(num_classes):
        target_cls = (target == cls)
        pred_cls = (pred == cls)
        
        # 클래스 빈도 (frequency)
        class_freq = target_cls.sum().float()
        
        if class_freq > 0:
            # IoU 계산
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union > 0:
                iou = intersection / union
                # 빈도로 가중평균
                fwiou += (class_freq / total_pixels) * iou
    
    return fwiou.item()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_miou = 0.0
    running_f1 = 0.0
    running_fwiou = 0.0
    batch_count = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Training (Epoch {epoch+1})")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            # Compute accuracy (ignoring ignore index)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = compute_accuracy(predicted, targets)
            
            # Compute mean IoU (ignoring ignore index)
            mean_iou, _ = compute_iou(predicted, targets, model.num_classes)
            
            # Compute F1-score (ignoring ignore index)
            macro_f1, _ = compute_f1_score(predicted, targets, model.num_classes)
            
            # Compute FWIoU (ignoring ignore index)
            fwiou = compute_fwiou(predicted, targets, model.num_classes)
            
            running_loss += loss.item()
            running_acc += accuracy
            running_miou += mean_iou
            running_f1 += macro_f1
            running_fwiou += fwiou
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{accuracy:.4f}",
            "mIoU": f"{mean_iou:.4f}",
            "F1": f"{macro_f1:.4f}",
            "FWIoU": f"{fwiou:.4f}",
            "batch": f"{batch_idx+1}/{batch_count}"
        })
    
    epoch_loss = running_loss / batch_count
    epoch_acc = running_acc / batch_count
    epoch_miou = running_miou / batch_count
    epoch_f1 = running_f1 / batch_count
    epoch_fwiou = running_fwiou / batch_count
    
    logger.info(f"Training completed: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
               f"mIoU: {epoch_miou:.4f}, F1: {epoch_f1:.4f}, FWIoU: {epoch_fwiou:.4f}")
    return epoch_loss, epoch_acc, epoch_miou, epoch_f1, epoch_fwiou

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_miou = 0.0
    val_f1 = 0.0
    val_fwiou = 0.0
    class_ious = np.zeros(model.num_classes)
    class_counts = np.zeros(model.num_classes)
    class_f1s = np.zeros(model.num_classes)
    class_f1_counts = np.zeros(model.num_classes)
    
    logger.info("Starting validation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            
            # Compute accuracy (ignoring ignore index)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = compute_accuracy(predicted, targets)
            
            # Compute IoU (ignoring ignore index)
            mean_iou, ious = compute_iou(predicted, targets, model.num_classes)
            
            # Compute F1-score (ignoring ignore index)
            macro_f1, f1_scores = compute_f1_score(predicted, targets, model.num_classes)
            
            # Compute FWIoU (ignoring ignore index)
            fwiou = compute_fwiou(predicted, targets, model.num_classes)
            
            val_loss += loss.item()
            val_acc += accuracy
            val_miou += mean_iou
            val_f1 += macro_f1
            val_fwiou += fwiou
            
            # Accumulate class-wise IoUs
            for i, iou in enumerate(ious):
                if not np.isnan(iou):
                    class_ious[i] += iou
                    class_counts[i] += 1
            
            # Accumulate class-wise F1-scores
            for i, f1 in enumerate(f1_scores):
                if not np.isnan(f1):
                    class_f1s[i] += f1
                    class_f1_counts[i] += 1
    
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_miou /= len(val_loader)
    val_f1 /= len(val_loader)
    val_fwiou /= len(val_loader)
    
    # Compute average class IoUs
    avg_class_ious = []
    for i in range(model.num_classes):
        if class_counts[i] > 0:
            avg_class_ious.append(class_ious[i] / class_counts[i])
        else:
            avg_class_ious.append(0.0)
    
    # Compute average class F1-scores
    avg_class_f1s = []
    for i in range(model.num_classes):
        if class_f1_counts[i] > 0:
            avg_class_f1s.append(class_f1s[i] / class_f1_counts[i])
        else:
            avg_class_f1s.append(0.0)
    
    logger.info(f"Validation: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_miou:.4f}, "
               f"F1: {val_f1:.4f}, FWIoU: {val_fwiou:.4f}")
    
    # Print class-wise metrics
    logger.info("Class-wise IoU:")
    logger.info(f"  Background: {avg_class_ious[0]:.4f}")
    for i, class_name in enumerate(CLASS_LABELS):
        if i + 1 < len(avg_class_ious):
            logger.info(f"  {class_name}: {avg_class_ious[i+1]:.4f}")
    
    logger.info("Class-wise F1-score:")
    logger.info(f"  Background: {avg_class_f1s[0]:.4f}")
    for i, class_name in enumerate(CLASS_LABELS):
        if i + 1 < len(avg_class_f1s):
            logger.info(f"  {class_name}: {avg_class_f1s[i+1]:.4f}")
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'miou': val_miou,
        'f1_score': val_f1,
        'fwiou': val_fwiou,
        'class_ious': avg_class_ious,
        'class_f1s': avg_class_f1s
    }

def main():
    parser = argparse.ArgumentParser(description='3D Semantic Segmentation with 3D Convolution')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0003, 
                       help='Learning rate (default: 0.0003)')
    parser.add_argument('--model_path', type=str, default='models/segmentation_3d.pth', 
                       help='Path to save/load model (default: models/segmentation_3d.pth)')
    parser.add_argument('--train_data', type=str, default='processed_data/train', 
                       help='Training data directory (default: processed_data/train)')
    parser.add_argument('--test_data', type=str, default='processed_data/test', 
                       help='Test data directory (default: processed_data/test)')
    parser.add_argument('--dataset', type=str, choices=['scannet', 'stanford3d'], default='scannet',
                       help='Dataset to use for training/testing (default: scannet)')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run testing (default: False, will run training)')
    parser.add_argument('--pe_type', type=str, choices=['transformer'],
                       help='Positional encoding type: transformer (automatically enables PE)')
    parser.add_argument('--3dratio', type=float, default=0.5, 
                       help='3D resolution ratio for hybrid model (default: 0.5 for half resolution, 0.25 for quarter resolution)')
    
    args = parser.parse_args()
    
    # Determine PE usage
    use_pe = args.pe_type is not None
    pe_type = args.pe_type if args.pe_type is not None else 'transformer'

    if args.pe_type is not None:
        logger.info(f"Positional encoding enabled with type: {pe_type}")
    
    # 데이터셋별 경로 설정 (사용자가 직접 경로를 지정하지 않은 경우에만)
    if args.train_data == 'processed_data/train' and args.test_data == 'processed_data/test':
        train_data_path = f'processed_data/{args.dataset}/train'
        test_data_path = f'processed_data/{args.dataset}/test'
        logger.info(f"Using dataset-specific paths for {args.dataset}")
    else:
        train_data_path = args.train_data
        test_data_path = args.test_data
        logger.info(f"Using custom data paths: train={train_data_path}, test={test_data_path}")
    
    # 모델은 항상 hybrid (our method)
    model_path = f'models/segmentation_hybrid_{args.dataset}.pth'

    # Add PE suffix to model path if PE is enabled
    if use_pe:
        model_path = model_path.replace('.pth', f'_{pe_type}.pth')
    
    # 사용자가 model_path를 지정한 경우 우선적용
    if args.model_path != 'models/segmentation_3d.pth':
        model_path = args.model_path
    
    # 프로그램 시작 시에만 중복 파일명 체크 (한 번만!)
    final_model_path = get_unique_filename(model_path)
    
    # 현재 실행 설정 출력
    logger.info(f"\n{'='*60}")
    logger.info(f"SEGMENTATION TRAINING CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info(f"Model Type: HYBRID")
    ratio_3d = args.__dict__.get('3dratio', 0.5)
    logger.info(f"3D Resolution Ratio: {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)})")
    if use_pe:
        logger.info(f"Positional Encoding: {pe_type}")
    logger.info(f"Mode: {'Test Only' if args.test_only else 'Training + Validation'}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Model Path: {final_model_path}")
    logger.info(f"Train Data: {train_data_path}")
    logger.info(f"Test Data: {test_data_path}")
    logger.info(f"{'='*60}\n")
    
    # 데이터 디렉토리 존재 확인
    if not args.test_only and not os.path.exists(train_data_path):
        logger.error(f"Training data directory not found: {train_data_path}")
        logger.info(f"Please check if the {args.dataset} training data exists or run data preprocessing first.")
        logger.info(f"Example: python prepare_segmentation.py --dataset {args.dataset} --save")
        return
    
    if not os.path.exists(test_data_path):
        logger.error(f"Test data directory not found: {test_data_path}")
        logger.info(f"Please check if the {args.dataset} test data exists or run data preprocessing first.")
        logger.info(f"Example: python prepare_segmentation.py --dataset {args.dataset} --save")
        return
    
    current_seed = seed_everything()  # Use random seed based on current time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load datasets
    train_dataset, test_dataset, train_loader, test_loader, flops_loader = load_datasets(
        train_data=train_data_path,
        test_data=test_data_path,
        batch_size=args.batch_size
    )
    
    # Initialize model (always hybrid - our method)
    model = HybridSegmentationNet(
        in_channels=4,  # occupancy + RGB
        num_classes=train_dataset.num_classes,
        resolution=train_dataset.resolution,
        use_pe=use_pe,
        pe_type=pe_type,
        ratio_3d=args.__dict__.get('3dratio', 0.5)
    ).to(device)

    # Calculate model complexity
    input_shape = (1, 4, train_dataset.resolution, train_dataset.resolution, train_dataset.resolution)
    flops, params = calculate_model_complexity(model, input_shape, device)

    if flops is not None and params is not None:
        logger.info(f"Model GFLOPs: {flops / 1e9:.2f}")
        logger.info(f"Model Parameters: {params / 1e6:.2f} M")
    
    # Setup training with class weights to handle imbalance
    # Calculate class weights based on inverse frequency (silently)
    
    # Sample a few files to calculate class distribution
    sample_files = train_dataset.file_list[:min(10, len(train_dataset.file_list))]
    class_counts = torch.zeros(train_dataset.num_classes)
    total_samples = 0
    
    for filename in sample_files:
        file_path = os.path.join(train_dataset.data_dir, filename)
        data = torch.load(file_path, weights_only=False)
        
        # 두 가지 키 형식 지원
        if 'labels' in data:
            # 새로운 형식 사용
            label_grid = data['labels']
        else:
            # 기존 형식 사용 (하위 호환성)
            label_grid = data['label_grid']
        
        # numpy array를 tensor로 변환 (필요한 경우)
        if isinstance(label_grid, np.ndarray):
            label_grid = torch.from_numpy(label_grid)
        
        # ignore(-1) 복셀을 제외하고 클래스 카운트
        valid_mask = (label_grid != -1)
        valid_labels = label_grid[valid_mask]
        
        for class_id in range(train_dataset.num_classes):
            class_counts[class_id] += (valid_labels == class_id).sum().item()
        total_samples += valid_labels.numel()
    
    # Calculate weights (inverse frequency, smoothed) - 더 강력한 가중치 적용
    class_weights = torch.zeros(train_dataset.num_classes)
    for i in range(train_dataset.num_classes):
        if class_counts[i] > 0:
            # 더 강력한 역빈도 가중치 (제곱근 적용으로 완화)
            class_weights[i] = np.sqrt(total_samples / (train_dataset.num_classes * class_counts[i]))
        else:
            class_weights[i] = 1.0
    
    # 배경 클래스 가중치를 더 낮게 설정
    class_weights[0] = class_weights[0] * 0.1  # 배경 가중치 대폭 감소
    
    # Apply smoothing and normalization
    class_weights = torch.clamp(class_weights, min=0.01, max=50.0)  # 더 넓은 범위
    class_weights = class_weights / class_weights.sum() * train_dataset.num_classes  # Normalize
    
    # Use Combined Loss (CrossEntropy + Dice) with class weights
    criterion = CombinedLoss(
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=class_weights.to(device),
        ignore_index=-1
    )

    # Better optimizer settings for segmentation
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    
    # CosineAnnealingWarmRestarts scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=args.lr*0.01
    )
    
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    if args.test_only:
        # Load model and test
        if os.path.exists(final_model_path):
            model.load_state_dict(torch.load(final_model_path, weights_only=False))  # PyTorch 2.6 호환성
            logger.info(f"Model loaded from {final_model_path}")
        else:
            logger.error(f"Model file not found: {final_model_path}")
            return
        
        test_results = validate(model, test_loader, criterion, device)
        logger.info("Testing completed!")
        return
    
    # Training loop
    best_miou = 0.0
    best_epoch = 0
    
    # Setup result logging
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create detailed results file with header
    detailed_result_file = os.path.join(results_dir, "detailed_segmentation.txt")
    
    # Check if detailed file exists to determine if this is the first run
    file_exists = os.path.exists(detailed_result_file)
    
    if not file_exists:
        # Create detailed results file with header
        with open(detailed_result_file, 'w') as f:
            f.write(f"3D Semantic Segmentation Training Results\n")
            f.write(f"{'='*50}\n\n")
    
    # Add model section to detailed file
    with open(detailed_result_file, 'a') as f:
        # Determine model name for display
        model_name = 'Hybrid 2D+3D Segmentation'

        # Add PE information if enabled
        if use_pe:
            model_name = f'{model_name} (PE: {pe_type})'

        f.write(f"Dataset: {args.dataset.upper()}\n")
        f.write(f"Model: HYBRID ({model_name})\n")
        ratio_3d = args.__dict__.get('3dratio', 0.5)
        f.write(f"3D Resolution Ratio: {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)})\n")
        f.write(f"{'='*30}\n")
        f.write(f"Training Data: {train_data_path}\n")
        f.write(f"Test Data: {test_data_path}\n")
        f.write(f"Classes: {train_dataset.num_classes}\n")
        f.write(f"Resolution: {train_dataset.resolution}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Total Epochs: {args.epochs}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*30}\n\n")

    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc, train_miou, train_f1, train_fwiou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)

        val_results = validate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        is_best = val_results['miou'] > best_miou
        if is_best:
            best_miou = val_results['miou']
            best_epoch = epoch
            # 같은 파일에 덮어쓰기 (프로그램 시작 시 결정된 파일명 사용)
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"New best model saved to: {final_model_path} | mIoU: {best_miou:.4f}")
        
        # Log detailed results only
        with open(detailed_result_file, 'a') as f:
            f.write(f"Epoch {epoch+1:3d}: Train(Loss={train_loss:.6f}, Acc={train_acc:.6f}, mIoU={train_miou:.6f}, F1={train_f1:.6f}, FWIoU={train_fwiou:.6f}) | ")
            f.write(f"Val(Loss={val_results['loss']:.6f}, Acc={val_results['accuracy']:.6f}, mIoU={val_results['miou']:.6f}, ")
            f.write(f"F1={val_results['f1_score']:.6f}, FWIoU={val_results['fwiou']:.6f})")
            if is_best:
                f.write(" ⭐ BEST")
            f.write("\n")
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch+1}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Detailed results saved to: {detailed_result_file}")
    
    # Save final summary to detailed file
    with open(detailed_result_file, 'a') as f:
        f.write(f"\nFinal Results for {args.dataset.upper()} - HYBRID:\n")
        f.write(f"  Best mIoU: {best_miou:.4f} (Epoch {best_epoch+1})\n")
        f.write(f"  Model saved to: {model_path}\n")
        f.write(f"  Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")

if __name__ == "__main__":
    main() 