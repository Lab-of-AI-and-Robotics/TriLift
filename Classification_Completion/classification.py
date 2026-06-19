import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Try to import thop for FLOPs calculation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop is not installed. FLOPs calculation will be skipped.")
    print("To install thop, run: pip install thop")

# Import necessary ModelNet40H5 and CoordinateTransformation classes
from utils.pointnet import ModelNet40H5, CoordinateTransformation

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set seed for reproducibility
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

# Preprocessed Occupancy Grid dataset
class OccupancyGridFileDataset(Dataset):
    """
    Dataset that loads occupancy grid data stored as individual files
    """
    def __init__(self, data_dir, ratio=1.0):
        self.data_dir = data_dir
        
        # Load file list
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        
        # Apply ratio to reduce dataset size
        num_files = int(len(all_files) * ratio)
        self.file_list = all_files[:num_files]
        
        # Load metadata
        meta_file = os.path.join(data_dir, "metadata.txt")
        
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                lines = f.readlines()
                metadata = {}
                for line in lines:
                    key, value = line.strip().split(': ')
                    metadata[key] = value
                
                self.resolution = int(metadata.get('resolution', 128))
                self.binary_voxels = metadata.get('binary_voxels', 'true') == 'true'
                self.gap_filling = metadata.get('gap_filling', 'enabled') == 'enabled'
        else:
            # Use default values if metadata file doesn't exist
            self.resolution = 128
            self.binary_voxels = True
            self.gap_filling = True
        
        print(f"Binary occupancy grid dataset loaded from: {data_dir}")
        print(f"  Total files available: {len(all_files)}")
        print(f"  Files used (ratio {ratio:.1f}): {len(self.file_list)}")
        print(f"  Resolution: {self.resolution}x{self.resolution}x{self.resolution}")
        print(f"  Binary voxels: {self.binary_voxels}")
        print(f"  Gap filling: {self.gap_filling}")
        print(f"  Individual file processing enabled: samples stored in separate files")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # File path
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        
        # Load data
        data = torch.load(file_path)
        
        # Check if label is a scalar and process it
        label = data['label']
        if isinstance(label, torch.Tensor):
            # If tensor is not a scalar, use only the first element
            if label.dim() > 0:
                label = label.item()
            # If tensor is a scalar, extract with item()
            else:
                label = label.item()
        
        # Convert to Long tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            "occupancy_grid": data['occupancy_grid'],
            "label": label
        }

class OccupancyGridH5Dataset(Dataset):
    """
    Dataset that loads preprocessed occupancy grid data from HDF5 files
    """
    def __init__(self, h5_file, ratio=1.0):
        self.h5_file = h5_file
        
        # Load HDF5 file metadata
        with h5py.File(h5_file, 'r') as f:
            self.resolution = f.attrs['resolution']
            total_samples = f.attrs['samples']
            # Handle both old and new metadata formats
            self.binary_voxels = f.attrs.get('binary_voxels', True)
            self.gap_filling = f.attrs.get('gap_filling', True)
        
        # Apply ratio to reduce dataset size
        self.num_samples = int(total_samples * ratio)
            
        print(f"Binary occupancy grid dataset loaded from: {h5_file}")
        print(f"  Total samples available: {total_samples}")
        print(f"  Samples used (ratio {ratio:.1f}): {self.num_samples}")
        print(f"  Resolution: {self.resolution}x{self.resolution}x{self.resolution}")
        print(f"  Binary voxels: {self.binary_voxels}")
        print(f"  Gap filling: {self.gap_filling}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load data
        with h5py.File(self.h5_file, 'r') as f:
            # Load grid data
            occupancy_grid = torch.from_numpy(f['occupancy_grids'][idx]).float()
            # Load label data
            label = torch.tensor(f['labels'][idx], dtype=torch.long)
            
        return {
            "occupancy_grid": occupancy_grid,
            "label": label
        }

def load_datasets(train_data, test_data, val_data=None, batch_size=16, ratio=1.0):
    """
    Load preprocessed data (HDF5 files or directories)
    """
    print("Loading preprocessed datasets...")
    
    # Check if data is a file or directory and use appropriate dataset class
    if os.path.isdir(train_data):
        print(f"Loading training data from directory: {train_data} (individual file processing)")
        train_dataset = OccupancyGridFileDataset(train_data, ratio=ratio)
    else:
        print(f"Loading training data from HDF5 file: {train_data}")
        train_dataset = OccupancyGridH5Dataset(train_data, ratio=ratio)
    
    # Load validation dataset (if provided)
    val_dataset = None
    if val_data:
        if os.path.isdir(val_data):
            print(f"Loading validation data from directory: {val_data} (individual file processing)")
            val_dataset = OccupancyGridFileDataset(val_data, ratio=ratio)
        else:
            print(f"Loading validation data from HDF5 file: {val_data}")
            val_dataset = OccupancyGridH5Dataset(val_data, ratio=ratio)
    
    # Load test dataset
    if os.path.isdir(test_data):
        print(f"Loading test data from directory: {test_data} (individual file processing)")
        test_dataset = OccupancyGridFileDataset(test_data, ratio=ratio)
    else:
        print(f"Loading test data from HDF5 file: {test_data}")
        test_dataset = OccupancyGridH5Dataset(test_data, ratio=ratio)
    
    print("\nCreating data loaders...")
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    flops_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Data loading complete. Batch size: {batch_size}")
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, flops_loader

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

# 2D Projection-based Classification model with PE support
class Conv2DClassificationNet(nn.Module):
    """
    2D Projection-based neural network for classification
    Projects 3D volumes onto 2D planes along each axis and applies 2D convolutions
    """
    def __init__(self, num_classes=40, input_channels=1, resolution=128, use_pe=False, pe_type='transformer'):
        super(Conv2DClassificationNet, self).__init__()
        self.resolution = resolution
        self.use_pe = use_pe
        self.pe_type = pe_type

        if self.use_pe:
            if self.pe_type == 'transformer':
                self._create_transformer_pe_layers(resolution)
        
        # 2D CNN for X-axis projection
        self.features_x = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 2D CNN for Y-axis projection
        self.features_y = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 2D CNN for Z-axis projection
        self.features_z = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier - concatenates features from each axis
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 3, 1024),  # Features from 3 directions concatenated
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def _create_transformer_pe_layers(self, resolution):
        self.pe_transformer = Transformer1D(seq_len=3, input_dim=resolution, output_dim=resolution)
        self._cached_transformer_weights = None

    def _apply_positional_encoding(self, x, axis):
        """Apply positional encoding during projection"""
        if not self.use_pe:
            return x
            
        if self.pe_type == 'transformer':
            if not hasattr(self, '_cached_transformer_weights'):
                self._cached_transformer_weights = self._apply_transformer_pe(x)
            
            weights = self._cached_transformer_weights
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
        """Apply transformer-based positional encoding"""
        batch_size = x.size(0)
        volume = x.squeeze(1)
        
        x_vec = volume.sum(dim=(2, 3))
        y_vec = volume.sum(dim=(1, 3))
        z_vec = volume.sum(dim=(1, 2))
        
        transformer_input = torch.stack([x_vec, y_vec, z_vec], dim=1)
        transformer_weights = self.pe_transformer(transformer_input)
        
        weight_x = transformer_weights[:, 0, :self.resolution]
        weight_y = transformer_weights[:, 1, :self.resolution]
        weight_z = transformer_weights[:, 2, :self.resolution]
        
        return {
            'weight_x': weight_x,
            'weight_y': weight_y, 
            'weight_z': weight_z
        }

    def project_3d_to_2d_with_pe(self, x):
        """Convert 3D input to three 2D projections with positional encoding"""
        if hasattr(self, '_cached_transformer_weights'):
            delattr(self, '_cached_transformer_weights')
        
        if self.use_pe:
            x_with_pe_x = self._apply_positional_encoding(x, axis=0)
            x_with_pe_y = self._apply_positional_encoding(x, axis=1)
            x_with_pe_z = self._apply_positional_encoding(x, axis=2)
            
            proj_x = torch.mean(x_with_pe_x, dim=2)  # [B, C, H, W]
            proj_y = torch.mean(x_with_pe_y, dim=3)  # [B, C, D, W]
            proj_z = torch.mean(x_with_pe_z, dim=4)  # [B, C, D, H]
        else:
            proj_x = torch.mean(x, dim=2)  # [B, C, H, W]
            proj_y = torch.mean(x, dim=3)  # [B, C, D, W]
            proj_z = torch.mean(x, dim=4)  # [B, C, D, H]
        
        return proj_x, proj_y, proj_z

    def forward(self, x):
        # Convert 3D input to three 2D projections with PE support
        proj_x, proj_y, proj_z = self.project_3d_to_2d_with_pe(x)
        
        # Apply 2D CNN to each projection
        feat_x = self.features_x(proj_x)
        feat_y = self.features_y(proj_y)
        feat_z = self.features_z(proj_z)
        
        # Flatten
        feat_x = feat_x.view(feat_x.size(0), -1)
        feat_y = feat_y.view(feat_y.size(0), -1)
        feat_z = feat_z.view(feat_z.size(0), -1)
        
        # Concatenate features
        combined_feat = torch.cat([feat_x, feat_y, feat_z], dim=1)
        # Apply classifier
        output = self.classifier(combined_feat)
        
        return output

# Hybrid 2D+3D Classification model
class HybridClassifier(nn.Module):
    """
    Hybrid neural network that combines 2D projection and 3D convolution for classification
    Uses configurable resolution 3D processing and full-resolution 2D processing, then combines them
    """
    def __init__(self, num_classes=40, input_channels=1, use_pe=False, pe_type='transformer', ratio_3d=0.5):
        super(HybridClassifier, self).__init__()
        self.resolution = 128  # Assuming standard resolution
        self.ratio_3d = ratio_3d
        self.resolution_3d = int(self.resolution * ratio_3d)
        
        # 2D projection branch - use Conv2DClassificationNet for PE support
        self.conv2d_classifier_net = Conv2DClassificationNet(
            num_classes=num_classes, 
            input_channels=input_channels, 
            resolution=128, 
            use_pe=use_pe, 
            pe_type=pe_type
        )
        
        # 3D convolution branch with configurable resolution
        if ratio_3d == 0.5:
            # Half resolution (64x64x64) - 4 conv layers
            self.conv3d_features = nn.Sequential(
                nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                nn.Conv3d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )
            
            self.conv3d_classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )
            
        elif ratio_3d == 0.25:
            # Quarter resolution (32x32x32) - 3 conv layers
            self.conv3d_features = nn.Sequential(
                nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )
            
            self.conv3d_classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )
            
        else:
            raise ValueError(f"Unsupported 3D ratio: {ratio_3d}. Supported values: 0.5, 0.25")
        
        # Fusion module to combine 2D and 3D outputs
        self.fusion = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # 2D branch processing with PE support
        out_2d = self.conv2d_classifier_net(x)  # Get classification output directly
        
        # 3D branch processing with downsampling
        x_downsampled = nn.functional.interpolate(x, size=(self.resolution_3d, self.resolution_3d, self.resolution_3d), mode='trilinear', align_corners=False)
        
        # Process through 3D features
        x_3d_feat = self.conv3d_features(x_downsampled)
        x_3d_feat = x_3d_feat.view(x_3d_feat.size(0), -1)
        out_3d_low = self.conv3d_classifier(x_3d_feat)
        
        # Combine outputs from both branches - simple average
        combined_features = (out_2d + out_3d_low) / 2.0
        final_output = self.fusion(combined_features)
        
        return final_output

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(train_loader)
    
    # Progress bar with more details
    pbar = tqdm(train_loader, desc=f"Training (hybrid)")
    for batch_idx, batch in enumerate(pbar):
        inputs = batch["occupancy_grid"].to(device)
        inputs = inputs.unsqueeze(1)  # Convert to [B, 1, D, H, W] format
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * outputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar with batch information
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "acc": f"{100. * correct / total:.2f}%",
            "batch": f"{batch_idx+1}/{batch_count}"
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Training completed: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

# Testing function
def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    batch_count = len(test_loader)
    
    print(f"Starting evaluation on test set (hybrid)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing (hybrid)")):
            inputs = batch["occupancy_grid"].to(device)
            inputs = inputs.unsqueeze(1)  # Convert to [B, 1, D, H, W] format
            labels = batch["label"].to(device)
            outputs = model(inputs)
            
            # Save predictions
            probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
            _, predicted = outputs.max(1)
            
            # Move to CPU and convert to numpy array
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print("Computing evaluation metrics...")
    
    # 1. Top-1 Accuracy (standard accuracy)
    top1_accuracy = accuracy_score(all_labels, all_preds)
    
    # 2. Weighted Accuracy (accounts for class imbalance)
    # Calculate class frequencies
    class_counts = np.bincount(all_labels)
    num_classes = len(class_counts)
    class_frequencies = class_counts / np.sum(class_counts)
    
    # Calculate per-class accuracies
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    
    for i in range(len(all_labels)):
        true_label = all_labels[i]
        per_class_total[true_label] += 1
        if all_preds[i] == true_label:
            per_class_correct[true_label] += 1
    
    # Calculate per-class accuracy (avoid division by zero)
    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_acc[i] = per_class_correct[i] / per_class_total[i]
    
    # Weighted accuracy is the average of per-class accuracies (balanced)
    # This gives equal importance to each class, regardless of sample count
    weighted_accuracy = np.mean(per_class_acc[per_class_total > 0])
    
    # 3. F1-Score (harmonic mean of Precision and Recall)
    # macro: average of F1-Score for each class
    # weighted: F1-Score weighted by number of samples in each class
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 4. ROC AUC Score (multiclass - One vs Rest approach)
    # For multiclass, use One-vs-Rest approach
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError as e:
        # Handle exception if some classes aren't in the test set
        print(f"Warning: ROC AUC calculation error: {e}")
        print("Some classes may not be present in the test set.")
        roc_auc = 0.0
    
    # Calculate confusion matrix (for debugging and detailed analysis)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print sample counts for each class
    print("\nClass distribution in test set:")
    for i in range(num_classes):
        if class_counts[i] > 0:
            print(f"  Class {i}: {class_counts[i]} samples ({class_frequencies[i]*100:.2f}%)")
    
    # Store results
    result = {
        'top1_accuracy': top1_accuracy * 100.0,  # Convert to percentage
        'weighted_accuracy': weighted_accuracy * 100.0,
        'macro_f1': macro_f1 * 100.0,
        'weighted_f1': weighted_f1 * 100.0,
        'roc_auc': roc_auc * 100.0,
        'confusion_matrix': cm,
        'class_counts': class_counts,
        'per_class_acc': per_class_acc * 100.0  # Convert to percentage
    }
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        if class_counts[i] > 0:
            print(f"  Class {i}: {per_class_acc[i]*100:.2f}%")
    
    return result

# Main function
def main():
    parser = argparse.ArgumentParser(description='3D Conv Classifier Training and Testing')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='models/model_best.pth', help='Path to save/load model')
    parser.add_argument('--pe_type', type=str, choices=['transformer'], help='Positional encoding type (transformer only; automatically enables PE)')
    parser.add_argument('--3dratio', type=float, default=0.5, 
                       help='3D resolution ratio for hybrid model (default: 0.5 for half resolution, 0.25 for quarter resolution)')
    parser.add_argument('--train_data', type=str, default='processed_data/classification_train_res128_binary', help='Training data file or directory')
    parser.add_argument('--test_data', type=str, default='processed_data/classification_test_res128_binary', help='Test data file or directory')
    parser.add_argument('--val_data', type=str, default=None, help='Validation data file or directory (None if not used)')
    parser.add_argument('--ratio', type=float, default=0.4, help='Ratio of data to use for training/testing ')
    args = parser.parse_args()

    # Fix seed for reproducibility
    current_seed = seed_everything()  # Use random seed based on current time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set model path (hybrid is the only supported model)
    model_path = 'models/hybrid_classifier_best.pth'
    ratio_3d = args.__dict__.get('3dratio', 0.5)
    print(f"Using hybrid 2D+3D convolution model with 3D ratio {ratio_3d} ({int(128 * ratio_3d)}×{int(128 * ratio_3d)}×{int(128 * ratio_3d)}).")

    # User specified model path
    if args.model_path != 'models/model_best.pth':
        model_path = args.model_path

    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, flops_loader = load_datasets(
        train_data=args.train_data,
        test_data=args.test_data,
        val_data=args.val_data,
        batch_size=args.batch_size,
        ratio=args.ratio
    )

    # Handle PE options (like in completion_3d.py)
    use_pe = args.pe_type is not None
    pe_type = args.pe_type if args.pe_type is not None else 'transformer'

    if args.pe_type is not None:
        print(f"Positional encoding enabled with type: {pe_type}")

    # Initialize the hybrid model (our method)
    model = HybridClassifier(num_classes=40, input_channels=1, use_pe=use_pe, pe_type=pe_type, ratio_3d=args.__dict__.get('3dratio', 0.5)).to(device)

    # Calculate FLOPs for the model
    print(f"\n{'='*50}")
    print(f"MODEL COMPLEXITY ANALYSIS")
    print(f"{'='*50}")
    
    # Create dummy input for FLOPs calculation (batch_size=1 for single iteration analysis)
    dummy_input = torch.randn(1, 1, 128, 128, 128).to(device)
    
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
        
        # Calculate parameters manually
        total_params = sum(p.numel() for p in model.parameters())
        params_m = total_params / 1e6
        print(f"Parameters: {params_m:.2f} M ({total_params:,} params)")
        print(f"FLOPs calculation skipped (thop not installed)")
        print(f"{'='*50}\n")

    # Create Results directory (if it doesn't exist)
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create detailed results file for epoch summaries
    detailed_result_file = os.path.join(results_dir, "detailed_classification.txt")
    
    # Check if detailed file exists to determine if this is the first run
    file_exists = os.path.exists(detailed_result_file)
    
    # Write headers if files don't exist
    if not file_exists:
        with open(detailed_result_file, 'w') as f:
            f.write(f"3D Classification Training Results\n")
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
        f.write(f"Data Ratio: {args.ratio}\n")
        f.write(f"Total Epochs: {args.epochs}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*30}\n\n")
    
    # Start training
    print(f"Starting training... Model type: hybrid")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
        # Perform training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on test set
        test_result = test(model, test_loader, device)
        test_acc = test_result['top1_accuracy']
        
        # Print and save detailed results for each epoch
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Training Loss: {train_loss:.6f}, Training Accuracy: {train_acc:.2f}%")
        print(f"  Test Top-1 Accuracy: {test_result['top1_accuracy']:.2f}%")
        print(f"  Test Weighted Accuracy: {test_result['weighted_accuracy']:.2f}%")
        print(f"  Test Macro F1 Score: {test_result['macro_f1']:.2f}%")
        print(f"  Test Weighted F1 Score: {test_result['weighted_f1']:.2f}%")
        print(f"  Test ROC AUC: {test_result['roc_auc']:.2f}%")
        
        # Check if this is the best model
        is_best = False
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"\n🔥 New best model saved: {test_acc:.2f}% accuracy 🔥")
            is_best = True
        
        # Save detailed epoch results only
        with open(detailed_result_file, 'a') as f:
            f.write(f"Epoch {epoch+1:3d}: ")
            f.write(f"Train(Loss={train_loss:.6f}, Acc={train_acc:.2f}%) | ")
            f.write(f"Test(Top1={test_result['top1_accuracy']:.2f}%, Weighted={test_result['weighted_accuracy']:.2f}%, ")
            f.write(f"F1={test_result['macro_f1']:.2f}%, AUC={test_result['roc_auc']:.2f}%)")
            if is_best:
                f.write(" ⭐ BEST")
            f.write("\n")
        
        print(f"Detailed results saved to {detailed_result_file}")
        if epoch % 10 == 0 or is_best:  # Print file location every 10 epochs or when best
            print(f"Detailed results saved to {detailed_result_file}")

    print(f"\n{'='*20} FINAL EVALUATION {'='*20}")
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    final_test_result = test(model, test_loader, device)
    
    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"\nFinal test performance:")
    print(f"  Top-1 accuracy: {final_test_result['top1_accuracy']:.2f}%")
    print(f"  Weighted accuracy: {final_test_result['weighted_accuracy']:.2f}%")
    print(f"  Macro F1 score: {final_test_result['macro_f1']:.2f}%")
    print(f"  Weighted F1 score: {final_test_result['weighted_f1']:.2f}%")
    print(f"  ROC AUC: {final_test_result['roc_auc']:.2f}%")
    print(f"\nModel saved to: {model_path}")
    print(f"Detailed results saved to: {detailed_result_file}")
    
    # Save final summary to detailed file
    with open(detailed_result_file, 'a') as f:
        f.write(f"\nFinal Results for HYBRID:\n")
        f.write(f"  Best Top-1 Accuracy: {best_test_acc:.2f}%\n")
        f.write(f"  Final Top-1 Accuracy: {final_test_result['top1_accuracy']:.2f}%\n")
        f.write(f"  Final Weighted Accuracy: {final_test_result['weighted_accuracy']:.2f}%\n")
        f.write(f"  Final Macro F1 Score: {final_test_result['macro_f1']:.2f}%\n")
        f.write(f"  Final ROC AUC: {final_test_result['roc_auc']:.2f}%\n")
        f.write(f"  Model saved to: {model_path}\n")
        f.write(f"  Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")

if __name__ == "__main__":
    main() 