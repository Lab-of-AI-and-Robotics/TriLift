#!/usr/bin/env python
import os
import torch
import numpy as np
import rerun as rr
from pathlib import Path
import argparse

# ScanNet 클래스 라벨 및 색상 정의
CLASS_LABELS = ['background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

# ScanNet 색상 매핑 (RGB 0-255)
SCANNET_COLOR_MAP = {
    0: [0, 0, 0],           # background - black
    1: [174, 199, 232],     # wall - light blue
    2: [152, 223, 138],     # floor - light green
    3: [31, 119, 180],      # cabinet - blue
    4: [255, 187, 120],     # bed - orange
    5: [188, 189, 34],      # chair - yellow-green
    6: [140, 86, 75],       # sofa - brown
    7: [255, 152, 150],     # table - pink
    8: [214, 39, 40],       # door - red
    9: [197, 176, 213],     # window - light purple
    10: [148, 103, 189],    # bookshelf - purple
    11: [196, 156, 148],    # picture - beige
    12: [23, 190, 207],     # counter - cyan
    13: [247, 182, 210],    # desk - light pink
    14: [219, 219, 141],    # curtain - light yellow
    15: [255, 127, 14],     # refrigerator - orange
    16: [158, 218, 229],    # shower curtain - light cyan
    17: [44, 160, 44],      # toilet - green
    18: [112, 128, 144],    # sink - gray
    19: [227, 119, 194],    # bathtub - magenta
    20: [82, 84, 163],      # otherfurniture - dark blue
}

def load_voxel_data(file_path):
    """복셀 데이터 로드"""
    print(f"Loading data from: {file_path}")
    data = torch.load(file_path, weights_only=False)
    
    occupancy_grid = data['occupancy_grid'].numpy()
    color_grid = data['color_grid'].numpy()
    label_grid = data['label_grid'].numpy()
    
    filename = data.get('filename', os.path.basename(file_path))
    original_points = data.get('original_points', 'unknown')
    voxel_occupancy = data.get('voxel_occupancy', np.sum(occupancy_grid) / occupancy_grid.size)
    gap_filled = data.get('gap_filled', False)
    
    print(f"  Filename: {filename}")
    print(f"  Original points: {original_points}")
    print(f"  Voxel occupancy: {voxel_occupancy:.2%}")
    print(f"  Gap filled: {gap_filled}")
    print(f"  Occupancy shape: {occupancy_grid.shape}")
    print(f"  Color shape: {color_grid.shape}")
    print(f"  Label shape: {label_grid.shape}")
    
    return {
        'occupancy': occupancy_grid,
        'colors': color_grid,
        'labels': label_grid,
        'filename': filename,
        'original_points': original_points,
        'voxel_occupancy': voxel_occupancy,
        'gap_filled': gap_filled
    }

def voxel_to_points(occupancy_grid, color_grid, label_grid):
    """복셀 그리드를 포인트 클라우드로 변환"""
    # 점유된 복셀의 인덱스 찾기
    occupied_indices = np.where(occupancy_grid)
    
    if len(occupied_indices[0]) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 3D 좌표 생성
    points = np.column_stack(occupied_indices).astype(np.float32)
    
    # 색상 추출 (점유된 복셀의 색상)
    colors = color_grid[occupied_indices]
    
    # 라벨 추출
    labels = label_grid[occupied_indices]
    
    return points, colors, labels

def get_label_colors(labels):
    """라벨에 따른 시각화 색상 반환"""
    label_colors = np.zeros((len(labels), 3), dtype=np.uint8)
    
    for i, label in enumerate(labels):
        if label in SCANNET_COLOR_MAP:
            label_colors[i] = SCANNET_COLOR_MAP[label]
        else:
            label_colors[i] = [128, 128, 128]  # 기본 회색
    
    return label_colors

def visualize_voxel_data(data, entity_prefix=""):
    """Rerun으로 복셀 데이터 시각화"""
    occupancy = data['occupancy']
    colors = data['colors']
    labels = data['labels']
    filename = data['filename']
    
    print(f"\nVisualizing {filename}...")
    
    # 복셀을 포인트로 변환
    points, point_colors, point_labels = voxel_to_points(occupancy, colors, labels)
    
    if len(points) == 0:
        print("  No occupied voxels found!")
        return
    
    print(f"  Converted to {len(points)} points")
    
    # 라벨별 색상 생성
    label_colors = get_label_colors(point_labels)
    
    # 1. 원본 색상으로 포인트 클라우드 시각화
    rr.log(
        f"{entity_prefix}/original_colors",
        rr.Points3D(
            positions=points,
            colors=(point_colors * 255).astype(np.uint8),  # 0-1 범위를 0-255로 변환
            radii=0.5
        )
    )
    
    # 2. 라벨 색상으로 포인트 클라우드 시각화
    rr.log(
        f"{entity_prefix}/segmentation_labels",
        rr.Points3D(
            positions=points,
            colors=label_colors,
            radii=0.5
        )
    )
    
    # 3. 라벨별 분리 시각화
    unique_labels = np.unique(point_labels)
    print(f"  Unique labels: {len(unique_labels)}")
    
    for label in unique_labels:
        if label == -1:  # ignore 라벨은 건너뛰기
            continue
            
        mask = point_labels == label
        if np.sum(mask) == 0:
            continue
            
        label_points = points[mask]
        label_name = CLASS_LABELS[label] if 0 <= label < len(CLASS_LABELS) else f"unknown_{label}"
        color = SCANNET_COLOR_MAP.get(label, [128, 128, 128])
        
        rr.log(
            f"{entity_prefix}/by_class/{label:02d}_{label_name}",
            rr.Points3D(
                positions=label_points,
                colors=[color] * len(label_points),
                radii=0.5
            )
        )
        
        print(f"    {label:2d} ({label_name:15s}): {len(label_points):5d} points")
    
    # 4. 메타데이터 로깅
    rr.log(
        f"{entity_prefix}/metadata",
        rr.TextLog(
            f"Filename: {filename}\n"
            f"Original points: {data['original_points']}\n"
            f"Voxel occupancy: {data['voxel_occupancy']:.2%}\n"
            f"Gap filled: {data['gap_filled']}\n"
            f"Total voxel points: {len(points)}\n"
            f"Unique labels: {len(unique_labels)}"
        )
    )

def main():
    parser = argparse.ArgumentParser(description='Visualize processed ScanNet voxel data with Rerun')
    parser.add_argument('--train-dir', type=str, 
                       default='processed_data/segmentation_scannet_train_res128_gap_filled_sample1pct',
                       help='Train data directory')
    parser.add_argument('--test-dir', type=str,
                       default='processed_data/segmentation_scannet_test_res128_gap_filled_sample1pct', 
                       help='Test data directory')
    parser.add_argument('--train-file', type=str, default=None, help='Specific train file to visualize')
    parser.add_argument('--test-file', type=str, default=None, help='Specific test file to visualize')
    
    args = parser.parse_args()
    
    # Rerun 초기화
    rr.init("ScanNet_Voxel_Visualization", spawn=True)
    
    print("🎨 ScanNet 복셀 데이터 시각화 (Rerun)")
    print("=" * 50)
    
    # Train 데이터 시각화
    train_dir = Path(args.train_dir)
    if train_dir.exists():
        train_files = list(train_dir.glob("*.pt"))
        train_files = [f for f in train_files if not f.name.startswith('metadata')]
        
        if train_files:
            if args.train_file:
                # 특정 파일 지정
                train_file = train_dir / args.train_file
                if not train_file.exists():
                    train_file = train_dir / f"{args.train_file}.pt"
            else:
                # 첫 번째 파일 선택
                train_file = train_files[0]
            
            if train_file.exists():
                print(f"\n📊 Train 데이터 시각화")
                train_data = load_voxel_data(train_file)
                visualize_voxel_data(train_data, "train")
            else:
                print(f"❌ Train 파일을 찾을 수 없습니다: {train_file}")
        else:
            print(f"❌ Train 디렉토리에 .pt 파일이 없습니다: {train_dir}")
    else:
        print(f"❌ Train 디렉토리를 찾을 수 없습니다: {train_dir}")
    
    # Test 데이터 시각화
    test_dir = Path(args.test_dir)
    if test_dir.exists():
        test_files = list(test_dir.glob("*.pt"))
        test_files = [f for f in test_files if not f.name.startswith('metadata')]
        
        if test_files:
            if args.test_file:
                # 특정 파일 지정
                test_file = test_dir / args.test_file
                if not test_file.exists():
                    test_file = test_dir / f"{args.test_file}.pt"
            else:
                # 첫 번째 파일 선택
                test_file = test_files[0]
            
            if test_file.exists():
                print(f"\n📊 Test 데이터 시각화")
                test_data = load_voxel_data(test_file)
                visualize_voxel_data(test_data, "test")
            else:
                print(f"❌ Test 파일을 찾을 수 없습니다: {test_file}")
        else:
            print(f"❌ Test 디렉토리에 .pt 파일이 없습니다: {test_dir}")
    else:
        print(f"❌ Test 디렉토리를 찾을 수 없습니다: {test_dir}")
    
    print(f"\n✅ 시각화 완료!")
    print(f"🌐 Rerun 뷰어에서 다음 항목들을 확인할 수 있습니다:")
    print(f"  - train/original_colors: 원본 RGB 색상")
    print(f"  - train/segmentation_labels: 세그멘테이션 라벨 색상")
    print(f"  - train/by_class/: 클래스별 분리된 포인트들")
    print(f"  - test/: 테스트 데이터 (동일한 구조)")
    print(f"  - metadata: 데이터 정보")
    
    # 사용 가능한 파일 목록 출력
    if train_dir.exists():
        train_files = [f.stem for f in train_dir.glob("*.pt") if not f.name.startswith('metadata')]
        if train_files:
            print(f"\n📁 사용 가능한 Train 파일들:")
            for i, filename in enumerate(train_files[:5]):  # 처음 5개만 표시
                print(f"  {i+1}. {filename}")
            if len(train_files) > 5:
                print(f"  ... 총 {len(train_files)}개 파일")
    
    if test_dir.exists():
        test_files = [f.stem for f in test_dir.glob("*.pt") if not f.name.startswith('metadata')]
        if test_files:
            print(f"\n📁 사용 가능한 Test 파일들:")
            for i, filename in enumerate(test_files[:5]):  # 처음 5개만 표시
                print(f"  {i+1}. {filename}")
            if len(test_files) > 5:
                print(f"  ... 총 {len(test_files)}개 파일")
    
    print(f"\n💡 특정 파일을 시각화하려면:")
    print(f"  python vis.py --train-file scene0355_01 --test-file scene0789_00")

if __name__ == "__main__":
    main() 