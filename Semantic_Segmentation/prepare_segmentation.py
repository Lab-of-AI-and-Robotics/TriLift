#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import glob
import torch
import logging
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_dilation
from collections import Counter

# Anchor relative data paths to this script's directory (cwd-independent)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ScanNet 클래스 라벨 및 ID 정의
CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

# ScanNet 라벨을 NYU40 클래스로 매핑
CLASS_LABELS_TO_NYU40 = {
    0: 0,   # unannotated
    1: 1,   # wall
    2: 2,   # floor  
    3: 3,   # cabinet
    4: 4,   # bed
    5: 5,   # chair
    6: 6,   # sofa
    7: 7,   # table
    8: 8,   # door
    9: 9,   # window
    10: 10, # bookshelf
    11: 11, # picture
    12: 12, # counter
    14: 13, # desk
    16: 14, # curtain
    24: 15, # refrigerator
    28: 16, # shower curtain
    33: 17, # toilet
    34: 18, # sink
    36: 19, # bathtub
    39: 20  # otherfurniture
}

# Stanford3D 클래스 라벨 정의
STANFORD3D_LABELS = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 
                     'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    16: (219., 219., 141.),
    24: (255., 127., 14.),
    28: (158., 218., 229.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    36: (227., 119., 194.),
    39: (82., 84., 163.),
}

class ImprovedPointCloudToVoxel:
    """
    개선된 포인트 클라우드를 복셀 그리드로 변환하는 클래스
    ScanNet과 Stanford3D 모두 지원
    """
    def __init__(self, resolution=128, enable_gap_filling=True, dataset_type='scannet'):
        self.resolution = resolution
        self.enable_gap_filling = enable_gap_filling
        self.dataset_type = dataset_type.lower()
        
        # 데이터셋별 라벨 매핑 설정
        if self.dataset_type == 'scannet':
            self.setup_scannet_labels()
        elif self.dataset_type == 'stanford3d':
            self.setup_stanford3d_labels()
        else:
            logger.warning(f"Unknown dataset type: {dataset_type}, using ScanNet default")
            self.setup_scannet_labels()
    
    def setup_scannet_labels(self):
        """ScanNet 라벨 매핑 설정 (41개 클래스 -> 21개 클래스, 배경 포함)"""
        self.label_map = {}
        for i in range(41):
            if i in VALID_CLASS_IDS:
                # VALID_CLASS_IDS의 인덱스를 1부터 시작하는 클래스 ID로 매핑
                self.label_map[i] = VALID_CLASS_IDS.index(i) + 1
            else:
                self.label_map[i] = 0  # 배경
        self.num_classes = 21  # 배경 포함
        
    def setup_stanford3d_labels(self):
        """Stanford3D 라벨 매핑 설정"""
        # Stanford3D는 보통 1부터 시작하는 연속된 라벨을 사용
        # 필요에 따라 수정 가능
        self.label_map = {}
        num_stanford_classes = len(STANFORD3D_LABELS)
        for i in range(num_stanford_classes + 1):  # 0부터 num_classes까지
            self.label_map[i] = i  # 직접 매핑
        self.num_classes = num_stanford_classes + 1  # 배경 포함
    
    def normalize_point_cloud(self, points):
        """포인트 클라우드를 [0, 1] 범위로 정규화"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        scale = np.max(max_coords - min_coords)
        if scale > 0:
            points = (points - min_coords) / scale
        return points

    def apply_gap_filling(self, occupancy_grid, color_grid, label_grid):
        """
        개선된 Gap filling: occupancy와 label 일관성 보장
        """
        if not self.enable_gap_filling:
            return occupancy_grid, color_grid, label_grid
        
        logger.info("개선된 Gap filling 적용 중...")
        
        # 1. 원본 occupancy 복사
        original_occupancy = occupancy_grid.copy()
        
        # 2. Morphological dilation으로 확장
        struct_element = np.ones((3, 3, 3), dtype=bool)
        filled_occupancy = binary_dilation(occupancy_grid, structure=struct_element, iterations=2)
        
        # 3. 새로 채워진 복셀 찾기
        new_voxels = filled_occupancy & (~original_occupancy)
        new_indices = np.where(new_voxels)
        
        logger.info(f"Gap filling으로 {len(new_indices[0])}개 복셀 추가")
        
        # 4. 새로 채워진 복셀에 적절한 라벨 할당
        for idx in range(len(new_indices[0])):
            i, j, k = new_indices[0][idx], new_indices[1][idx], new_indices[2][idx]
            
            # 주변 점유된 복셀들의 정보 수집
            neighbor_colors = []
            neighbor_labels = []
            
            # 3x3x3 윈도우에서 원래 점유된 복셀들만 고려
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if (0 <= ni < self.resolution and 
                            0 <= nj < self.resolution and 
                            0 <= nk < self.resolution and
                            original_occupancy[ni, nj, nk]):  # 원래 점유된 복셀만
                            
                            neighbor_colors.append(color_grid[ni, nj, nk])
                            current_label = label_grid[ni, nj, nk]
                            if current_label >= 0:  # 유효한 라벨만
                                neighbor_labels.append(current_label)
            
            # 5. 색상 할당 (평균)
            if neighbor_colors:
                color_grid[i, j, k] = np.mean(neighbor_colors, axis=0)
            else:
                color_grid[i, j, k] = [0.5, 0.5, 0.5]  # 기본 회색
            
            # 6. 라벨 할당 (majority voting)
            if neighbor_labels:
                counter = Counter(neighbor_labels)
                most_common_label = counter.most_common(1)[0][0]
                label_grid[i, j, k] = most_common_label
            else:
                # 주변에 유효한 라벨이 없으면 배경으로 설정
                label_grid[i, j, k] = 0
        
        return filled_occupancy, color_grid, label_grid

    def enforce_consistency(self, occupancy_grid, label_grid):
        """
        occupancy와 label의 일관성 강제 적용
        핵심 규칙:
        - occupancy=0 → label=-1 (ignore)
        - occupancy=1 → label∈{0,1,2,...,num_classes-1} (유효한 클래스)
        """
        logger.info("🔧 Occupancy-Label 일관성 검증 및 수정 중...")
        
        # 일관성 위반 카운트
        violations_empty_to_ignore = 0
        violations_filled_to_valid = 0
        
        # 1. occupancy=0인 곳은 반드시 label=-1
        empty_mask = (occupancy_grid == 0)
        non_ignore_in_empty = empty_mask & (label_grid != -1)
        if np.any(non_ignore_in_empty):
            violations_empty_to_ignore = np.sum(non_ignore_in_empty)
            label_grid[non_ignore_in_empty] = -1
        
        # 2. occupancy=1인 곳은 반드시 유효한 라벨 (0~num_classes-1)
        filled_mask = (occupancy_grid == 1)
        ignore_in_filled = filled_mask & (label_grid == -1)
        if np.any(ignore_in_filled):
            violations_filled_to_valid = np.sum(ignore_in_filled)
            # ignore가 있는 점유된 복셀은 배경(0)으로 설정
            label_grid[ignore_in_filled] = 0
        
        logger.info(f"✅ 일관성 수정 완료:")
        logger.info(f"    빈 복셀 → ignore 변경: {violations_empty_to_ignore:,}개")
        logger.info(f"    점유 복셀 → 유효 라벨 변경: {violations_filled_to_valid:,}개")
        
        # 최종 검증
        final_empty_with_valid = np.sum((occupancy_grid == 0) & (label_grid != -1))
        final_filled_with_ignore = np.sum((occupancy_grid == 1) & (label_grid == -1))
        
        if final_empty_with_valid == 0 and final_filled_with_ignore == 0:
            logger.info("🎯 완벽한 occupancy-label 일관성 달성!")
        else:
            logger.warning(f"⚠️  일관성 문제 잔존: empty+valid={final_empty_with_valid}, filled+ignore={final_filled_with_ignore}")
        
        return occupancy_grid, label_grid
    
    def convert_to_voxel(self, coords, colors, labels):
        """
        개선된 포인트 클라우드를 복셀 그리드로 변환 + 일관성 보장
        
        Args:
            coords: 3D 좌표 (Nx3)
            colors: RGB 색상 (Nx3)
            labels: 세그멘테이션 라벨 (N,)
            
        Returns:
            occupancy_grid: 점유 복셀 그리드 (boolean)
            color_grid: 색상 복셀 그리드 (resolution x resolution x resolution x 3)
            label_grid: 라벨 복셀 그리드 (resolution x resolution x resolution)
        """
        # 좌표 정규화
        coords = self.normalize_point_cloud(coords)
        
        # 복셀 그리드 초기화
        occupancy_grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=bool)
        color_grid = np.zeros((self.resolution, self.resolution, self.resolution, 3), dtype=np.float32)
        label_grid = np.full((self.resolution, self.resolution, self.resolution), -1, dtype=np.int32)  # ignore로 초기화
        
        # 색상 정규화 [0, 255] → [0, 1]
        if colors.max() > 1.0:
            colors = colors / 255.0
        
        # 라벨 매핑 적용
        mapped_labels = np.array([self.label_map.get(label, 0) for label in labels])
        
        # 포인트를 복셀 인덱스로 변환
        voxel_coords = np.floor(coords * (self.resolution - 1)).astype(int)
        voxel_coords = np.clip(voxel_coords, 0, self.resolution - 1)
        
        # 복셀별로 포인트 집계
        for i in range(len(voxel_coords)):
            x, y, z = voxel_coords[i]
            occupancy_grid[x, y, z] = True
            color_grid[x, y, z] = colors[i]  # 마지막 색상으로 덮어쓰기
            label_grid[x, y, z] = mapped_labels[i]  # 마지막 라벨로 덮어쓰기
        
        # Gap filling 적용 (선택적)
        if self.enable_gap_filling:
            occupancy_grid, color_grid, label_grid = self.apply_gap_filling(
                occupancy_grid, color_grid, label_grid)
        
        # 일관성 강제 적용
        occupancy_grid, label_grid = self.enforce_consistency(occupancy_grid, label_grid)
        
        return occupancy_grid, color_grid, label_grid

class DatasetProcessor:
    """
    다중 데이터셋 처리를 위한 통합 클래스
    ScanNet과 Stanford3D를 모두 지원
    """
    def __init__(self, input_base_dir="preprocessed", 
                 output_base_dir="processed_data", resolution=128, save_data=True, 
                 enable_gap_filling=True, sample_ratio=1.0):
        """
        Args:
            input_base_dir: 입력 데이터 기본 디렉토리
            output_base_dir: 출력 데이터 기본 디렉토리  
            resolution: 복셀 그리드 해상도
            save_data: 데이터 저장 여부
            enable_gap_filling: Gap filling 활성화 여부
            sample_ratio: 데이터 샘플링 비율 (0.0-1.0)
        """
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.resolution = resolution
        self.save_data = save_data
        self.enable_gap_filling = enable_gap_filling
        self.sample_ratio = sample_ratio
        
        # 시드 설정
        np.random.seed(42)
        
        # 처리할 특정 데이터셋 (None이면 모든 데이터셋 처리)
        self.target_dataset = None
        
        # 지원되는 데이터셋 설정
        self.dataset_configs = {
            'scannet': {
                'num_classes': 21,
                'class_labels': CLASS_LABELS,
                'valid_class_ids': VALID_CLASS_IDS,
                'label_mapping': CLASS_LABELS_TO_NYU40,
                'ignore_index': -1
            },
            'stanford3d': {
                'num_classes': 13,
                'class_labels': STANFORD3D_LABELS,
                'valid_class_ids': None,
                'label_mapping': None,
                'ignore_index': -1
            }
        }
        
    def set_target_dataset(self, dataset_name):
        """특정 데이터셋만 처리하도록 설정"""
        if dataset_name in self.dataset_configs:
            self.target_dataset = dataset_name
            logger.info(f"🎯 Target dataset set to: {dataset_name.upper()}")
        else:
            logger.error(f"❌ Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {list(self.dataset_configs.keys())}")
            self.target_dataset = None
    
    def detect_available_datasets(self):
        """사용 가능한 데이터셋을 감지하고 경로를 반환"""
        datasets = {}
        
        # ScanNet 확인
        scannet_path = os.path.join(self.input_base_dir, "scannet")
        if os.path.exists(scannet_path):
            datasets['scannet'] = scannet_path
            logger.info(f"✅ SCANNET 데이터셋 발견: {scannet_path}")
        else:
            logger.info(f"❌ SCANNET 데이터셋 없음: {scannet_path}")
        
        # Stanford3D 확인 (대문자 경로로 수정)
        stanford3d_path = os.path.join(self.input_base_dir, "Stanford3D")
        if os.path.exists(stanford3d_path):
            datasets['stanford3d'] = stanford3d_path
            logger.info(f"✅ STANFORD3D 데이터셋 발견: {stanford3d_path}")
        else:
            logger.info(f"❌ STANFORD3D 데이터셋 없음: {stanford3d_path}")
        
        return datasets
        
    def read_plyfile(self, filepath):
        """PLY 파일 읽기"""
        with open(filepath, 'rb') as f:
            plydata = PlyData.read(f)
        data = plydata.elements[0].data
        
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        colors = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32) if 'label' in data.dtype.names else None
        
        return coords, colors, labels
    
    def process_single_dataset(self, dataset_name, phase="train"):
        """단일 데이터셋의 특정 phase 처리"""
        logger.info(f"🚀 Processing {dataset_name.upper()} {phase} dataset with resolution {self.resolution}x{self.resolution}x{self.resolution}")
        logger.info(f"🔧 Gap filling: {'enabled' if self.enable_gap_filling else 'disabled'}")
        logger.info(f"📊 Sample ratio: {self.sample_ratio * 100:.1f}%")
        
        # 데이터셋별 설정
        config = self.dataset_configs[dataset_name]
        dataset_path = self.detect_available_datasets()[dataset_name]
        
        # 파일 목록 가져오기
        if dataset_name == 'scannet':
            # ScanNet: train/test 폴더가 이미 분리되어 있음
            phase_path = os.path.join(dataset_path, phase)
            if os.path.exists(phase_path):
                all_files = glob.glob(os.path.join(phase_path, "*.ply"))
                scene_names = [os.path.basename(f).replace('.ply', '') for f in all_files]
            else:
                logger.error(f"ScanNet {phase} directory not found: {phase_path}")
                return None
        elif dataset_name == 'stanford3d':
            # Stanford3D: Area별 파일 수집
            all_files = []
            scene_names = []
            
            # train/test split 정의 (일반적으로 Area 5가 test)
            if phase == "train":
                areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
            else:  # test
                areas = ['Area_5']
            
            for area in areas:
                area_path = os.path.join(dataset_path, area)
                if os.path.exists(area_path):
                    area_files = glob.glob(os.path.join(area_path, "*.ply"))
                    all_files.extend(area_files)
                    # Area_1_office_1 형식으로 scene name 생성
                    for f in area_files:
                        room_name = os.path.basename(f).replace('.ply', '')
                        scene_name = f"{area}_{room_name}"
                        scene_names.append(scene_name)
            
            logger.info(f"Stanford3D {phase} areas: {areas}")
            logger.info(f"Found {len(all_files)} files in {len(areas)} areas")
        
        # 파일이 없으면 early return
        if not all_files:
            logger.warning(f"⚠️  No files found for {dataset_name.upper()} {phase}")
            return None
        
        # 샘플링
        total_files = len(all_files)
        num_samples = max(1, int(total_files * self.sample_ratio))
        
        if num_samples < total_files:
            # 랜덤 샘플링
            indices = np.random.choice(total_files, num_samples, replace=False)
            selected_files = [all_files[i] for i in indices]
            selected_scenes = [scene_names[i] for i in indices]
        else:
            selected_files = all_files
            selected_scenes = scene_names
        
        logger.info(f"전체 {total_files}개 파일 중 {len(selected_files)}개 파일 선택 ({self.sample_ratio * 100:.1f}%)")
        logger.info(f"{phase.capitalize()} 파일: {len(selected_files)}개")
        if len(selected_scenes) <= 10:
            logger.info(f"선택된 {phase} 파일들: {selected_scenes[:3]}...")
        
        # 출력 디렉토리 설정
        output_dir = os.path.join(self.output_base_dir, dataset_name, phase)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        if not self.save_data:
            logger.info("⚠️  Save mode disabled - processing without saving")
        
        # Point Cloud to Voxel 변환기 초기화
        pc_to_voxel = ImprovedPointCloudToVoxel(
            resolution=self.resolution, 
            enable_gap_filling=self.enable_gap_filling,
            dataset_type=dataset_name
        )
        
        # 통계 변수
        total_valid_voxels = 0
        total_voxels = self.resolution ** 3 * len(selected_files)
        label_counts = Counter()
        occupancy_ratios = []
        
        # 데이터 처리
        for i, (file_path, scene_name) in enumerate(tqdm(zip(selected_files, selected_scenes), 
                                                      desc=f"Converting {dataset_name} {phase} data", 
                                                      total=len(selected_files))):
            try:
                # PLY 파일 읽기
                coords, colors, labels = self.read_plyfile(file_path)
                
                if coords is None:
                    logger.warning(f"⚠️  Skipping {scene_name}: Invalid data")
                    continue
                
                # Voxel 변환
                occupancy_grid, color_grid, label_grid = pc_to_voxel.convert_to_voxel(coords, colors, labels)
                
                # 통계 계산
                valid_voxels = np.sum(occupancy_grid)
                total_valid_voxels += valid_voxels
                occupancy_ratio = valid_voxels / (self.resolution ** 3)
                occupancy_ratios.append(occupancy_ratio)
                
                # 라벨 통계
                unique_labels, counts = np.unique(label_grid, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    label_counts[label] += count
                
                # 데이터 저장
                if self.save_data:
                    output_file = os.path.join(output_dir, f"{scene_name}.pt")
                    data = {
                        'occupancy': occupancy_grid.astype(np.uint8),
                        'colors': color_grid.astype(np.float32),
                        'labels': label_grid.astype(np.int32),
                        'original_points': len(coords),
                        'scene_name': scene_name,
                        'dataset': dataset_name.upper(),
                        'resolution': self.resolution
                    }
                    torch.save(data, output_file)
                
                # 주기적 로그 출력
                if (i + 1) % max(1, len(selected_files) // 3) == 0 or i == len(selected_files) - 1:
                    consistency_check = "✅ Perfect" if np.sum(occupancy_grid) == np.sum(label_grid != config['ignore_index']) else "❌ Inconsistent"
                    gap_fill_status = "Applied" if self.enable_gap_filling else "Disabled"
                    
                    logger.info(f"📄 Sample {i+1}/{len(selected_files)}: {scene_name}")
                    logger.info(f"     Original points: {len(coords):,}")
                    logger.info(f"     Voxel occupancy: {valid_voxels:,}/{self.resolution**3:,} ({occupancy_ratio*100:.2f}%)")
                    logger.info(f"     Unique labels: {len(unique_labels)} (including ignore)")
                    logger.info(f"     Consistency: {consistency_check}")
                    logger.info(f"     Gap filling: {gap_fill_status}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {scene_name}: {str(e)}")
                continue
        
        # 전체 데이터셋 통계 출력
        logger.info(f"\n🎯 === {dataset_name.upper()} {phase} Dataset Statistics ===")
        logger.info(f"Source: {phase} data split")
        logger.info(f"Processed samples: {len(selected_files)} (from {total_files} total)")
        logger.info(f"Sample ratio: {self.sample_ratio*100:.1f}%")
        logger.info(f"Voxel grid shape: {self.resolution}x{self.resolution}x{self.resolution}")
        logger.info(f"Gap filling: {'enabled' if self.enable_gap_filling else 'disabled'}")
        
        # 라벨 분포 출력
        logger.info(f"\n📊 === {dataset_name.upper()} Label Distribution ===")
        total_voxels = sum(label_counts.values())
        
        if total_voxels == 0:
            logger.warning("⚠️  No voxel data processed - skipping statistics")
            return None
        
        # ignore(-1) 라벨부터 시작
        if -1 in label_counts:
            logger.info(f"  {-1:2d}: {'ignore':15s}: {label_counts[-1]:,} ({label_counts[-1]/total_voxels:.2%})")
        
        # 0부터 시작하는 클래스 라벨들
        for label_id in range(config['num_classes']):
            if label_id in label_counts:
                if label_id == 0:
                    label_name = 'background'
                else:
                    label_name = config['class_labels'][label_id - 1] if label_id - 1 < len(config['class_labels']) else f'class_{label_id}'
                logger.info(f"  {label_id:2d}: {label_name:15s}: {label_counts[label_id]:,} ({label_counts[label_id]/total_voxels:.2%})")
        
        # 학습 데이터 비율 계산
        ignore_count = label_counts.get(-1, 0)
        valid_count = total_voxels - ignore_count
        logger.info(f"\n💡 Learning Data Ratio: {valid_count:,} / {total_voxels:,} ({valid_count/total_voxels:.2%})")
        logger.info(f"💡 Ignore Data Ratio: {ignore_count:,} / {total_voxels:,} ({ignore_count/total_voxels:.2%})")
        
        if self.save_data:
            logger.info(f"\n✅ Data saved to: {output_dir}")
        logger.info(f"✅ {dataset_name.upper()} {phase} processing completed!")
        
        return {
            "dataset": dataset_name,
            "total_samples": len(selected_files),
            "total_files": total_files,
            "selected_files": len(selected_files),
            "sample_ratio": self.sample_ratio,
            "split_method": "existing_split" if dataset_name == 'scannet' else "area_based_split",
            "shape": (self.resolution, self.resolution, self.resolution),
            "avg_occupancy": np.mean(occupancy_ratios) if occupancy_ratios else 0.0,
            "min_occupancy": np.min(occupancy_ratios) if occupancy_ratios else 0.0,
            "max_occupancy": np.max(occupancy_ratios) if occupancy_ratios else 0.0,
            "label_distribution": label_counts,
            "gap_filling": self.enable_gap_filling,
            "consistency_enforced": True,
            "valid_data_ratio": valid_count / total_voxels if total_voxels > 0 else 0.0,
            "ignore_data_ratio": ignore_count / total_voxels if total_voxels > 0 else 0.0,
            "output_dir": output_dir if self.save_data else None,
            "num_classes": config['num_classes']
        }
    
    def process_all_datasets(self):
        """모든 사용 가능한 데이터셋 처리"""
        if self.target_dataset:
            logger.info(f"🚀 Starting single dataset processing: {self.target_dataset.upper()}")
        else:
            logger.info("🚀 Starting multi-dataset processing")
        
        logger.info(f"📂 Input base directory: {self.input_base_dir}")
        logger.info(f"📂 Output base directory: {self.output_base_dir}")
        logger.info(f"📊 Sample ratio: {self.sample_ratio * 100:.1f}%")
        logger.info(f"🔧 Gap filling: {'enabled' if self.enable_gap_filling else 'disabled'}")
        logger.info(f"🎯 Consistency enforcement: enabled")
        
        # 사용 가능한 데이터셋 감지
        self.available_datasets = self.detect_available_datasets()
        
        if not self.available_datasets:
            logger.error("❌ 사용 가능한 데이터셋이 없습니다!")
            logger.info(f"지원되는 데이터셋: {list(self.dataset_configs.keys())}")
            logger.info(f"입력 디렉토리: {self.input_base_dir}")
            return {}
        
        # 특정 데이터셋만 처리하는 경우 필터링
        if self.target_dataset:
            if self.target_dataset in self.available_datasets:
                filtered_datasets = {self.target_dataset: self.available_datasets[self.target_dataset]}
                logger.info(f"🎯 Processing only {self.target_dataset.upper()} dataset")
            else:
                logger.error(f"❌ Target dataset '{self.target_dataset}' not found!")
                logger.info(f"Available datasets: {list(self.available_datasets.keys())}")
                return {}
        else:
            filtered_datasets = self.available_datasets
            logger.info(f"🔄 Processing all available datasets: {list(filtered_datasets.keys())}")
        
        all_results = {}
        
        for dataset_name, dataset_path in filtered_datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 Processing {dataset_name.upper()} Dataset")
            logger.info(f"{'='*60}")
            
            dataset_results = {}
            
            # Train 데이터 처리
            logger.info(f"\n📊 Processing {dataset_name.upper()} train phase...")
            train_result = self.process_single_dataset(dataset_name, "train")
            if train_result:
                dataset_results["train"] = train_result
                logger.info(f"✅ {dataset_name.upper()} train processing completed!")
            else:
                logger.warning(f"⚠️  {dataset_name.upper()} train processing failed or no data!")
            
            # Test 데이터 처리
            logger.info(f"\n📊 Processing {dataset_name.upper()} test phase...")
            test_result = self.process_single_dataset(dataset_name, "test")
            if test_result:
                dataset_results["test"] = test_result
                logger.info(f"✅ {dataset_name.upper()} test processing completed!")
            else:
                logger.warning(f"⚠️  {dataset_name.upper()} test processing failed or no data!")
            
            # 결과가 있는 경우만 추가
            if dataset_results:
                all_results[dataset_name] = dataset_results
        
        # 전체 요약 출력
        self._print_final_summary(all_results)
        
        return all_results

    def _print_final_summary(self, all_results):
        """전체 처리 결과 요약 출력"""
        logger.info(f"\n🎯 === Final Multi-Dataset Statistics ===")
        logger.info(f"Resolution: {self.resolution}x{self.resolution}x{self.resolution}")
        logger.info(f"Sample ratio: {self.sample_ratio * 100:.1f}%")
        logger.info(f"Gap filling: {'enabled' if self.enable_gap_filling else 'disabled'}")
        logger.info(f"Consistency enforcement: enabled")
        
        total_train_samples = 0
        total_test_samples = 0
        
        for dataset_name, results in all_results.items():
            logger.info(f"\n📊 {dataset_name.upper()} Dataset:")
            
            if 'train' in results and results['train']:
                train_result = results['train']
                total_train_samples += train_result['total_samples']
                logger.info(f"   Train: {train_result['total_samples']} samples / {train_result['total_files']} total files")
                logger.info(f"     Average occupancy: {train_result['avg_occupancy']*100:.2f}%")
                logger.info(f"     Valid data ratio: {train_result['valid_data_ratio']*100:.2f}%")
                logger.info(f"     Classes: {train_result['num_classes']}")
            else:
                logger.info(f"   Train: No data processed")
            
            if 'test' in results and results['test']:
                test_result = results['test']
                total_test_samples += test_result['total_samples']
                logger.info(f"   Test: {test_result['total_samples']} samples / {test_result['total_files']} total files")
                logger.info(f"     Average occupancy: {test_result['avg_occupancy']*100:.2f}%")
                logger.info(f"     Valid data ratio: {test_result['valid_data_ratio']*100:.2f}%")
                logger.info(f"     Classes: {test_result['num_classes']}")
            else:
                logger.info(f"   Test: No data processed")
        
        logger.info(f"\n🎉 Processing Summary:")
        logger.info(f"   📂 Processed datasets: {len(all_results)}")
        logger.info(f"   🚂 Total train samples: {total_train_samples}")
        logger.info(f"   🧪 Total test samples: {total_test_samples}")
        
        if self.save_data and total_train_samples + total_test_samples > 0:
            logger.info(f"\n✅ All processed data saved to: {self.output_base_dir}")
            logger.info(f"📁 Directory structure:")
            for dataset_name in all_results.keys():
                logger.info(f"   {self.output_base_dir}/{dataset_name}/train")
                logger.info(f"   {self.output_base_dir}/{dataset_name}/test")
        
        logger.info(f"\n🎉 Multi-dataset voxel segmentation data generation completed!")
        logger.info(f"🔥 Key features:")
        logger.info(f"   ✅ Multi-dataset support (ScanNet, Stanford3D)")
        logger.info(f"   ✅ Perfect occupancy-label consistency")
        logger.info(f"   ✅ Enhanced gap filling algorithm")
        logger.info(f"   ✅ Dataset-specific folder structure")
        logger.info(f"   ✅ Comprehensive statistics and validation")

def main():
    parser = argparse.ArgumentParser(description="Convert multiple datasets (ScanNet, Stanford3D) to voxel grids for semantic segmentation")
    parser.add_argument("--input_dir", default=os.path.join(BASE_DIR, "preprocessed"), help="Input directory containing datasets")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "processed_data"), help="Output directory for processed data")
    parser.add_argument("--resolution", type=int, default=128, help="Voxel grid resolution")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Sample ratio (0.0-1.0)")
    parser.add_argument("--enable_gap_filling", action="store_true", default=True, help="Enable gap filling")
    parser.add_argument("--disable_gap_filling", action="store_false", dest="enable_gap_filling", help="Disable gap filling")
    parser.add_argument("--save", action="store_true", default=False, help="Save processed data")
    parser.add_argument("--dataset", type=str, choices=['scannet', 'stanford3d', 'all'], default='all',
                       help="Specify which dataset to process: 'scannet', 'stanford3d', or 'all' (default: all)")
    
    args = parser.parse_args()
    
    # 처리기 초기화
    processor = DatasetProcessor(
        input_base_dir=args.input_dir,
        output_base_dir=args.output_dir,
        resolution=args.resolution,
        save_data=args.save,
        enable_gap_filling=args.enable_gap_filling,
        sample_ratio=args.sample_ratio
    )
    
    # 특정 데이터셋만 처리하도록 설정
    if args.dataset != 'all':
        processor.set_target_dataset(args.dataset)
    
    # 모든 데이터셋 처리
    all_results = processor.process_all_datasets()
    
    logger.info(f"\n🎉 Processing completed successfully!")
    return all_results

if __name__ == "__main__":
    main() 