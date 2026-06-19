import os
import argparse
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import h5py
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# 필요한 ModelNet40H5 및 CoordinateTransformation 클래스 가져오기
from utils.pointnet import ModelNet40H5, CoordinateTransformation

# 스크립트 자신의 디렉토리 기준으로 경로 고정 (cwd 무관)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 재현성을 위한 시드 설정
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CombinedDataset(Dataset):
    """여러 데이터셋을 결합하는 래퍼 클래스"""
    def __init__(self, datasets):
        self.datasets = datasets
        # 각 데이터셋의 크기 저장
        self.sizes = [len(ds) for ds in datasets]
        # 누적 크기 계산
        self.cumulative_sizes = np.cumsum(self.sizes)
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        # idx가 속한 데이터셋 찾기
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx > 0:
            idx_within_dataset = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            idx_within_dataset = idx
        
        return self.datasets[dataset_idx][idx_within_dataset]

class OccupancyGridProcessor:
    """
    포인트 클라우드 데이터를 binary occupancy 그리드로 변환하고 빈 공간을 채우는 프로세서
    """
    def __init__(self, resolution=128):
        self.resolution = resolution
    
    def normalize_point_cloud(self, points):
        """포인트 클라우드를 [0, 1] 범위로 정규화"""
        min_coords = torch.min(points, dim=0)[0]
        max_coords = torch.max(points, dim=0)[0]
        scale = torch.max(max_coords - min_coords)
        points = (points - min_coords) / scale
        return points
    
    def points_to_occupancy_grid(self, points):
        """
        포인트 클라우드를 binary occupancy 그리드로 변환하고 빈 공간을 채움
        """
        # 포인트 클라우드 정규화
        points = self.normalize_point_cloud(points)
        
        # Binary occupancy 그리드 초기화
        occupancy_grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.bool_)
        
        # 각 포인트를 복셀 인덱스로 변환
        indices = torch.floor(points * (self.resolution - 1)).long()
        
        # 유효한 인덱스만 선택
        valid_mask = torch.all((indices >= 0) & (indices < self.resolution), dim=1)
        indices = indices[valid_mask]
        
        # 복셀 그리드에 포인트 마킹
        for coord in indices:
            x, y, z = coord
            occupancy_grid[x, y, z] = True
        
        # Morphological dilation으로 빈 공간 채우기
        # 3x3x3 structuring element로 2번 반복 적용
        struct_element = np.ones((3, 3, 3), dtype=bool)
        occupancy_grid = binary_dilation(occupancy_grid, structure=struct_element, iterations=2)
                
        # Float32 텐서로 변환
        occupancy_grid = torch.from_numpy(occupancy_grid).float()
        
        return occupancy_grid

    def process_dataset(self, dataset, output_dir, max_samples=None):
        """
        데이터셋을 처리하고 각 샘플을 개별 파일로 저장
        """
        # 사용할 샘플 수 계산
        if max_samples is not None and max_samples < len(dataset):
            indices = list(range(len(dataset)))
            np.random.shuffle(indices)
            indices = indices[:max_samples]
            samples_to_process = max_samples
        else:
            indices = list(range(len(dataset)))
            samples_to_process = len(dataset)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 메타데이터 파일 생성
        meta_file = os.path.join(output_dir, "metadata.txt")
        with open(meta_file, 'w') as f:
            f.write(f"resolution: {self.resolution}\n")
            f.write(f"samples: {samples_to_process}\n")
            f.write(f"binary_voxels: true\n")
            f.write(f"gap_filling: enabled\n")
        
        # 각 샘플 처리 및 개별 저장
        successful_samples = 0
        for i, idx in enumerate(tqdm(indices, desc=f"처리 중: {output_dir}")):
            try:
                data = dataset[idx]
                
                if data is None:
                    # 잘못된 데이터는 건너뜀
                    continue
                
                # 포인트 클라우드 좌표 추출
                points = data["coordinates"]
                label = data["label"]
                
                # 포인트 클라우드를 binary occupancy 그리드로 변환
                occupancy_grid = self.points_to_occupancy_grid(points)
                
                # 개별 파일에 저장
                sample_file = os.path.join(output_dir, f"sample_{i:05d}_label_{label.item():02d}.pt")
                torch.save({
                    'occupancy_grid': occupancy_grid,
                    'label': label,
                    'resolution': self.resolution
                }, sample_file)
                
                successful_samples += 1
                
                # 진행 상황 확인을 위해 주기적으로 상태 출력
                if (i + 1) % 100 == 0:
                    occupancy_ratio = torch.sum(occupancy_grid) / occupancy_grid.numel()
                    print(f"처리 진행 중: {i+1}/{samples_to_process} 완료 (점유율: {occupancy_ratio:.2%})")
                
            except Exception as e:
                print(f"샘플 {i} 처리 중 오류 발생: {str(e)}")
        
        print(f"{successful_samples}개 샘플 처리 완료. 디렉토리에 저장됨: {output_dir}")
        
        return successful_samples

def prepare_datasets(resolution=128, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, max_samples=None, use_percentage=0.5):
    """
    학습, 검증, 테스트 데이터셋을 지정된 비율로 준비하고 처리
    """
    print(f"Binary voxel 데이터셋 준비 중... (전체 데이터의 {use_percentage*100}% 사용)")
    print(f"분할 비율 - Train {train_ratio * 100}% : Val {val_ratio * 100}% : Test {test_ratio * 100}%")
    
    # 비율 검증
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합은 1이어야 합니다."
    assert 0.0 < use_percentage <= 1.0, "사용 비율은 0보다 크고 1 이하여야 합니다."
    
    # 재현성을 위한 시드 설정
    seed_everything(42)
    
    # 데이터 디렉토리
    data_root = os.path.join(BASE_DIR, "datasets", "modelnet40_ply_hdf5_2048")
    
    # 기본 데이터셋 로드
    train_base_dataset = ModelNet40H5(
        phase="train",
        transform=CoordinateTransformation(trans=0.0),
        data_root=data_root,
    )
    
    test_base_dataset = ModelNet40H5(
        phase="test",
        transform=CoordinateTransformation(trans=0.0),
        data_root=data_root,
    )
    
    # 전체 데이터 결합
    combined_base_dataset = CombinedDataset([train_base_dataset, test_base_dataset])
    
    # 사용할 데이터 샘플 수 계산
    print(f"총 데이터 샘플 수: {len(combined_base_dataset)}")
    total_available = len(combined_base_dataset)
    samples_to_use = int(total_available * use_percentage)
    
    # 최대 샘플 수 제한 (메모리/시간 절약을 위해)
    if max_samples is not None:
        samples_to_use = min(samples_to_use, max_samples)
    
    print(f"사용할 샘플 수: {samples_to_use} ({samples_to_use/total_available*100:.1f}%)")
    
    # 데이터셋 분할
    full_indices = list(range(total_available))
    np.random.shuffle(full_indices)
    indices_to_use = full_indices[:samples_to_use]
    
    total_size = len(indices_to_use)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_indices = indices_to_use[:train_size]
    val_indices = indices_to_use[train_size:train_size+val_size]
    test_indices = indices_to_use[train_size+val_size:]
    
    print(f"Train 세트: {len(train_indices)} 샘플 ({len(train_indices)/total_size*100:.1f}%)")
    print(f"Validation 세트: {len(val_indices)} 샘플 ({len(val_indices)/total_size*100:.1f}%)")
    print(f"Test 세트: {len(test_indices)} 샘플 ({len(test_indices)/total_size*100:.1f}%)")
    
    # 데이터 디렉토리 생성
    base_output_dir = os.path.join(BASE_DIR, "processed_data")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 디렉토리 경로 설정 (binary로 구분)
    train_dir = os.path.join(base_output_dir, f"classification_train_res{resolution}_binary")
    val_dir = os.path.join(base_output_dir, f"classification_val_res{resolution}_binary")
    test_dir = os.path.join(base_output_dir, f"classification_test_res{resolution}_binary")
    
    # 프로세서 초기화
    processor = OccupancyGridProcessor(resolution=resolution)
    
    # 훈련 데이터 처리
    train_dataset = SubsetDataset(combined_base_dataset, train_indices)
    processor.process_dataset(train_dataset, train_dir)
    
    # 검증 데이터 처리 (있는 경우)
    if val_size > 0:
        val_dataset = SubsetDataset(combined_base_dataset, val_indices)
        processor.process_dataset(val_dataset, val_dir)
    
    # 테스트 데이터 처리
    test_dataset = SubsetDataset(combined_base_dataset, test_indices)
    processor.process_dataset(test_dataset, test_dir)
    
    return train_dir, val_dir if val_size > 0 else None, test_dir

class SubsetDataset(Dataset):
    """
    데이터셋의 일부만 사용하는 서브셋 데이터셋
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def main():
    parser = argparse.ArgumentParser(description='Binary Occupancy Grid 데이터 준비 (Classification용)')
    parser.add_argument('--resolution', type=int, default=128, help='occupancy 그리드 해상도')
    parser.add_argument('--use_percentage', type=float, default=0.5, help='전체 데이터셋 중 사용할 비율 (0.0~1.0)')
    parser.add_argument('--max_samples', type=int, default=None, help='처리할 최대 샘플 수')
    args = parser.parse_args()
    
    # 시드 고정
    seed_everything(42)
    
    # 데이터셋 준비 및 처리
    train_dir, val_dir, test_dir = prepare_datasets(
        resolution=args.resolution,
        train_ratio=0.8,
        val_ratio=0.0,
        test_ratio=0.2,
        max_samples=args.max_samples,
        use_percentage=args.use_percentage
    )
    
    print("Binary voxel 데이터 처리 완료!")
    print(f"훈련 데이터: {train_dir}")
    if val_dir:
        print(f"검증 데이터: {val_dir}")
    print(f"테스트 데이터: {test_dir}")

if __name__ == "__main__":
    main() 