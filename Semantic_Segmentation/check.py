import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def analyze_tensor(data, name, detailed=True):
    """텐서 데이터를 분석하고 출력하는 함수"""
    print(f"\n=== {name} 분석 ===")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {data.min():.6f}")
    print(f"Max value: {data.max():.6f}")
    
    # 정수형 텐서의 경우 float로 변환하여 mean, std 계산
    if data.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
        data_float = data.float()
        mean_val = data_float.mean()
        std_val = data_float.std()
        print(f"Mean: {mean_val:.6f}")
        print(f"Std: {'nan' if torch.isnan(std_val) else f'{std_val:.6f}'}")
    else:
        mean_val = data.mean()
        std_val = data.std()
        print(f"Mean: {mean_val:.6f}")
        print(f"Std: {'nan' if torch.isnan(std_val) else f'{std_val:.6f}'}")
    
    if detailed and len(data.shape) > 1:
        print(f"Non-zero values: {torch.nonzero(data).shape[0]} / {data.numel()}")
        print(f"Sparsity: {1 - torch.nonzero(data).shape[0] / data.numel():.4f}")
    
    # 라벨 데이터인 경우 클래스 분석 추가
    if 'label' in name.lower() or 'target' in name.lower():
        unique_values = torch.unique(data)
        print(f"🏷️  클래스 분석:")
        print(f"    고유 값들: {unique_values.tolist()}")
        print(f"    클래스 개수: {len(unique_values)}개")
        
        # 클래스별 분포 (상위 10개만)
        unique_values_cpu = unique_values.cpu()
        print(f"    클래스별 분포:")
        for i, class_id in enumerate(unique_values_cpu):
            if i >= 10:  # 최대 10개 클래스만 표시
                print(f"      ... (총 {len(unique_values)}개 클래스)")
                break
            count = torch.sum(data == class_id).item()
            percentage = count / data.numel() * 100
            if class_id == -1:
                print(f"      클래스 {class_id:2d} (ignore): {count:,}개 ({percentage:.2f}%)")
            elif class_id == 0:
                print(f"      클래스 {class_id:2d} (background): {count:,}개 ({percentage:.2f}%)")
            else:
                print(f"      클래스 {class_id:2d}: {count:,}개 ({percentage:.2f}%)")
    
    # RGB 색상 채널별 분석 (4차원이고 마지막 차원이 3인 경우)
    if len(data.shape) == 4 and data.shape[-1] == 3:
        print(f"🎨 RGB 채널별 분석:")
        channel_names = ['Red', 'Green', 'Blue']
        
        for i in range(3):  # RGB 3채널
            channel_data = data[..., i]  # [..., i]로 마지막 차원의 i번째 채널 선택
            non_zero_mask = channel_data != 0
            non_zero_count = torch.sum(non_zero_mask).item()
            total_count = channel_data.numel()
            
            if non_zero_count > 0:
                non_zero_values = channel_data[non_zero_mask]
                channel_min = non_zero_values.min().item()
                channel_max = non_zero_values.max().item()
                channel_mean = non_zero_values.mean().item()
                channel_std = non_zero_values.std().item()
                print(f"  📊 {channel_names[i]:5s}: min={channel_min:.4f}, max={channel_max:.4f}, "
                      f"mean={channel_mean:.4f}, std={channel_std:.4f}")
                print(f"         Non-zero: {non_zero_count:,}/{total_count:,} ({non_zero_count/total_count:.2%})")
                
                # 분포 히스토그램 (간단한 텍스트 버전)
                if non_zero_count > 100:  # 충분한 데이터가 있을 때만
                    hist_bins = 10
                    hist, bin_edges = torch.histogram(non_zero_values.float(), bins=hist_bins)
                    print(f"         분포: ", end="")
                    for j in range(hist_bins):
                        bin_start = bin_edges[j].item()
                        bin_end = bin_edges[j+1].item()
                        count = hist[j].item()
                        if count > 0:
                            print(f"[{bin_start:.2f}-{bin_end:.2f}]:{count:,} ", end="")
                    print()
            else:
                print(f"  📊 {channel_names[i]:5s}: 모든 값이 0 (빈 채널)")
    
    # 일반적인 채널별 분석 (다른 4차원 데이터)
    elif len(data.shape) >= 4 and data.shape[-1] != 3:
        print(f"채널별 분석 (총 {data.shape[1]}개 채널):")
        for i in range(min(data.shape[1], 10)):  # 최대 10개 채널만 표시
            channel_data = data[0, i] if data.shape[0] > 0 else data[i]
            non_zero_count = torch.nonzero(channel_data).shape[0]
            total_count = channel_data.numel()
            print(f"  채널 {i}: min={channel_data.min():.4f}, max={channel_data.max():.4f}, "
                  f"mean={channel_data.mean():.4f}, non-zero={non_zero_count}/{total_count}")

def check_classification_data():
    """Classification 데이터 분석"""
    print("\n" + "="*50)
    print("CLASSIFICATION 데이터 분석")
    print("="*50)
    
    # 테스트 데이터 확인
    test_dir = Path("../Classification_Completion/processed_data/classification_test_res128_binary")
    if test_dir.exists():
        files = list(test_dir.glob("*.pt"))
        if files:
            sample_file = files[0]
            print(f"분석 파일: {sample_file.name}")
            data = torch.load(sample_file, weights_only=False)
            
            if isinstance(data, dict):
                print("데이터 구조: Dictionary")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        analyze_tensor(value, f"Key: {key}")
                    else:
                        print(f"Key: {key}, Type: {type(value)}, Value: {value}")
                
                # Classification 클래스 정보 요약 (여러 파일에서 라벨 수집)
                print(f"\n📊 CLASSIFICATION 클래스 요약:")
                all_labels = set()
                sample_count = min(50, len(files))  # 최대 50개 파일만 체크
                for i, file_path in enumerate(files[:sample_count]):
                    file_data = torch.load(file_path, weights_only=False)
                    if 'label' in file_data:
                        label_value = file_data['label']
                        if isinstance(label_value, torch.Tensor):
                            all_labels.add(label_value.item())
                        else:
                            all_labels.add(label_value)
                
                sorted_labels = sorted(list(all_labels))
                print(f"    샘플링한 파일 수: {sample_count}개")
                print(f"    발견된 클래스: {sorted_labels}")
                print(f"    클래스 개수: {len(sorted_labels)}개")
                print(f"    클래스 범위: {min(sorted_labels)} ~ {max(sorted_labels)}" if sorted_labels else "    클래스 없음")
                
            elif isinstance(data, torch.Tensor):
                analyze_tensor(data, "Classification Data")
            else:
                print(f"데이터 타입: {type(data)}")
                print(f"데이터: {data}")

def check_completion_data():
    """Completion 데이터 분석"""
    print("\n" + "="*50)
    print("COMPLETION 데이터 분석")
    print("="*50)
    
    # 테스트 데이터 확인
    test_dir = Path("../Classification_Completion/processed_data/completion_chair_test_res128")
    if test_dir.exists():
        files = list(test_dir.glob("*.pt"))
        if files:
            sample_file = files[0]
            print(f"분석 파일: {sample_file.name}")
            data = torch.load(sample_file, weights_only=False)
            
            if isinstance(data, dict):
                print("데이터 구조: Dictionary")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        analyze_tensor(value, f"Key: {key}")
                    else:
                        print(f"Key: {key}, Type: {type(value)}, Value: {value}")
                
                # Completion 클래스 정보 요약
                print(f"\n📊 COMPLETION 클래스 요약:")
                all_labels = set()
                sample_count = min(20, len(files))  # 최대 20개 파일만 체크
                for i, file_path in enumerate(files[:sample_count]):
                    file_data = torch.load(file_path, weights_only=False)
                    if 'label' in file_data:
                        label_value = file_data['label']
                        if isinstance(label_value, torch.Tensor):
                            all_labels.add(label_value.item())
                        elif isinstance(label_value, (int, float)):
                            all_labels.add(label_value)
                
                sorted_labels = sorted(list(all_labels)) if all_labels else []
                print(f"    샘플링한 파일 수: {sample_count}개")
                print(f"    태스크 타입: Shape Completion (의자)")
                print(f"    발견된 라벨: {sorted_labels}")
                print(f"    클래스 개수: {len(sorted_labels)}개 (단일 객체 completion)")
                
            elif isinstance(data, torch.Tensor):
                analyze_tensor(data, "Completion Data")
            else:
                print(f"데이터 타입: {type(data)}")

def check_h5_data():
    """HDF5 데이터 분석"""
    print("\n" + "="*50)
    print("HDF5 데이터 분석")
    print("="*50)
    
    h5_file = Path("../Classification_Completion/processed_data/train_res128_thick5.h5")
    if h5_file.exists():
        print(f"분석 파일: {h5_file.name}")
        with h5py.File(h5_file, 'r') as f:
            print("HDF5 키들:", list(f.keys()))
            labels_data = None
            
            for key in f.keys():
                data = f[key]
                print(f"\nKey: {key}")
                print(f"Shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                
                # 라벨 데이터 저장
                if key == 'labels':
                    labels_data = data[:]
                
                # 샘플 데이터 분석 (첫 번째 샘플만)
                if len(data.shape) > 0:
                    sample = torch.tensor(data[0] if data.shape[0] > 0 else data[:])
                    analyze_tensor(sample, f"{key} (첫 번째 샘플)")
            
            # HDF5 클래스 정보 요약
            if labels_data is not None:
                unique_labels = np.unique(labels_data)
                print(f"\n📊 HDF5 클래스 요약:")
                print(f"    총 샘플 수: {len(labels_data):,}개")
                print(f"    고유 클래스: {unique_labels.tolist()}")
                print(f"    클래스 개수: {len(unique_labels)}개")
                print(f"    클래스 범위: {unique_labels.min()} ~ {unique_labels.max()}")
                
                # 클래스별 분포 (상위 10개만)
                unique, counts = np.unique(labels_data, return_counts=True)
                print(f"    클래스별 분포:")
                for i, (class_id, count) in enumerate(zip(unique, counts)):
                    if i >= 10:
                        print(f"      ... (총 {len(unique)}개 클래스)")
                        break
                    percentage = count / len(labels_data) * 100
                    print(f"      클래스 {class_id:2d}: {count:,}개 ({percentage:.1f}%)")

def check_scannet_data():
    """ScanNet 데이터 분석"""
    print("\n" + "="*50)
    print("SCANNET 데이터 분석")
    print("="*50)
    
    test_dir = Path("processed_data/scannet/test")
    if test_dir.exists():
        files = list(test_dir.glob("*.pt"))
        if files:
            sample_file = files[0]
            print(f"분석 파일: {sample_file.name}")
            data = torch.load(sample_file, weights_only=False)
            
            if isinstance(data, dict):
                print("데이터 구조: Dictionary")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        analyze_tensor(value, f"Key: {key}")
                    else:
                        print(f"Key: {key}, Type: {type(value)}, Value: {value}")
                
                # ScanNet 클래스 정보 요약
                if 'label_grid' in data:
                    label_data = data['label_grid']
                    if isinstance(label_data, torch.Tensor):
                        unique_labels = torch.unique(label_data)
                        valid_classes = unique_labels[unique_labels >= 0]  # ignore(-1) 제외
                        print(f"\n📊 SCANNET 클래스 요약:")
                        print(f"    총 클래스 개수: {len(valid_classes)}개 (background 포함)")
                        print(f"    클래스 범위: {valid_classes.min().item()} ~ {valid_classes.max().item()}")
                        print(f"    Ignore 라벨: -1 (포함됨)" if -1 in unique_labels else "    Ignore 라벨: 없음")
                        
            elif isinstance(data, torch.Tensor):
                analyze_tensor(data, "ScanNet Segmentation Data")
            else:
                print(f"데이터 타입: {type(data)}")

def check_stanford3d_data():
    """Stanford3D 데이터 분석"""
    print("\n" + "="*50)
    print("STANFORD3D 데이터 분석")
    print("="*50)
    
    test_dir = Path("processed_data/stanford3d/test")
    if test_dir.exists():
        files = list(test_dir.glob("*.pt"))
        if files:
            sample_file = files[0]
            print(f"분석 파일: {sample_file.name}")
            data = torch.load(sample_file, weights_only=False)
            
            if isinstance(data, dict):
                print("데이터 구조: Dictionary")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        analyze_tensor(value, f"Key: {key}")
                    elif isinstance(value, np.ndarray):
                        # numpy 배열을 torch 텐서로 변환해서 분석
                        tensor_value = torch.from_numpy(value)
                        analyze_tensor(tensor_value, f"Key: {key} (numpy->tensor)")
                    else:
                        print(f"Key: {key}, Type: {type(value)}, Value: {value}")
                
                # Stanford3D 클래스 정보 요약
                label_key = 'labels' if 'labels' in data else None
                if label_key and label_key in data:
                    label_data = data[label_key]
                    if isinstance(label_data, np.ndarray):
                        label_data = torch.from_numpy(label_data)
                    if isinstance(label_data, torch.Tensor):
                        unique_labels = torch.unique(label_data)
                        valid_classes = unique_labels[unique_labels >= 0]  # ignore(-1) 제외
                        print(f"\n📊 STANFORD3D 클래스 요약:")
                        print(f"    총 클래스 개수: {len(valid_classes)}개 (background 포함)")
                        print(f"    클래스 범위: {valid_classes.min().item()} ~ {valid_classes.max().item()}")
                        print(f"    Ignore 라벨: -1 (포함됨)" if -1 in unique_labels else "    Ignore 라벨: 없음")
                        
            elif isinstance(data, torch.Tensor):
                analyze_tensor(data, "Stanford3D Segmentation Data")
            else:
                print(f"데이터 타입: {type(data)}")

def main():
    print("PROCESSED_DATA 분석 시작")
    print("="*70)
    
    # 각 데이터셋 분석
    check_classification_data()
    check_completion_data()
    check_h5_data()
    check_scannet_data()
    check_stanford3d_data()
    
    print("\n" + "="*70)
    print("분석 완료!")

if __name__ == "__main__":
    main()
