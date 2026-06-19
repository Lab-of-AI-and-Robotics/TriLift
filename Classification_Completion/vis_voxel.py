#!/usr/bin/env python3
"""
복셀 데이터를 rerun으로 시각화하는 스크립트
prepare_completion.py로 생성된 .pt 파일을 로드하여 3D로 표시합니다.
"""

import argparse
import os
import numpy as np
import torch
import rerun as rr


def load_voxel_data(file_path):
    """
    .pt 파일에서 복셀 데이터를 로드합니다.
    
    Args:
        file_path: .pt 파일 경로
        
    Returns:
        data: 로드된 데이터 딕셔너리
        voxel_grid: 복셀 그리드 (numpy array)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    # 데이터 로드
    data = torch.load(file_path, map_location='cpu')
    
    # 복셀 그리드 추출
    voxel_grid = data['occupancy_grid'].numpy()
    
    print(f"파일 로드 완료: {file_path}")
    print(f"파일명: {data.get('filename', 'unknown')}")
    print(f"복셀 그리드 크기: {voxel_grid.shape}")
    print(f"활성화된 복셀 수: {np.sum(voxel_grid > 0.5)}")
    print(f"점유율: {np.sum(voxel_grid > 0.5) / voxel_grid.size * 100:.2f}%")
    
    return data, voxel_grid


def voxel_to_points(voxel_grid, threshold=0.5):
    """
    복셀 그리드에서 활성화된 복셀들의 3D 좌표를 추출합니다.
    
    Args:
        voxel_grid: 복셀 그리드 (3D numpy array)
        threshold: 복셀이 활성화된 것으로 간주할 임계값
        
    Returns:
        points: 활성화된 복셀들의 3D 좌표 (N, 3)
        colors: 각 점의 색상 (N, 3)
    """
    # 활성화된 복셀의 인덱스 찾기
    active_indices = np.where(voxel_grid > threshold)
    
    # 3D 좌표로 변환
    points = np.column_stack(active_indices).astype(np.float32)
    
    # 복셀 중심으로 이동 (각 복셀의 중앙점)
    points += 0.5
    
    # 색상 설정 (높이에 따라 그라디언트)
    if len(points) > 0:
        # Z축(높이) 기반 색상 그라디언트
        z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
        colors = np.column_stack([
            z_normalized,           # Red
            1.0 - z_normalized,     # Green  
            np.full(len(points), 0.7)  # Blue
        ])
    else:
        colors = np.array([]).reshape(0, 3)
    
    return points, colors


def visualize_voxel(file_path, app_id="voxel_viewer"):
    """
    복셀 데이터를 rerun으로 시각화합니다.
    
    Args:
        file_path: .pt 파일 경로
        app_id: rerun 애플리케이션 ID
    """
    # Rerun 초기화 (spawn=True로 뷰어 자동 실행)
    rr.init(app_id, spawn=True)
    
    # 데이터 로드
    data, voxel_grid = load_voxel_data(file_path)
    
    # 복셀을 3D 점으로 변환
    points, colors = voxel_to_points(voxel_grid)
    
    if len(points) == 0:
        print("❌ 활성화된 복셀이 없습니다!")
        return
    
    # 파일명 추출
    filename = data.get('filename', os.path.basename(file_path))
    
    # 3D 점으로 시각화
    rr.log(
        f"voxel/{filename}",
        rr.Points3D(
            positions=points,
            colors=colors,
            radii=0.3  # 점 크기
        )
    )
    
    # 축 표시
    resolution = voxel_grid.shape[0]
    rr.log(
        "axes",
        rr.Points3D(
            positions=[[0, 0, 0], [resolution, 0, 0], [0, resolution, 0], [0, 0, resolution]],
            colors=[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            radii=1.0
        )
    )
    
    # 메타데이터 로그
    rr.log(
        "metadata",
        rr.TextLog(
            f"""
복셀 데이터 정보:
- 파일: {filename}
- 해상도: {voxel_grid.shape}
- 활성화된 복셀: {len(points)}
- 점유율: {len(points) / voxel_grid.size * 100:.2f}%
            """.strip()
        )
    )
    
    print(f"✅ 시각화 완료!")
    print(f"📊 총 {len(points)}개의 복셀이 표시되었습니다.")
    print(f"🎨 높이에 따른 색상 그라디언트가 적용되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="복셀 데이터를 rerun으로 시각화")
    parser.add_argument("file_path", type=str, help=".pt 파일 경로")
    parser.add_argument("--app_id", type=str, default="voxel_viewer", help="Rerun 앱 ID")
    
    args = parser.parse_args()
    
    try:
        visualize_voxel(args.file_path, args.app_id)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main() 