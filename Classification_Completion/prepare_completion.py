import os
import sys
import argparse
import numpy as np
import glob
import torch
import logging
from tqdm import tqdm
import open3d as o3d
from scipy.ndimage import binary_dilation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 스크립트 자신의 디렉토리 기준으로 경로 고정 (cwd 무관)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PointCloudToVoxel:
    """
    Class for converting point clouds to voxel grids with gap filling
    """
    def __init__(self, resolution=128):
        self.resolution = resolution
    
    def normalize_point_cloud(self, points):
        """Normalize point cloud to [0, 1] range"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        scale = np.max(max_coords - min_coords)
        points = (points - min_coords) / scale
        return points
    
    def convert_to_voxel(self, points):
        """
        점 구름을 복셀 그리드로 변환하고 빈 공간을 채움
        
        Args:
            points: 점 구름 데이터 (Nx3 array)
            
        Returns:
            voxel_grid: 빈 공간이 채워진 복셀 그리드
        """
        # Normalize point cloud
        points = self.normalize_point_cloud(points)
        
        # Initialize voxel grid
        voxel_grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.bool_)
        
        # Convert each point to voxel index
        indices = np.floor(points * (self.resolution - 1)).astype(int)
        
        # Set each voxel at the corresponding index
        valid_indices = np.all((indices >= 0) & (indices < self.resolution), axis=1)
        indices = indices[valid_indices]
        
        # Mark points in the voxel grid
        for i, j, k in indices:
            voxel_grid[i, j, k] = True
        
        # Apply morphological dilation to fill gaps
        # 3x3x3 structuring element for moderate gap filling
        struct_element = np.ones((3, 3, 3), dtype=bool)
        voxel_grid = binary_dilation(voxel_grid, structure=struct_element, iterations=2)
        
        return voxel_grid

class ModelNet40Processor:
    """
    Class for processing ModelNet40 dataset
    """
    def __init__(self, root_dir="./datasets/ModelNet40", output_dir="./processed_data", 
                 resolution=128, class_name="chair", save_data=True, n_points=100000):
        self.root_dir = root_dir
        self.class_name = class_name
        self.resolution = resolution
        self.output_dir = output_dir
        self.save_data = save_data
        self.n_points = n_points  # 많은 점으로 빈 공간 최소화
        self.converter = PointCloudToVoxel(resolution=resolution)
        
    def load_off_file(self, file_path):
        """Load mesh from OFF file"""
        mesh = o3d.io.read_triangle_mesh(file_path)
        return mesh
    
    def mesh_to_point_cloud(self, mesh, n_points=None):
        """Convert mesh to dense point cloud for better coverage"""
        if n_points is None:
            n_points = self.n_points
            
        # Prepare mesh for point sampling
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Sample many points for dense coverage
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        points = np.asarray(pcd.points)
        
        return points
    
    def process_dataset(self, phase="train"):
        """Process dataset for a specific class and save to disk"""
        logger.info(f"Processing {self.class_name} {phase} dataset with resolution {self.resolution}x{self.resolution}x{self.resolution}")
        logger.info(f"Dense sampling with {self.n_points} points per mesh + gap filling")
        
        # Construct file paths
        data_dir = os.path.join(self.root_dir, f"{self.class_name}/{phase}")
        off_files = glob.glob(os.path.join(data_dir, "*.off"))
        
        if not off_files:
            logger.error(f"No OFF files found in {data_dir}")
            return
        
        logger.info(f"Found {len(off_files)} files")
        
        # Create output directory if saving data
        output_dir = None
        if self.save_data:
            output_dir = os.path.join(self.output_dir, f"completion_{self.class_name}_{phase}_res{self.resolution}")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            
            # Create metadata file
            with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
                f.write(f"class: {self.class_name}\n")
                f.write(f"phase: {phase}\n")
                f.write(f"resolution: {self.resolution}\n")
                f.write(f"num_files: {len(off_files)}\n")
                f.write(f"n_points: {self.n_points}\n")
                f.write(f"gap_filling: enabled\n")
        
        # Data processing statistics
        voxel_stats = []
        occupancy_stats = []
        
        # Process each OFF file
        for i, file_path in enumerate(tqdm(off_files, desc=f"Converting {phase} data")):
            try:
                # Get filename without extension for saving
                filename = os.path.splitext(os.path.basename(file_path))[0]
                
                # Load mesh
                mesh = self.load_off_file(file_path)
                
                # Sample point cloud from mesh
                points = self.mesh_to_point_cloud(mesh)
                
                # Convert point cloud to voxel grid with gap filling
                voxel_grid = self.converter.convert_to_voxel(points)
                
                # Convert to tensor for saving
                voxel_tensor = torch.from_numpy(voxel_grid).float()
                
                # Collect statistics
                voxel_stats.append(voxel_grid.shape)
                occupancy_ratio = np.sum(voxel_grid) / voxel_grid.size
                occupancy_stats.append(occupancy_ratio)
                
                # Save voxel grid if requested
                if self.save_data:
                    save_path = os.path.join(output_dir, f"{filename}.pt")
                    data_to_save = {
                        'occupancy_grid': voxel_tensor,
                        'label': 0,  # Since we're focusing on one class
                        'filename': filename
                    }
                    torch.save(data_to_save, save_path)
                
                # Print statistics for first, middle, and last samples
                if i == 0 or i == len(off_files) // 2 or i == len(off_files) - 1:
                    logger.info(f"Sample {i+1}/{len(off_files)}: Shape={voxel_grid.shape}, "
                               f"Occupancy={np.sum(voxel_grid)}/{voxel_grid.size} ({occupancy_ratio:.2%})")
            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Print overall dataset statistics
        logger.info(f"\n--- {self.class_name} {phase} Dataset Statistics ---")
        logger.info(f"Total samples: {len(off_files)}")
        logger.info(f"Voxel grid shape: {self.resolution}x{self.resolution}x{self.resolution}")
        logger.info(f"Average occupancy ratio: {np.mean(occupancy_stats):.2%}")
        logger.info(f"Min occupancy ratio: {np.min(occupancy_stats):.2%}")
        logger.info(f"Max occupancy ratio: {np.max(occupancy_stats):.2%}")
        
        if self.save_data:
            logger.info(f"Data saved to: {output_dir}")
        
        return {
            "total_samples": len(off_files),
            "shape": (self.resolution, self.resolution, self.resolution),
            "avg_occupancy": np.mean(occupancy_stats),
            "min_occupancy": np.min(occupancy_stats),
            "max_occupancy": np.max(occupancy_stats),
            "output_dir": output_dir if self.save_data else None
        }
    
    def process_all(self):
        """Process both train and test datasets"""
        logger.info(f"Starting conversion of ModelNet40 {self.class_name} dataset to {self.resolution}^3 voxel grids")
        logger.info(f"Using dense point sampling + gap filling for solid shapes")
        
        # Check directory
        class_dir = os.path.join(self.root_dir, self.class_name)
        if not os.path.exists(class_dir):
            logger.error(f"Class directory not found: {class_dir}")
            return
        
        # Process train and test datasets
        train_stats = self.process_dataset("train")
        test_stats = self.process_dataset("test")
        
        return {
            "train": train_stats,
            "test": test_stats
        }

def main():
    parser = argparse.ArgumentParser(description="Convert ModelNet40 dataset to solid voxel grids with gap filling")
    parser.add_argument("--resolution", type=int, default=128, help="Voxel grid resolution (default: 128)")
    parser.add_argument("--root", type=str, default=os.path.join(BASE_DIR, "datasets", "ModelNet40"), help="Root directory of ModelNet40 dataset")
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "processed_data"), help="Output directory for processed data")
    parser.add_argument("--class_name", type=str, default="chair", help="Class name to process (default: chair)")
    parser.add_argument("--save", action="store_true", help="Save processed data to disk")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Don't save data, just show statistics")
    parser.add_argument("--n_points", type=int, default=100000, 
                       help="Number of points to sample from mesh (default: 100000)")
    parser.set_defaults(save=True)
    
    args = parser.parse_args()
    
    # Initialize and run processor
    processor = ModelNet40Processor(
        root_dir=args.root,
        output_dir=args.output,
        resolution=args.resolution,
        class_name=args.class_name,
        save_data=args.save,
        n_points=args.n_points
    )
    
    stats = processor.process_all()
    
    if stats:
        logger.info("\n--- Final Dataset Statistics ---")
        logger.info(f"Class: {args.class_name}")
        logger.info(f"Resolution: {args.resolution}x{args.resolution}x{args.resolution}")
        logger.info(f"Dense point sampling: {args.n_points} points per mesh")
        logger.info(f"Gap filling: enabled")
        logger.info(f"Train samples: {stats['train']['total_samples']}")
        logger.info(f"Test samples: {stats['test']['total_samples']}")
        logger.info(f"Train average occupancy: {stats['train']['avg_occupancy']:.2%}")
        logger.info(f"Test average occupancy: {stats['test']['avg_occupancy']:.2%}")
        
        if args.save:
            logger.info(f"\nTrain data saved to: {stats['train']['output_dir']}")
            logger.info(f"Test data saved to: {stats['test']['output_dir']}")
            logger.info("\nSolid voxel data generation completed!")
        else:
            logger.info("\nData processing completed (no data saved)!")

if __name__ == "__main__":
    main() 