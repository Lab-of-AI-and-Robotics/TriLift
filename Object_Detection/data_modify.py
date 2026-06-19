import os
import glob
import time
import argparse
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------
# Example Usage:
# ---------------
# python data_modify.py --dataset front3d
# python data_modify.py --dataset hypersim
# python data_modify.py --dataset scannet

def main(dataset):
    assert dataset in ['front3d', 'scannet', 'hypersim'], f"Unsupported dataset: {dataset}"
    
    feature_path = os.path.join(BASE_DIR, 'data', dataset, f'{dataset}_rpn_data', 'features')

    output_path = os.path.join(BASE_DIR, 'cube_results', dataset)
    os.makedirs(output_path, exist_ok=True)

    npz_paths = glob.glob(os.path.join(feature_path, '*'))

    for npz_path in tqdm(npz_paths):
        if os.path.isfile(npz_path):
            cuboid_rgbsigma = np.load(npz_path)

            res = cuboid_rgbsigma['resolution']
            rgbsigma = cuboid_rgbsigma['rgbsigma']
            max_res = max(res)
            max_rgbsigma_shape = (max_res, max_res, max_res, 4)

            cube_rgbsigma = np.zeros(max_rgbsigma_shape, dtype=np.float32)
            cube_rgbsigma[:res[0], :res[1], :res[2], :] = rgbsigma

            output_npz = os.path.join(output_path, os.path.basename(npz_path))                
            np.savez_compressed(output_npz,
                                rgbsigma=cube_rgbsigma,
                                resolution=[max_res] * 3,
                                bbox_min=cuboid_rgbsigma['bbox_min'],
                                bbox_max=cuboid_rgbsigma['bbox_max'],
                                scale=cuboid_rgbsigma['scale'],
                                offset=cuboid_rgbsigma['offset'],
                                from_mitsuba=cuboid_rgbsigma['from_mitsuba'])
            time.sleep(0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['front3d', 'scannet', 'hypersim'],
                        help="Dataset name to process.")
    args = parser.parse_args()
    main(args.dataset)