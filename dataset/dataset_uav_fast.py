"""
UAV (EuRoC) Dataset Loader for Epona - Fast Version

This is an optimized version of dataset_uav.py that uses cv2 instead of PIL
for faster image loading and resizing.

Performance improvements:
- cv2.imread() is ~3x faster than PIL.Image.open()
- cv2.resize() is ~4x faster than PIL.Image.resize()
- Skip resize if images are pre-resized during conversion

Usage:
    In configs/dit_config_dcae_uav.py, the dataset is loaded via create_dataset.py
    Modify create_dataset.py to use UAVDatasetFast instead of UAVDataset
"""

import json
import cv2
import os
import math
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R


def radians_to_degrees(radians):
    degrees = radians * (180 / math.pi)
    return degrees


def get_meta_data(poses, condition_frames):
    """Get relative pose metadata from absolute poses."""
    poses = np.concatenate([poses[0:1], poses], axis=0)
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    xyzs = rel_pose[:, :3, 3]
    xys = xyzs[:, :2]
    rel_yaws = radians_to_degrees(
        R.from_matrix(rel_pose[:, :3, :3]).as_euler('zyx', degrees=False)[:, 0]
    )[:, np.newaxis]

    return {
        'rel_poses': xys,
        'rel_yaws': rel_yaws,
    }


class UAVDatasetFast(Dataset):
    """
    UAV Dataset loader for Epona - Fast version using cv2.
    
    This is an optimized version that uses cv2 for image loading,
    which is significantly faster than PIL.
    """
    
    def __init__(
        self, 
        data_root, 
        json_root, 
        cache_path=None, 
        vae=None, 
        split='train', 
        condition_frames=3, 
        block_size=1, 
        downsample_fps=3, 
        downsample_size=16, 
        h=256, 
        w=512, 
        no_pose=False, 
        clip_num=10000, 
        augmenter=None, 
        paug=0.9,
        ori_fps=10,
        skip_resize=False,  # Set to True if images are pre-resized
    ):
        """
        Initialize UAV dataset (fast version).
        
        Args:
            data_root: Root directory containing sensor_blobs/
            json_root: Directory containing meta JSON files
            split: 'train' or 'test'
            condition_frames: Number of conditioning frames
            block_size: Block size for training
            downsample_fps: Target frame rate after downsampling
            h, w: Output image size
            no_pose: If True, don't use pose data
            ori_fps: Original frame rate of the data
            skip_resize: If True, skip resizing (for pre-resized data)
        """
        self.split = split
        if split == 'train':
            self.meta_path = f'{json_root}/train_meta.json'
            self.pose_meta_path = f'{json_root}/ego_meta'
        elif split == 'test':
            self.meta_path = f'{json_root}/test_meta.json'
            self.pose_meta_path = f'{json_root}/test_ego_meta'
        else:
            self.meta_path = f'{json_root}/test_meta.json'
            self.pose_meta_path = f'{json_root}/test_ego_meta'

        self.condition_frames = condition_frames
        self.block_size = block_size
        self.data_root = data_root
        self.cache_path = cache_path
        self.vae = vae
        
        self.ori_fps = ori_fps
        self.downsample = max(1, self.ori_fps // downsample_fps)
        self.h = h
        self.w = w
        self.no_pose = no_pose
        self.skip_resize = skip_resize

        # Load preprocessed meta
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                f"Meta file not found: {self.meta_path}\n"
                f"Please run convert_uav_to_epona.py first to generate the dataset."
            )
        
        with open(self.meta_path, 'r') as f:
            json_data = json.load(f)

        # Filter sequences with enough frames
        json_data_filter = []
        for data in json_data:
            if len(data['CAM_F0']) > self.condition_frames * self.downsample:
                json_data_filter.append(data)
        
        if len(json_data_filter) == 0:
            raise ValueError(
                f"No valid sequences found in {self.meta_path}. "
                f"Required frames: {self.condition_frames * self.downsample}"
            )
        
        self.sequences = json_data_filter
        self.downsample_size = downsample_size
        
        print(f"UAV Dataset (Fast) initialized:")
        print(f"  - Split: {split}")
        print(f"  - Sequences: {len(self.sequences)}")
        print(f"  - Original FPS: {self.ori_fps}")
        print(f"  - Downsample factor: {self.downsample}")
        print(f"  - Condition frames: {self.condition_frames}")
        print(f"  - Skip resize: {self.skip_resize}")
        print(f"  - Image loader: cv2 (fast)")
    
    def __len__(self):
        return len(self.sequences)
    
    def load_pose(self, pose_path, front_cam_list):
        """Load pose data from JSON file."""
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        front_cam_pose = pose_data['CAM_F0'] if 'CAM_F0' in pose_data else pose_data
        
        poses = {}
        for key, pose_meta in front_cam_pose.items():
            poses[key] = [
                pose_meta['x'], pose_meta['y'], pose_meta['z'],
                pose_meta['qx'], pose_meta['qy'], pose_meta['qz'], pose_meta['qw'],
            ]
        
        poses_filter = np.array([poses[f"CAM_F0/{ts}"] for ts in front_cam_list])
        return poses_filter
    
    def normalize_imgs(self, imgs):
        """Normalize images to [-1, 1] range."""
        imgs = imgs / 255.0
        imgs = (imgs - 0.5) * 2
        return imgs

    def __loadarray_tum_single(self, array):
        """Convert pose array to 4x4 transformation matrix."""
        absolute_transforms = np.zeros((4, 4))
        absolute_transforms[3, 3] = 1
        absolute_transforms[:3, :3] = R.from_quat(array[3:7]).as_matrix()
        absolute_transforms[:3, 3] = array[0:3]
        return absolute_transforms
        
    def downsample_sequences(self, img_ts, poses):
        """Downsample image sequence and poses to target FPS."""
        ori_size = len(img_ts)
        assert len(img_ts) == len(poses)
        index_list = np.arange(0, ori_size, step=self.downsample)
        img_ts_downsample = np.array(img_ts)[index_list]
        poses_downsample = poses[index_list]
        return img_ts_downsample, poses_downsample

    def getimg(self, index):
        """Get images and poses for a given sequence index (using cv2)."""
        seq_data = self.sequences[index]

        seq_root = os.path.join(self.data_root, seq_data['data_root'])
        seq_db_name = os.path.basename(seq_root)
        pose_path = f"{self.pose_meta_path}/{seq_db_name}.json"

        rgb_front_dir = f"{seq_root}/CAM_F0"
        
        try:
            poses = self.load_pose(pose_path, seq_data['CAM_F0'])
        except Exception as e:
            print(f'Warning: Failed to load pose from {pose_path}: {e}')
            return None, None

        # Downsample fps
        img_ts_downsample, poses_downsample = self.downsample_sequences(
            seq_data['CAM_F0'], poses
        )
        clip_length = len(img_ts_downsample)
        
        if clip_length < self.condition_frames + self.block_size:
            print(f'Warning: Sequence too short: {clip_length} < {self.condition_frames + self.block_size}')
            return None, None
        
        if self.split == "val" or self.split == "test":
            start = 0
        else:
            start = random.randint(0, clip_length - self.condition_frames - self.block_size)
        
        ims = []
        poses_new = []
        
        for i in range(self.condition_frames + self.block_size):
            img_path = f"{rgb_front_dir}/{img_ts_downsample[start + i]}"
            
            # Use cv2 for faster loading
            im = cv2.imread(img_path)
            if im is None:
                print(f'Warning: Failed to load image {img_path}')
                return None, None
            
            # Convert BGR to RGB (cv2 loads as BGR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            # Convert grayscale to RGB if needed (single channel)
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            
            # Resize if not pre-resized (use INTER_LINEAR for speed)
            if not self.skip_resize:
                im = cv2.resize(im, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
            ims.append(im)
            poses_new.append(self.__loadarray_tum_single(poses_downsample[start + i]))
        
        poses_yaw = np.array(poses_new)
        return ims, poses_yaw

    def __getitem__(self, index):
        """Get a training sample."""
        max_retries = 100
        retry_count = 0
        
        while retry_count < max_retries:
            imgs, poses = self.getimg(index)
            if (imgs is not None) and (poses is not None):
                break
            else:
                index = random.randint(0, self.__len__() - 1)
                retry_count += 1
        
        if imgs is None or poses is None:
            raise RuntimeError(f"Failed to load data after {max_retries} retries")
        
        imgs_tensor = []
        poses_tensor = []
        
        for img, pose in zip(imgs, poses):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
        
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        
        if self.no_pose:
            return imgs, torch.tensor(0.0)
        else:
            return imgs, torch.stack(poses_tensor, 0).float()
    
    def check_data(self, index, imgs, ps):
        """Debug function to save images and print poses."""
        save_dir = f'./check_uav_data_fast/{index}'
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            print(index, i, ps[i])
            # Convert RGB to BGR for cv2.imwrite
            cv2.imwrite(save_dir + f'/{i}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# Factory function for easy creation
def create_uav_dataset_fast(
    data_root,
    json_root,
    split='test',
    condition_frames=10,
    block_size=1,
    downsample_fps=5,
    h=512,
    w=1024,
    no_pose=False,
    ori_fps=10,
    skip_resize=False,
):
    """
    Create a fast UAV dataset instance using cv2.
    
    Args:
        data_root: Root directory of converted UAV data
        json_root: Directory containing JSON metadata
        split: 'train' or 'test'
        condition_frames: Number of conditioning frames
        block_size: Block size for prediction
        downsample_fps: Target frame rate
        h, w: Output image dimensions
        no_pose: If True, don't load pose data
        ori_fps: Original frame rate of data
        skip_resize: If True, skip resize (for pre-resized data)
        
    Returns:
        UAVDatasetFast instance
    """
    return UAVDatasetFast(
        data_root=data_root,
        json_root=json_root,
        split=split,
        condition_frames=condition_frames,
        block_size=block_size,
        downsample_fps=downsample_fps,
        h=h,
        w=w,
        no_pose=no_pose,
        ori_fps=ori_fps,
        skip_resize=skip_resize,
    )
