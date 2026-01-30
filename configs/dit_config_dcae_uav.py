# Epona Configuration for UAV (EuRoC) Dataset
# =============================================
# This config is adapted from dit_config_dcae_nuplan.py for UAV datasets
# converted from EuRoC MAV ROS bags.

# Random seed
seed = 1234

#! Dataset paths
# Point these to your converted UAV dataset location
datasets_paths = dict(
    nuscense_root='',
    nuscense_train_json_path='',
    nuscense_val_json_path='',
    
    # UAV dataset paths (converted from EuRoC bags)
    nuplan_root='/home/dataset-local/uav_epona_dataset',
    nuplan_json_root='/home/dataset-local/uav_epona_dataset',
)

# Use nuplan loader since our UAV data is converted to nuplan format
train_data_list = ['nuplan']
val_data_list = ['nuplan']

# Frame rate settings
# EuRoC original: ~20 Hz, downsampled to 10 Hz in conversion
# If you used target_fps: 10 in config.yaml, set downsample_fps=5
# If you kept original ~20 Hz, set downsample_fps=10
downsample_fps = 5  # Adjust based on your conversion settings

mask_data = 0  # 1 means all masked, 0 means all gt

# Image size
# EuRoC camera resolution: 752x480, but DCAE requires dimensions divisible by 32
# 752 is not divisible by 32, so we use 768 (24*32) or match nuplan's 1024
# Option 1: Similar aspect ratio to EuRoC: (480, 768) - 480/32=15, 768/32=24 ✓
# Option 2: Match nuplan for better pretrained model compatibility: (512, 1024)
image_size = (512, 1024)  # Match nuplan for best pretrained model performance

pkeep = 0.7  # Percentage for how much latent codes to keep
reverse_seq = False
paug = 0

# VAE configs
vae_embed_dim = 32
downsample_size = 32
patch_size = 1
vae = 'DCAE_f32c32'
vae_ckpt = '/home/dataset-local/Epona_data/dcae_td_20000.pkl'  #! VAE checkpoint path
add_encoder_temporal = False
add_decoder_temporal = True
temporal_patch_size = 6

# World Model configs
condition_frames = 10
n_layer = [12, 6, 6]
n_head = 16
n_embd = 2048
gpt_type = 'diffgpt_mar'
pose_x_vocab_size = 128
pose_y_vocab_size = 128
yaw_vocab_size = 512

# Logs
outdir = "exp/ckpt_uav"
logdir = "exp/job_log_uav"
tdir = "exp/job_tboard_uav"
validation_dir = "exp/validation_uav"

diffusion_model_type = "flow"
num_sampling_steps = 100
lambda_yaw_pose = 1.0

diff_only = True
forward_iter = 3
multifw_perstep = 10
block_size = 1

n_embd_dit = 2048
n_head_dit = 16
axes_dim_dit = [16, 56, 56]
return_predict = True

traj_len = 15
n_layer_traj = [1, 1]
n_embd_dit_traj = 1024
n_head_dit_traj = 8
axes_dim_dit_traj = [16, 56, 56]
return_predict_traj = True

fix_stt = False
test_video_frames = 50
drop_feature = 0
no_pose = False
sample_prob = [1.0]

# UAV-specific pose bounds
# These may need adjustment based on your UAV trajectories
# EuRoC datasets have different motion patterns than driving
pose_x_bound = 50  # meters, adjust based on UAV motion range
pose_y_bound = 10  # meters, adjust based on UAV motion range  
yaw_bound = 12     # degrees per frame, UAV can have more aggressive rotations
