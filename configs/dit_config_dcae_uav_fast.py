# Epona Configuration for UAV (EuRoC) Dataset - Fast Version
# ===========================================================
# This config uses the optimized UAVDatasetFast with cv2 for faster data loading.
# Use this config for faster training with pre-resized images.
#
# Key differences from dit_config_dcae_uav.py:
# - Uses 'uav_fast' dataset loader (cv2-based, ~3x faster)
# - skip_resize=True for pre-resized images (set to False if not pre-resized)

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

# Use uav_fast for cv2-based fast loading
train_data_list = ['uav_fast']
val_data_list = ['uav_fast']

# UAV-specific settings
uav_ori_fps = 10  # Original FPS after conversion (10 if converted with target_fps=10)
skip_resize = false  # Set to True if images are pre-resized during conversion

# Frame rate settings
downsample_fps = 5  # Adjust based on your conversion settings

mask_data = 0  # 1 means all masked, 0 means all gt

# Image size (should match the pre-resized dimensions from conversion)
# If you set resize_images: [512, 1024] in uav_to_epona config.yaml, use this:
image_size = (512, 1024)

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
outdir = "exp/ckpt_uav_fast"
logdir = "exp/job_log_uav_fast"
tdir = "exp/job_tboard_uav_fast"
validation_dir = "exp/validation_uav_fast"

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
pose_x_bound = 50  # meters
pose_y_bound = 10  # meters
yaw_bound = 12     # degrees per frame
