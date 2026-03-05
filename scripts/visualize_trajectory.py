import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_trajectory(traj_path):
    """Load trajectory file"""
    traj = torch.load(traj_path)
    print(f"Original shape: {traj.shape}")
    
    # If 2D, reshape to (N, 15, 3)
    if len(traj.shape) == 2:
        if traj.shape[1] == 3:
            total_frames = traj.shape[0]
            if total_frames % 15 == 0:
                n_steps = total_frames // 15
                traj = traj.reshape(n_steps, 15, 3)
                print(f"Reshaped to: {traj.shape} = ({n_steps} steps x 15 future frames x 3 dims)")
    
    return traj

def compute_absolute_trajectory(rel_traj):
    """Convert relative trajectory to absolute positions"""
    if isinstance(rel_traj, torch.Tensor):
        rel_traj = rel_traj.numpy()
    
    # Handle different shapes
    if len(rel_traj.shape) == 3:
        # Shape (N, 15, 3) - use first prediction of each step
        next_poses = rel_traj[:, 0, :2]
        next_yaws = rel_traj[:, 0, 2]
    else:
        # Shape (N, 3)
        next_poses = rel_traj[:, :2]
        next_yaws = rel_traj[:, 2]
    
    positions = np.zeros((len(next_poses) + 1, 2))
    absolute_yaw = 0.0
    yaw_history = [0.0]
    
    for i in range(len(next_poses)):
        cos_yaw = np.cos(np.radians(absolute_yaw))
        sin_yaw = np.sin(np.radians(absolute_yaw))
        rotation = np.array([[cos_yaw, -sin_yaw],
                           [sin_yaw, cos_yaw]])
        global_displacement = rotation @ next_poses[i]
        positions[i+1] = positions[i] + global_displacement
        absolute_yaw += next_yaws[i]
        yaw_history.append(absolute_yaw)
    
    return positions, yaw_history

def visualize_trajectory_2d_with_time(traj, save_path, arrow_interval=20, gt_traj=None):
    """
    2D visualization with clearer time representation using color segments
    """
    print("Generating 2D trajectory visualization with time segments...")
    
    # Compute predicted trajectory
    positions, yaw_history = compute_absolute_trajectory(traj)
    
    # Compute GT trajectory if provided
    gt_positions, gt_yaw_history = None, None
    if gt_traj is not None:
        gt_positions, gt_yaw_history = compute_absolute_trajectory(gt_traj)
        # Truncate to same length
        min_len = min(len(positions), len(gt_positions))
        gt_positions = gt_positions[:min_len]
        gt_yaw_history = gt_yaw_history[:min_len]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Main trajectory plot
    ax1 = plt.subplot(1, 2, 1)
    
    # Calculate extent for both trajectories
    all_positions = positions
    if gt_positions is not None:
        all_positions = np.vstack([positions, gt_positions])
    
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    trajectory_extent = max(x_range, y_range, 1.0)
    
    # Scale marker size proportionally to trajectory extent
    scale_factor = trajectory_extent / 20.0
    scale_factor = np.clip(scale_factor, 0.1, 3.0)
    
    # Marker size for direction triangles
    marker_size = 150 * scale_factor
    
    print(f"Trajectory extent: {trajectory_extent:.2f}m, Arrow scale: {scale_factor:.2f}")
    
    # Draw GT trajectory first (as dashed line)
    if gt_positions is not None:
        ax1.plot(gt_positions[:, 0], gt_positions[:, 1],
                'k--', linewidth=2, alpha=0.6, label='Ground Truth', zorder=3)
        
        # GT direction markers (gray triangles on the line)
        gt_arrow_indices = list(range(0, len(gt_positions)-1, arrow_interval))
        for i in gt_arrow_indices:
            x, y = gt_positions[i]
            yaw = gt_yaw_history[i]
            # Create rotated triangle marker
            marker = (3, 0, yaw - 90)  # triangle pointing in yaw direction
            ax1.scatter(x, y, marker=marker, s=marker_size*0.6, 
                       c='gray', edgecolors='black', linewidth=1, alpha=0.7, zorder=4)
    
    # Draw predicted trajectory with time-based color segments
    n_segments = 10
    segment_size = max(1, len(positions) // n_segments)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_segments))
    
    for seg in range(n_segments):
        start_idx = seg * segment_size
        end_idx = min((seg + 1) * segment_size + 1, len(positions))
        
        ax1.plot(positions[start_idx:end_idx, 0], 
                positions[start_idx:end_idx, 1],
                color=colors[seg], linewidth=3, alpha=0.8,
                label=f'Pred {start_idx}-{end_idx-1}')
    
    # Draw orientation markers at regular intervals (triangles on the line)
    arrow_indices = list(range(0, len(positions)-1, arrow_interval))
    if (len(positions)-2) not in arrow_indices:
        arrow_indices.append(len(positions)-2)
    
    print(f"Drawing {len(arrow_indices)} direction markers at positions: {arrow_indices[:10]}... (showing first 10)")
    
    for i in arrow_indices:
        x, y = positions[i]
        yaw = yaw_history[i]
        
        color_idx = int(i / len(positions) * n_segments)
        color_idx = min(color_idx, n_segments - 1)
        
        # Create rotated triangle marker pointing in yaw direction
        marker = (3, 0, yaw - 90)  # triangle with rotation
        ax1.scatter(x, y, marker=marker, s=marker_size, 
                   c=[colors[color_idx]], edgecolors='black', linewidth=1.5, 
                   alpha=0.9, zorder=5)
        
        # Frame number label (offset based on scale)
        text_offset = 0.6 * scale_factor
        ax1.text(x - text_offset, y - text_offset, str(i),
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.15', 
                         facecolor='white', 
                         edgecolor='gray', 
                         alpha=0.7),
                zorder=6)
    
    # Mark start and end
    ax1.scatter(positions[0, 0], positions[0, 1],
               c='green', s=400, marker='o',
               label='Start', zorder=10,
               edgecolors='black', linewidth=3)
    
    ax1.scatter(positions[-1, 0], positions[-1, 1],
               c='red', s=400, marker='X',
               label='Pred End', zorder=10,
               edgecolors='black', linewidth=3)
    
    if gt_positions is not None:
        ax1.scatter(gt_positions[-1, 0], gt_positions[-1, 1],
                   c='blue', s=300, marker='X',
                   label='GT End', zorder=10,
                   edgecolors='black', linewidth=2)
    
    # Statistics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    total_rotation = yaw_history[-1]
    
    stats_lines = [
        f'Pred Distance: {total_distance:.2f} m',
        f'Pred Rotation: {total_rotation:.2f} deg',
        f'Avg Speed: {total_distance/max(1,len(positions)-1):.3f} m/frame',
        f'Arrow scale: {scale_factor:.2f}x'
    ]
    
    if gt_positions is not None:
        gt_distance = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
        gt_rotation = gt_yaw_history[-1]
        # Calculate ATE (Absolute Trajectory Error)
        min_len = min(len(positions), len(gt_positions))
        ate = np.sqrt(np.mean(np.sum((positions[:min_len] - gt_positions[:min_len])**2, axis=1)))
        # Final position error
        final_error = np.linalg.norm(positions[min_len-1] - gt_positions[min_len-1])
        
        stats_lines.extend([
            f'GT Distance: {gt_distance:.2f} m',
            f'GT Rotation: {gt_rotation:.2f} deg',
            f'ATE: {ate:.3f} m',
            f'Final Error: {final_error:.3f} m'
        ])
    
    stats_text = '\n'.join(stats_lines)
    
    ax1.text(0.02, 0.02, stats_text,
            transform=ax1.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    
    ax1.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    title = f'Vehicle Trajectory (Color = Time Progression)\nTotal Frames: {len(positions)-1}'
    if gt_positions is not None:
        title += ' | Black dashed = GT'
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    
    # Time evolution plot (X, Y, Yaw vs Time)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 4)
    
    time_steps = np.arange(len(positions))
    ax2.plot(time_steps, positions[:, 0], 'b-', linewidth=2, label='Pred X')
    ax2.plot(time_steps, positions[:, 1], 'r-', linewidth=2, label='Pred Y')
    if gt_positions is not None:
        gt_time = np.arange(len(gt_positions))
        ax2.plot(gt_time, gt_positions[:, 0], 'b--', linewidth=1.5, alpha=0.6, label='GT X')
        ax2.plot(gt_time, gt_positions[:, 1], 'r--', linewidth=1.5, alpha=0.6, label='GT Y')
    ax2.set_xlabel('Time Step (Frame)', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title('X & Y Positions over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(time_steps, yaw_history, 'g-', linewidth=2, label='Pred Yaw')
    if gt_yaw_history is not None:
        gt_time = np.arange(len(gt_yaw_history))
        ax3.plot(gt_time, gt_yaw_history, 'g--', linewidth=1.5, alpha=0.6, label='GT Yaw')
    ax3.set_xlabel('Time Step (Frame)', fontsize=12)
    ax3.set_ylabel('Yaw Angle (degrees)', fontsize=12)
    ax3.set_title('Vehicle Orientation over Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.close()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total frames: {len(positions)-1}")
    print(f"  Pred distance: {total_distance:.2f} m")
    print(f"  Pred rotation: {total_rotation:.2f} deg")
    if gt_positions is not None:
        print(f"  GT distance: {gt_distance:.2f} m")
        print(f"  ATE: {ate:.3f} m")
        print(f"  Final position error: {final_error:.3f} m")


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory with clear time axis')
    parser.add_argument('--traj_path', type=str, required=True,
                       help='pred_traj.pt file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: trajectory_2d_timeaxis.png)')
    parser.add_argument('--arrow_interval', type=int, default=20,
                       help='Arrow interval (default: 20)')
    
    args = parser.parse_args()
    
    # Set output path
    output_dir = os.path.dirname(args.traj_path)
    if args.output is None:
        args.output = os.path.join(output_dir, 'trajectory_2d_timeaxis.png')
    
    print(f"Loading trajectory file: {args.traj_path}")
    traj = load_trajectory(args.traj_path)
    
    # Load GT trajectory if gt_traj.pt exists in same directory
    gt_traj = None
    gt_traj_path = os.path.join(output_dir, 'gt_traj.pt')
    
    if os.path.exists(gt_traj_path):
        print(f"\nLoading GT trajectory from: {gt_traj_path}")
        gt_traj = torch.load(gt_traj_path)
        print(f"GT trajectory shape: {gt_traj.shape}")
    else:
        print(f"\nNote: gt_traj.pt not found, showing prediction only")
        print(f"Run test_free.py to generate gt_traj.pt for comparison")
    
    # Generate visualization
    visualize_trajectory_2d_with_time(traj, args.output, args.arrow_interval, gt_traj)

if __name__ == "__main__":
    main()
