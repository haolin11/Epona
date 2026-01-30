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

def visualize_trajectory_2d_with_time(traj, save_path, arrow_interval=20):
    """
    2D visualization with clearer time representation using color segments
    """
    print("Generating 2D trajectory visualization with time segments...")
    
    # Extract trajectory data
    next_poses = traj[:, 0, :2].numpy()
    next_yaws = traj[:, 0, 2].numpy()
    
    # Calculate absolute positions
    positions = np.zeros((len(traj) + 1, 2))
    absolute_yaw = 0.0
    yaw_history = [0.0]
    
    for i in range(len(traj)):
        cos_yaw = np.cos(np.radians(absolute_yaw))
        sin_yaw = np.sin(np.radians(absolute_yaw))
        rotation = np.array([[cos_yaw, -sin_yaw],
                           [sin_yaw, cos_yaw]])
        global_displacement = rotation @ next_poses[i]
        positions[i+1] = positions[i] + global_displacement
        absolute_yaw += next_yaws[i]
        yaw_history.append(absolute_yaw)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Main trajectory plot
    ax1 = plt.subplot(1, 2, 1)
    
    # Draw trajectory with time-based color segments
    n_segments = 10  # Divide into 10 time segments
    segment_size = len(positions) // n_segments
    colors = plt.cm.rainbow(np.linspace(0, 1, n_segments))
    
    for seg in range(n_segments):
        start_idx = seg * segment_size
        end_idx = min((seg + 1) * segment_size, len(positions))
        
        # Draw this segment
        ax1.plot(positions[start_idx:end_idx, 0], 
                positions[start_idx:end_idx, 1],
                color=colors[seg], linewidth=3, alpha=0.8,
                label=f'Frames {start_idx}-{end_idx-1}')
    
    # Draw orientation arrows at regular intervals
    arrow_indices = list(range(0, len(positions)-1, arrow_interval))
    if (len(positions)-2) not in arrow_indices:  # Ensure last position has arrow
        arrow_indices.append(len(positions)-2)
    
    print(f"Drawing {len(arrow_indices)} arrows at positions: {arrow_indices[:10]}... (showing first 10)")
    
    for i in arrow_indices:
        x, y = positions[i]
        yaw = yaw_history[i]
        
        # Calculate arrow direction
        arrow_length = 4.0
        dx = arrow_length * np.cos(np.radians(yaw))
        dy = arrow_length * np.sin(np.radians(yaw))
        
        # Draw arrow
        color_idx = int(i / len(positions) * n_segments)
        color_idx = min(color_idx, n_segments - 1)
        
        ax1.arrow(x, y, dx, dy,
                 head_width=2.0, head_length=1.5,
                 fc=colors[color_idx], ec='black',
                 linewidth=1.5, alpha=0.9, zorder=5)
        
        # Add frame number
        ax1.text(x - 2, y - 3, str(i),
                fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor='black', 
                         alpha=0.8))
    
    # Mark start and end
    ax1.scatter(positions[0, 0], positions[0, 1],
               c='green', s=400, marker='o',
               label='Start', zorder=10,
               edgecolors='black', linewidth=3)
    
    ax1.scatter(positions[-1, 0], positions[-1, 1],
               c='red', s=400, marker='X',
               label='End', zorder=10,
               edgecolors='black', linewidth=3)
    
    # Statistics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    total_rotation = yaw_history[-1]
    
    stats_text = (f'Total Distance: {total_distance:.2f} m\n'
                 f'Total Rotation: {total_rotation:.2f} deg\n'
                 f'Avg Speed: {total_distance/len(traj):.3f} m/frame\n'
                 f'Arrows: {len(arrow_indices)} (interval: {arrow_interval})')
    
    ax1.text(0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax1.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Vehicle Trajectory (Color = Time Progression)\nTotal Frames: {len(traj)}',
                 fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    
    # Time evolution plot (X, Y, Yaw vs Time)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 4)
    
    # Plot X and Y positions over time
    time_steps = np.arange(len(positions))
    ax2.plot(time_steps, positions[:, 0], 'b-', linewidth=2, label='X Position')
    ax2.plot(time_steps, positions[:, 1], 'r-', linewidth=2, label='Y Position')
    ax2.set_xlabel('Time Step (Frame)', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title('X & Y Positions over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot Yaw angle over time
    ax3.plot(time_steps, yaw_history, 'g-', linewidth=2)
    ax3.set_xlabel('Time Step (Frame)', fontsize=12)
    ax3.set_ylabel('Yaw Angle (degrees)', fontsize=12)
    ax3.set_title('Vehicle Orientation over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.close()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total frames: {len(traj)}")
    print(f"  Total distance: {total_distance:.2f} m")
    print(f"  Total rotation: {total_rotation:.2f} deg")
    print(f"  Average speed: {total_distance/len(traj):.3f} m/frame")
    print(f"  Arrows drawn: {len(arrow_indices)}")


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
    
    # Generate visualization
    visualize_trajectory_2d_with_time(traj, args.output, args.arrow_interval)

if __name__ == "__main__":
    main()
