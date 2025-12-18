import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def load_video(video_path):
    """
    Load video file into numpy array
    Args:
        video_path: path to video file
    Returns:
        frames: numpy array of shape (T, H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def save_video(frames, output_path, fps=30):
    """
    Save numpy array as video file
    Args:
        frames: numpy array of shape (T, H, W, C)
        output_path: path to save video
        fps: frames per second
    """
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frames.shape[2], frames.shape[1])
    )
    
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()

def visualize_flow(frame1, frame2):
    """
    Visualize optical flow between two frames
    Args:
        frame1: numpy array of first frame
        frame2: numpy array of second frame
    Returns:
        flow_img: visualization of optical flow
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Convert flow to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV visualization
    hsv = np.zeros_like(frame1)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow_img

def plot_physics_metrics(metrics_dict, save_path=None):
    """
    Plot physics metrics over time
    Args:
        metrics_dict: dictionary of metric names and values
        save_path: optional path to save plot
    """
    plt.figure(figsize=(12, 6))
    for name, values in metrics_dict.items():
        plt.plot(values, label=name)
    
    plt.xlabel('Frame')
    plt.ylabel('Metric Value')
    plt.title('Physics Metrics Over Time')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_visualization_grid(frames, flow_frames, metrics, output_path):
    """
    Create a grid visualization with original frames, flow visualization, and metrics
    Args:
        frames: original video frames
        flow_frames: optical flow visualization frames
        metrics: dictionary of physics metrics
        output_path: path to save visualization
    """
    fig = plt.figure(figsize=(15, 8))
    
    def update(frame_idx):
        plt.clf()
        
        # Original frame
        plt.subplot(131)
        plt.imshow(frames[frame_idx])
        plt.title('Original Frame')
        plt.axis('off')
        
        # Flow visualization
        plt.subplot(132)
        plt.imshow(flow_frames[frame_idx])
        plt.title('Optical Flow')
        plt.axis('off')
        
        # Metrics plot
        plt.subplot(133)
        for name, values in metrics.items():
            plt.plot(values[:frame_idx+1], label=name)
        plt.title('Physics Metrics')
        plt.legend()
        plt.grid(True)
    
    anim = FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=50
    )
    
    anim.save(output_path, writer='pillow')
    plt.close() 