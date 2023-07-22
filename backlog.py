import math
import numpy as np
from scipy.signal import convolve2d

#-------------------------------------------------------------------------------------

def calculate_direction(x1, y1, x2, y2):
    # Calculate the change in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the angle in radians between the positive x-axis and the line connecting the two points
    # The result is in the range [-pi, pi]
    angle_radians = math.atan2(dy, dx)
    
    # Convert the angle from radians to degrees and make sure it's in the range [0, 360)
    angle_degrees = math.degrees(angle_radians) % 360
    
    return angle_degrees

#-------------------------------------------------------------------------------------

def motion_blur(video, kernels, angles):
    """
    Apply motion blur to a video.

    Args:
        video (numpy.ndarray): Input video with shape [F, H, W, C], where F is the number of frames,
                               H is the height, W is the width, and C is the number of channels.
        kernel_size (int): Size of the motion blur kernel. Should be an odd integer.
        angle (float): Angle of the motion blur in degrees (0-360).

    Returns:
        numpy.ndarray: Blurred video with the same shape as the input video.
    """
    def generate_motion_blur_kernel(size, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        center = (size - 1) // 2
        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                if np.cos(angle_rad) * x + np.sin(angle_rad) * y >= 0:
                    kernel[i, j] = 1
        kernel /= kernel.sum()
        return kernel
    
    # Prepare the output video array
    output_video = np.zeros_like(video)

    # Apply motion blur to each frame of the video
    for frame_idx in trange(video.shape[0]):
        kernel_size = kernels[frame_idx]
        angle = angles[frame_idx]
        kernel = generate_motion_blur_kernel(kernel_size, angle)
        for channel_idx in range(video.shape[3]):
            output_video[frame_idx, :, :, channel_idx] = convolve2d(
                video[frame_idx, :, :, channel_idx], kernel, mode='same', boundary='symm'
            )

    return output_video