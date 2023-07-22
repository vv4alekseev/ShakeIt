"""
This Python script contains a set of functions that facilitate video processing tasks.
It provides a collection of useful functions to handle video files, perform various 
operations, and extract valuable information from videos.
"""

import cv2 as cv
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from model import Generator

import logging
from utils import green_message

#------------------------------------------------------------------------------------

def read_video(path: str) -> tuple[np.ndarray, float]:
    """
    Read a video file from the given path and return its frames and frames per second 
    (fps).

    Parameters:
        path (str): The file path of the video.

    Returns:
        tuple: A tuple containing the following items:
            - video (np.ndarray): A NumPy array containing the video frames with the 
                                  shape [F, H, W, C].
            - fps (float): Frames per second of the video.
    """

    logging.info("Reading video...")
    
    cpr = cv.VideoCapture(path)
    fps = cpr.get(cv.CAP_PROP_FPS)
    has_frame = True
    frames = []

    while has_frame:
        has_frame, frame = cpr.read()
        if has_frame:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame)
            
    cpr.release()
    numpy_video = np.array(frames)

    logging.info(green_message("Video read successfully!"))
    
    return (numpy_video, fps)

#------------------------------------------------------------------------------------

def init_model(device: str) -> nn.Module:
    """
    Initialize a neural network model and configure it for the specified device.

    Parameters:
        device (str): A string specifying the device where the model will be deployed.
                      Choose between "cpu" for CPU or "cuda" for GPU acceleration.

    Returns:
        nn.Module: An instance of PyTorch's `nn.Module` representing the initialized 
                   neural network model.
    """

    
    message = "Inpainting model initialization..."
    logging.info(message)
    
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator_state_dict = torch.load("weights/states_pt_places2.pth")['G']
    generator.load_state_dict(generator_state_dict, strict=True)

    message = green_message("Inpainting model initialized!")
    logging.info(message)

    return generator

#------------------------------------------------------------------------------------

def inpaint_n_frames(video: np.ndarray, 
                     each_x_frame: int, 
                     padding: int,
                     generator: nn.Module,
                     device: str) -> tuple[list[np.ndarray], list[int]]:
    """
    Inpaint each N frames in a video using a given neural network generator model.

    Parameters:
        video (np.ndarray): The input video represented as a NumPy array with shape 
                            (F, H, W, C).
        each_x_frame (int): The number of frames to skip between inpainting.
        padding (int): Padding value for for each side of frames.
        generator (nn.Module): A PyTorch neural network module representing the 
                               inpainting generator model. This model takes an 
                               incomplete frame as input and generates the inpainted 
                               frame.
        device (str): A string specifying the device where the generator model will 
                      be deployed.

    Returns:
        tuple[list[np.ndarray], list[int]]: A tuple containing two lists.
            - The first list contains the inpainted frames, represented as NumPy 
              arrays with shape (F, H, W, C).
            - The second list contains the indices of the inpainted frames within the
              original video.
    """
    
    grid = 8
    inpainted_frames = []
    f = video.shape[0]
    idxs = (np.linspace(0, f-1, f//each_x_frame)).astype(int).tolist()

    message = f"Inpainting each {each_x_frame} frame..."
    logging.info(message)

    for idx in idxs:
        
        frame = video[idx]
    
        h_new = frame.shape[0] + padding * 2
        w_new = frame.shape[1] + padding * 2
        
        mask = np.zeros((h_new, w_new), dtype=bool)
        mask[:padding, :] = True
        mask[-padding:, :] = True
        mask[:, :padding] = True
        mask[:, -padding:] = True
    
        mask = T.ToTensor()(mask)
        frame = T.ToTensor()(frame)
        frame = T.Pad(padding=padding, fill=0, padding_mode='constant')(frame)
    
        _, h, w = frame.shape
        frame = frame[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    
        frame = (frame*2 - 1.).to(device)
        mask = (mask > 0.5).to(dtype=torch.float32, device=device)
        frame_masked = frame * (1.-mask)
    
        ones_x = torch.ones_like(frame_masked)[:, 0:1, :, :]
        x = torch.cat([frame_masked, ones_x, ones_x*mask], dim=1)
    
        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)
    
        frame_inpainted = frame * (1.-mask) + x_stage2 * mask
        frame_out = ((frame_inpainted[0].permute(1, 2, 0) + 1)*127.5)
        frame_out = frame_out.to(device='cpu', dtype=torch.uint8)
        frame_out = frame_out.numpy()
        inpainted_frames.append(frame_out)

    message = green_message(f"{len(inpainted_frames)} frames inpainted!")
    logging.info(message)

    return (inpainted_frames, idxs)

#------------------------------------------------------------------------------------

def interpolate_frames(video: np.ndarray, 
                       inpainted_frames: list[np.ndarray],
                       padding: int,
                       idxs: list[int]) -> np.ndarray:
    """
    Interpolate missing frames in a video using inpainted frames.

    Parameters:
        video (np.ndarray): The input video represented as a NumPy array with shape 
                            (F, H, W, C).
        inpainted_frames (list[np.ndarray]): A list of inpainted frames, represented 
                                             as NumPy arrays with shape (H, W, C).
        padding (int): Padding value for for each side of frames.
        idxs (list[int]): A list of indices corresponding to the positions of the 
                          inpainted frames within the original video.

    Returns:
        np.ndarray: The video with the missing frames interpolated, represented as a 
                    NumPy array with shape (F, H, W, C).
    """

    message = f"Interpolating other frames..."
    logging.info(message)
    
    interpolated_frames = []
    n_intervals = len(idxs) - 1
    
    for i in range(n_intervals):
        start = idxs[i]
        end = idxs[i+1]
        
        frame1 = inpainted_frames[i]
        frame2 = inpainted_frames[i+1]
    
        j = 0
        
        for k in range(start, end):
            alpha = j / (end - start + 1)
            j += 1
            
            interpolated_frame = (1 - alpha) * frame1 + alpha * frame2
            interpolated_frame = interpolated_frame.astype(np.uint8)
            
            original_frame_pad = pad_image(image=video[k], padding=padding)
            original_frame_pad[:padding, :, :] = interpolated_frame[:padding, :, :]
            original_frame_pad[-padding:, :, :] = interpolated_frame[-padding:, :, :]
            original_frame_pad[:, :padding, :] = interpolated_frame[:, :padding, :]
            original_frame_pad[:, -padding:, :] = interpolated_frame[:, -padding:, :]
            
            interpolated_frames.append(original_frame_pad)
    
    interpolated_video = np.stack(interpolated_frames)

    message = green_message("All frames interpolated!")
    logging.info(message)

    return interpolated_video

#------------------------------------------------------------------------------------

def to_circular(video: np.ndarray) -> np.ndarray:
    """
    Apply a circular mask to each frame of the video.

    Parameters:
        video (np.ndarray): The input video as a NumPy array with shape (F, H, W, C).

    Returns:
        np.ndarray: The video with circular masks applied to each frame. The output 
                    has the same shape as the input video.
    """

    logging.info("Applying the circular mask to the video...")
    
    f, h, w, c = video.shape
    
    center_coordinates = (h // 2, w // 2)
    radius = min(center_coordinates)
    color = (0, 0, 0)
    thickness = -1

    mask = np.ones((h, w, c)) * 255
    mask = cv.circle(mask, center_coordinates, radius, color, thickness)

    for i in range(f):
        masked = mask + video[i]
        masked = np.clip(masked, 0, 255).astype(np.uint8)
        video[i] = masked

    logging.info(green_message("Mask applied!"))

    return video

#------------------------------------------------------------------------------------

def generate_translation(
    n_frames: int, 
    fps: float, 
    amplitudes: tuple[float, float],
    frequencies: tuple[float, float]
) -> list:
    """
    Generate a list of translation values to create a hand shaking effect along one 
    axis in an 80-second video.

    Parameters:
        n_frames (int): The total number of frames in the video.
        fps (float): Frames per second of the video.
        amplitudes (tuple[float, float]): The amplitudes of the sinusoidal waves 
                                          along y-axis.
        frequencies (tuple[float, float]): The frequencies of the sinusoidal waves 
                                           along the x-axis.

    Returns:
        list: A list of translation values for each frame to create the hand shaking
              effect.
    """
    logging.info("Generating the translations...")
    
    num_points = 2000
    num_waves = np.random.randint(30, 40)
    amplitude_min, amplitude_max = amplitudes
    frequence_min, frequence_max = frequencies
    
    x = np.linspace(0, 10, num_points)
    y = np.zeros_like(x)
    
    for _ in range(num_waves):
        frequency = np.random.uniform(frequence_min, frequence_max)
        amplitude = np.random.uniform(amplitude_min, amplitude_max)
        phase_shift = np.random.uniform(0, 2*np.pi)
        y += amplitude * np.sin(2*np.pi*frequency*x + phase_shift)

    duration = n_frames / fps
    fixed = int(num_points * duration / 80)

    x_fixed, y_fixed = x[:fixed], y[:fixed]

    x_interpolated = np.linspace(0, max(x_fixed), n_frames)
    y_interpolated = np.interp(x_interpolated, x_fixed, y_fixed)

    logging.info(green_message("Translations generated!"))
    
    return y_interpolated

#------------------------------------------------------------------------------------

def shake(video: np.ndarray, x_translation: list, y_translation: list) -> np.ndarray:
    """
    Apply a hand shaking effect to a video by translating frames along the x and y 
    axes.

    Parameters:
        video (np.ndarray): The input video as a NumPy array with shape (F, H, W, C).
        x_translation (list): A list of x-axis translation values for each frame.
        y_translation (list): A list of y-axis translation values for each frame.

    Returns:
        shaken_video (np.ndarray): The video with the hand shaking effect applied. 
                                   It has the same shape as the input video.
    """

    logging.info("Applying shaking effect...")
    
    video_out = np.zeros_like(video)
    f, h, w, c = video.shape

    for i in range(f):
        frame = video[i]
        shift_y = y_translation[i]
        shift_x = x_translation[i]

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv.warpAffine(frame, M, (w, h))
        
        video_out[i] = frame

    logging.info(green_message("Shaking effect applied!"))

    return video_out

#------------------------------------------------------------------------------------

def crop_video(video: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Crop the frames of a video to the specified size.

    Parameters:
        video (np.ndarray): The input video as a NumPy array with shape (F, H, W, C).
        size (tuple[int, int]): The target size for cropping, specified as 
                                (H_new, W_new).

    Returns:
        cropped_video (np.ndarray): The cropped video as a NumPy array with shape 
                                    (F, H_new, W_new, C).
    """

    logging.info("Cropping the video...")
    
    f, h, w, c = video.shape
    h_new, w_new = size

    cropped_video = np.zeros((f, h_new, w_new, c), dtype=np.uint8)

    y_start = int((h - h_new) / 2)
    x_start = int((w - w_new) / 2)

    for i in range(f):
        frame = video[i]
        cropped_video[i] = frame[y_start:y_start+h_new, x_start:x_start+w_new]

    logging.info(green_message("Video cropped!"))

    return cropped_video

#------------------------------------------------------------------------------------

def save_video(path: str, video: np.ndarray, fps: float) -> None:
    """
    Save a video to the specified path.

    Parameters:
        path (str): The file path where the video will be saved.
        video (np.ndarray): The video to be saved as a NumPy array with shape 
                            (F, H, W, C).
        fps (float): Frames per second of the video.

    Returns:
        None
    """

    logging.info("Saving the video...")
    
    f, h, w, c = video.shape

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(path, fourcc, fps, (h, w))

    for i in range(f):
        frame = video[i]
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        out.write(frame)

    logging.info(green_message("Video saved!"))

    out.release()

#------------------------------------------------------------------------------------

def pad_image(image: np.ndarray, padding: int, fill_value: int=0) -> np.ndarray:
    """
    Pad an image with a specified value on each side of the frame.

    Parameters:
        image (np.ndarray): The input image represented as a NumPy array.
        padding (int): The number of pixels to add as padding around the image.
        fill_value (int or tuple, optional): The value to fill in the padded regions.

    Returns:
        np.ndarray: The padded image represented as a NumPy array with shape 
                    (H + 2*padding, W + 2*padding, C).
    """
    h, w = image.shape[:2]

    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding), (0, 0)),
        mode='constant',
        constant_values=fill_value
    )

    return padded_image

#------------------------------------------------------------------------------------
