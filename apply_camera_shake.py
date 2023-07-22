import logging
import argparse
import torch

from utils import init_logging, read_config

from video_processing import (read_video,
                              init_model,
                              inpaint_n_frames,
                              interpolate_frames,
                              to_circular,
                              generate_translation,
                              shake,
                              crop_video,
                              save_video)


def apply_camera_shake(args) -> None:
    """
    Apply camera shake effects to a video sequence.

    Parameters:
        args (dict): A dictionary containing various parameters and settings.
    
    Returns:
        None
    """
    # Read config file
    init_logging()
    configs = read_config(path="configs.yaml")

    # Read video
    video, fps = read_video(args.input_video)
    f, h_orig, w_orig, _ = video.shape

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model initialization
    generator = init_model(device=device)

    # Inpaint some frames
    inpainted_frames, idxs = inpaint_n_frames(video=video, 
                                              each_x_frame=configs["each_x_frame"], 
                                              padding=32,
                                              generator=generator,
                                              device=device)

    # Interpolate missed frames
    video = interpolate_frames(video=video, 
                               inpainted_frames=inpainted_frames,
                               padding=configs["padding"],
                               idxs=idxs)

    # Generate a random waveform function along the y-axis
    y_trans = generate_translation(n_frames=f, 
                                   fps=fps, 
                                   amplitudes=configs["y_amplitudes"], 
                                   frequencies=configs["y_frequencies"])

    # Generate a random waveform function along the x-axis
    x_trans = generate_translation(n_frames=f, 
                                   fps=fps, 
                                   amplitudes=configs["x_amplitudes"], 
                                   frequencies=configs["x_frequencies"])

    # Apply the shake effect
    video = shake(video, x_trans, y_trans)

    # Crop the video to the initial size
    video = crop_video(video=video, size=(h_orig, w_orig))

    # Apply a mask to the video
    video = to_circular(video=video)

    # Save the video
    save_video(args.output_video, video, fps)


def main():
    parser = argparse.ArgumentParser(description="Apply camera shake effect to a video.")
    parser.add_argument("--input_video", required=True, help="Path of the input video file.")
    parser.add_argument("--output_video", required=True, help="Path of the output video file.")
    args = parser.parse_args()
    apply_camera_shake(args)


if __name__ == "__main__":
    main()
