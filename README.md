# ShakeIt

Descirption

## Overview

Overview

## Prerequisites
- Linux (We tested our codes on Ubuntu 18.04.6)
- Anaconda
- Python 3.11
- Pytorch 2.0.1

To get started, first please clone the repo

```
git clone https://github.com/vv4alekseev/ShakeIt
```

Then, please run the following commands:

```
conda create -n shakeit python=3.11
conda activate shakeit
pip install -r requirements.txt
```

## Quick start


1. Download the pre-trained models and the data. [Link](https://www.google.com)
2. Put the downloaded zip files to the root directory of this project
3. Run the shaking effect demo

```
python3 apply_camera_shake.py --input_video demo/video.mp4  --output_video results/demo_shaked.mp4
```

If everythings works, you will find a `demo_shaked.mp4` file in `results`. And the video should be like:

## License

This work is licensed under MIT license. See the LICENSE for details.

## Contact

If you have any questions, please contact us via

- [vv4alekseev@gmail.com](mailto:vv4alekseev@gmail.com)

## Acknowledgement

Some parts of this repo are based on [DeepFillv2-pytorch](https://github.com/nipponjo/deepfillv2-pytorch).