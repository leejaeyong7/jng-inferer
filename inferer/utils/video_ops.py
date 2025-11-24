import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract_video(ffmpeg_bin, video_file_path, output_path):
    subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "panic",
            "-i",
            str(video_file_path),
            "-vf",
            "fps=24",
            f"{str(output_path)}/%06d.jpg",
        ]
    )


def create_video(ffmpeg_bin, frame_path, output_file_path):
    subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "panic",
            "-i",
            f"{str(frame_path)}/%06d.jpg",
            "-r",
            "24",
            str(output_file_path),
        ]
    )


def get_vid_count(vid_path):
    return len(
        [img_file for img_file in vid_path.iterdir() if img_file.name.endswith(".jpg")]
    )


def enumerate_vid(vid_path, desc="Processing", callback=None, total=None):
    for img_id, img_file in enumerate(sorted(vid_path.iterdir())):
        if img_file.suffix == ".jpg":
            with Image.open(str(img_file)) as img:
                img_data = np.array(img)
            if callback is not None:
                callback(desc, img_id, total)
            else:
                logger.debug(f'{desc} - {img_id}/{total if total is not None else "?"}')
            yield img_id, img_data
