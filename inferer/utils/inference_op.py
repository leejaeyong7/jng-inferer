import numpy as np
from PIL import Image
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor


def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = np.array(img).astype(np.float16) / 255.0  #
    img = img.transpose(2, 0, 1)  # 3 x H x W
    return img


def load_keypoint(kp_path):
    keypoints = np.load(kp_path).astype(np.float16)  # 17 x 2
    return keypoints


def multi_load_images(image_paths, num_workers=8):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_image, image_paths))
    return np.stack(images)


def multi_load_keypoints(kp_paths, num_workers=8):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        keypoints = list(executor.map(load_keypoint, kp_paths))
    return np.stack(keypoints)


def parallel_load_data(data_dir):
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    kp_dir = data_dir / "keypoints"
    frame_ids = sorted([int(p.stem) for p in img_dir.iterdir() if p.suffix == ".jpg"])

    # first load all images
    imgs = multi_load_images(
        [img_dir / f"{frame_id:06d}.jpg" for frame_id in frame_ids]
    )
    kps = multi_load_keypoints(
        [kp_dir / f"{frame_id:06d}.npy" for frame_id in frame_ids]
    )

    return imgs, kps
