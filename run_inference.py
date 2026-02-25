import sys

sys.path.append(".")

from pathlib import Path
import numpy as np
from PIL import Image

from inferer import OnlineJNGInferer
from modules.preprocessor import iter_online_processed_frames


FFMPEG_PATH = "binaries/ffmpeg.exe"
DETECTION_MODEL_PATH = "binaries/jng.det.onnx"
KEYPOINT_MODEL_PATH = "binaries/jng.pose.onnx"


def infer_online(
    model_folder,
    video_file,
    output_folder,
    model_type="simultaneous",
    device="auto",
    save_processed=False,
):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_processed:
        (output_path / "processed" / "images").mkdir(parents=True, exist_ok=True)
        (output_path / "processed" / "keypoints").mkdir(parents=True, exist_ok=True)

    inferer = OnlineJNGInferer(
        model_folder=model_folder,
        model_type=model_type,
        device=device,
    )

    for frame_id, _, crop_img, keypoints in iter_online_processed_frames(
        ffmpeg_path=FFMPEG_PATH,
        detection_model_path=DETECTION_MODEL_PATH,
        keypoint_model_path=KEYPOINT_MODEL_PATH,
        video_path=Path(video_file),
    ):
        inferer.step(crop_img, keypoints)

        if save_processed:
            prefix = f"{frame_id:06d}"
            Image.fromarray(crop_img).save(
                output_path / "processed" / "images" / f"{prefix}.jpg"
            )
            np.save(output_path / "processed" / "keypoints" / f"{prefix}.npy", keypoints)

    counts = inferer.finalize()
    with open(output_path / "counts.txt", "w") as f:
        f.write("\n".join([str(c) for c in counts]))

    print(f"Counts saved to {output_path / 'counts.txt'}")
    return counts


def infer(model_folder, vid_file, output_folder, model_type="simultaneous"):
    return infer_online(
        model_folder=model_folder,
        video_file=vid_file,
        output_folder=output_folder,
        model_type=model_type,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True, help="Path to model dir")
    parser.add_argument(
        "--model_type",
        type=str,
        default="alternating",
        choices=["simultaneous", "alternating"],
    )
    parser.add_argument("--video_file", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output dir")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="ONNX runtime provider preference",
    )
    parser.add_argument(
        "--save_processed",
        action="store_true",
        help="Persist cropped images/keypoints for debugging",
    )
    parser.add_argument(
        "--future_context_policy",
        type=str,
        default=None,
        help="Deprecated: kept for CLI compatibility. Ignored.",
    )
    args = parser.parse_args()
    infer_online(
        model_folder=args.model_folder,
        video_file=args.video_file,
        output_folder=args.output_folder,
        model_type=args.model_type,
        device=args.device,
        save_processed=args.save_processed,
    )
