import sys

sys.path.append(".")

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from inferer import OnlineJNGInferer
from modules.preprocessor import iter_online_processed_frames
from modules.preprocessor.preprocess.utils.video_ops import create_video


FFMPEG_PATH = "binaries/ffmpeg.exe"
DETECTION_MODEL_PATH = "binaries/jng.det.onnx"
KEYPOINT_MODEL_PATH = "binaries/jng.pose.onnx"
FONT_PATH = "binaries/superbasic.ttf"


def _load_font(size=96):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def infer_and_write(
    model_folder,
    video_file,
    output_folder,
    model_type="simultaneous",
    device="auto",
):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    count_frame_path = output_path / "counts_frames"
    count_frame_path.mkdir(parents=True, exist_ok=True)
    raw_frame_path = output_path / "raw_frames"
    raw_frame_path.mkdir(parents=True, exist_ok=True)

    inferer = OnlineJNGInferer(
        model_folder=model_folder,
        model_type=model_type,
        device=device,
    )
    draw_font = _load_font()

    for frame_id, raw_frame, crop_img, keypoints in iter_online_processed_frames(
        ffmpeg_path=FFMPEG_PATH,
        detection_model_path=DETECTION_MODEL_PATH,
        keypoint_model_path=KEYPOINT_MODEL_PATH,
        video_path=Path(video_file),
    ):
        inferer.step(crop_img, keypoints)
        Image.fromarray(raw_frame).convert("RGB").save(raw_frame_path / f"{frame_id:06d}.jpg")

    counts = inferer.finalize()

    for frame_id, count in enumerate(counts):
        img = Image.open(raw_frame_path / f"{frame_id:06d}.jpg").convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Count: {count}", font=draw_font, fill=(0, 255, 0))
        img.save(count_frame_path / f"{frame_id:06d}.jpg")

    output_video = output_path / "output.mp4"
    output_video.unlink(missing_ok=True)
    create_video(FFMPEG_PATH, count_frame_path, output_video)

    with open(output_path / "counts.txt", "w") as f:
        f.write("\n".join([str(c) for c in counts]))

    print(f"Counts saved to {output_path / 'counts.txt'}")
    print(f"Output video saved to {output_video}")
    return counts


def infer(model_folder, vid_file, output_folder, model_type="simultaneous"):
    return infer_and_write(
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
        "--future_context_policy",
        type=str,
        default=None,
        help="Deprecated: kept for CLI compatibility. Ignored.",
    )
    args = parser.parse_args()
    infer_and_write(
        model_folder=args.model_folder,
        video_file=args.video_file,
        output_folder=args.output_folder,
        model_type=args.model_type,
        device=args.device,
    )
