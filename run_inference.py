import sys

sys.path.append(".")
from pathlib import Path
from modules.preprocessor import process_video
from modules.preprocessor.preprocess.utils.video_ops import (
    extract_video,
    create_video,
    enumerate_vid,
)
from PIL import Image, ImageDraw, ImageFont
from inferer import compute_scores


def infer(
    ffmpeg_path,
    detection_model_path,
    keypoint_model_path,
    model_folder,
    font_file,
    vid_file,
    tmep_folder,
    output_video_file,
    output_count_file,
    model_type="simultaneous",
):
    temp_path = Path(tmep_folder)
    video_path = Path(vid_file)
    output_video_file = Path(output_video_file)
    processed_path = temp_path / "processed"

    process_video(
        ffmpeg_path,
        detection_model_path,
        keypoint_model_path,
        video_path,
        processed_path,
    )
    counts = compute_scores(model_folder, processed_path, model_type=model_type)

    (temp_path / "raw_frames").mkdir(parents=True, exist_ok=True)

    extract_video(ffmpeg_path, video_path, temp_path / "raw_frames")
    count_img_path = temp_path / "counts_frames"
    count_img_path.mkdir(parents=True, exist_ok=True)

    draw_font = ImageFont.truetype(font_file, 128)
    for iid, np_img in enumerate_vid(temp_path / "raw_frames"):
        img = Image.fromarray(np_img).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Count: {counts[iid]}", font=draw_font, fill=(0, 255, 0))
        img.save(count_img_path / f"{iid:06d}.jpg")

    (output_video_file).unlink(missing_ok=True)

    create_video(ffmpeg_path, temp_path / "counts_frames", output_video_file)
    final_count = counts[-1] if len(counts) > 0 else 0
    with open(output_count_file, "w") as f:
        f.write(str(final_count))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument(
        "--model_type", type=str, default="alternating", help="Type of the model"
    )
    parser.add_argument("--video_file", type=str, help="Path to the video file")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    args = parser.parse_args()
    infer(args.model_folder, args.video_file, args.output_folder)
