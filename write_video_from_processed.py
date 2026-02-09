import sys

sys.path.append(".")
from pathlib import Path
from modules.preprocessor.preprocess.utils.video_ops import (
    extract_video,
    create_video,
    enumerate_vid,
    get_vid_count,
)
from PIL import Image, ImageDraw, ImageFont


def infer(proc_folder):
    process_video(
        ffmpeg_path,
        detection_model_path,
        keypoint_model_path,
        video_path,
        processed_path,
    )
    counts = compute_scores(model_folder, processed_path)
    count_file = processed_path / "counts.txt"
    with open(count_file, "w") as f:
        f.write("\n".join([str(c) for c in counts]))
    print(f"Counts saved to {count_file}")

    extract_video(ffmpeg_path, video_path, output_path / "raw_frames")
    num_frames = get_vid_count(output_path / "raw_frames")
    draw_font = ImageFont.truetype("binaries/superbasic.ttf", 32)
    for i in range(num_frames):
        frame_path = output_path / "raw_frames" / f"frame_{i:05d}.png"
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Count: {counts[i]}", font=draw_font, fill=(255, 0, 0))
        img.save(frame_path)

    create_video(ffmpeg_path, output_path / "raw_frames", output_path / "output.mp4")

    return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--video_file", type=str, help="Path to the video file")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    args = parser.parse_args()
    infer(args.model_folder, args.video_file, args.output_folder)
