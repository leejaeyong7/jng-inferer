import sys

sys.path.append(".")
from pathlib import Path
from modules.preprocessor import process_video
from modules.preprocessor.preprocess.utils.video_ops import (
    extract_video,
    create_video,
    enumerate_vid,
    get_vid_count,
)
from PIL import Image, ImageDraw, ImageFont
from inferer import compute_scores

ffmpeg_path = "binaries/ffmpeg.exe"
detection_model_path = "binaries/jng.det.onnx"
keypoint_model_path = "binaries/jng.pose.onnx"


def infer(model_folder, vid_file, output_folder, model_type="simultaneous"):
    video_path = Path(vid_file)
    output_path = Path(output_folder)
    processed_path = output_path / "processed"

    process_video(
        ffmpeg_path,
        detection_model_path,
        keypoint_model_path,
        video_path,
        processed_path,
    )
    counts = compute_scores(model_folder, processed_path, model_type=model_type)
    count_file = processed_path / "counts.txt"
    with open(count_file, "w") as f:
        f.write("\n".join([str(c) for c in counts]))
    print(f"Counts saved to {count_file}")

    (output_path / "raw_frames").mkdir(parents=True, exist_ok=True)
    extract_video(ffmpeg_path, video_path, output_path / "raw_frames")
    num_frames = get_vid_count(output_path / "raw_frames")
    draw_font = ImageFont.truetype("binaries/superbasic.ttf", 128)
    count_img_path = output_path / "counts_frames"
    count_img_path.mkdir(parents=True, exist_ok=True)
    for iid, np_img in enumerate_vid(output_path / "raw_frames"):
        img = Image.fromarray(np_img).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Count: {counts[iid]}", font=draw_font, fill=(0, 255, 0))
        img.save(count_img_path / f"{iid:06d}.jpg")

    (output_path / "output.mp4").unlink(missing_ok=True)
    create_video(ffmpeg_path, output_path / "counts_frames", output_path / "output.mp4")
    with open(output_path / "counts.txt", "w") as f:
        f.write("\n".join([str(c) for c in counts]))

    print(f"Output video saved to {output_path / 'output.mp4'}")

    return


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
    infer(
        args.model_folder,
        args.video_file,
        args.output_folder,
        model_type=args.model_type,
    )
