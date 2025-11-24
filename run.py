import sys

sys.path.append(".")
from pathlib import Path
from modules.preprocessor import process_video
from PIL import Image, ImageDraw, ImageFont
from inferer import compute_scores

ffmpeg_path = "binaries/ffmpeg.exe"
detection_model_path = "binaries/jng.det.onnx"
keypoint_model_path = "binaries/jng.pose.onnx"


def infer(model_folder, vid_file, output_folder):
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
    counts = compute_scores(model_folder, processed_path)
    count_file = processed_path / "counts.txt"
    with open(count_file, "w") as f:
        f.write("\n".join([str(c) for c in counts]))
    print(f"Counts saved to {count_file}")
    return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_folder", type=str, help="Path to the model folder")
    parser.add_argument("--video_file", type=str, help="Path to the video file")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    args = parser.parse_args()
    infer(args.model_folder, args.video_file, args.output_folder)
