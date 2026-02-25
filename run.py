import sys

sys.path.append(".")

from run_inference import infer_online
from infer_and_write import infer_and_write


def infer(model_folder, vid_file, model_type, output_folder, with_video, device):
    if with_video:
        return infer_and_write(
            model_folder=model_folder,
            video_file=vid_file,
            output_folder=output_folder,
            model_type=model_type,
            device=device,
        )
    return infer_online(
        model_folder=model_folder,
        video_file=vid_file,
        output_folder=output_folder,
        model_type=model_type,
        device=device,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True, help="Path to model dir")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["simultaneous", "alternating"],
        default="simultaneous",
    )
    parser.add_argument("--video_file", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output dir")
    parser.add_argument(
        "--with_video",
        action="store_true",
        help="Generate output video with overlaid counts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="ONNX runtime provider preference",
    )
    args = parser.parse_args()
    infer(
        model_folder=args.model_folder,
        vid_file=args.video_file,
        model_type=args.model_type,
        output_folder=args.output_folder,
        with_video=args.with_video,
        device=args.device,
    )
