import sys

sys.path.append(".")

from run_inference import infer_online
from infer_and_write import infer_and_write


def infer(
    model_folder,
    vid_file,
    model_type,
    output_folder,
    with_video,
    device,
    show_progress,
    visualize_tracking,
    decode_mode,
    realtime,
    profile_stages,
    dll_path,
):
    if with_video:
        return infer_and_write(
            model_folder=model_folder,
            video_file=vid_file,
            output_folder=output_folder,
            model_type=model_type,
            device=device,
            show_progress=show_progress,
            visualize_tracking=visualize_tracking,
            decode_mode=decode_mode,
            realtime=realtime,
            profile_stages=profile_stages,
            dll_path=dll_path,
        )
    return infer_online(
        model_folder=model_folder,
        video_file=vid_file,
        output_folder=output_folder,
        model_type=model_type,
        device=device,
        show_progress=show_progress,
        decode_mode=decode_mode,
        realtime=realtime,
        profile_stages=profile_stages,
        dll_path=dll_path,
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
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="ONNX runtime provider preference",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show preprocessing/inference progress",
    )
    parser.add_argument(
        "--visualize_tracking",
        action="store_true",
        help="When --with_video is set, also write tracking frames",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="stream",
        choices=["stream", "extract"],
        help="Video decode mode: stream frames from ffmpeg pipe or extract temporary jpgs.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Force low-latency mode (stream decode, no processed dumps, stream output write).",
    )
    parser.add_argument(
        "--profile_stages",
        action="store_true",
        help="Print decode/det/pose/feature/score(/write) stage timing summary.",
    )
    parser.add_argument(
        "--dll_path",
        type=str,
        default=None,
        help="Optional directory containing CUDA/ORT runtime DLLs to preload before session creation.",
    )
    args = parser.parse_args()
    infer(
        model_folder=args.model_folder,
        vid_file=args.video_file,
        model_type=args.model_type,
        output_folder=args.output_folder,
        with_video=args.with_video,
        device=args.device,
        show_progress=args.show_progress,
        visualize_tracking=args.visualize_tracking,
        decode_mode=args.decode_mode,
        realtime=args.realtime,
        profile_stages=args.profile_stages,
        dll_path=args.dll_path,
    )
