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
    pipeline_mode,
    disable_prenms_detector,
    require_prenms_detector,
    det_keyframe_interval,
    mp_queue_size,
    mp_pose_batch_size,
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
            use_prenms_detector=not disable_prenms_detector,
            require_prenms_detector=require_prenms_detector,
            det_keyframe_interval=det_keyframe_interval,
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
        pipeline_mode=pipeline_mode,
        use_prenms_detector=not disable_prenms_detector,
        require_prenms_detector=require_prenms_detector,
        det_keyframe_interval=det_keyframe_interval,
        mp_queue_size=mp_queue_size,
        mp_pose_batch_size=mp_pose_batch_size,
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
        choices=["auto", "cpu", "nnapi", "cuda"],
        help="ONNX Runtime provider preference. 'nnapi' falls back to CPU; 'cuda' uses CUDA if available.",
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
        "--pipeline_mode",
        type=str,
        default="sequential",
        choices=["sequential", "multiprocess"],
        help="Inference pipeline mode when --with_video is not set.",
    )
    parser.add_argument(
        "--disable_prenms_detector",
        action="store_true",
        help="Disable pre-NMS detector path and use model's built-in postprocess outputs.",
    )
    parser.add_argument(
        "--require_prenms_detector",
        action="store_true",
        help="Fail if pre-NMS detector model cannot be prepared.",
    )
    parser.add_argument(
        "--det_keyframe_interval",
        type=int,
        default=3,
        help="Run person detection every N frames and reuse the last bbox between keyframes.",
    )
    parser.add_argument(
        "--mp_queue_size",
        type=int,
        default=8,
        help="Queue max size per stage when --pipeline_mode=multiprocess.",
    )
    parser.add_argument(
        "--mp_pose_batch_size",
        type=int,
        default=1,
        help="Crop batch size inside pose+score stage when --pipeline_mode=multiprocess.",
    )
    parser.add_argument(
        "--dll_path",
        type=str,
        default=None,
        help="Optional directory containing native runtime/delegate DLLs to preload before inference.",
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
        pipeline_mode=args.pipeline_mode,
        disable_prenms_detector=args.disable_prenms_detector,
        require_prenms_detector=args.require_prenms_detector,
        det_keyframe_interval=args.det_keyframe_interval,
        mp_queue_size=args.mp_queue_size,
        mp_pose_batch_size=args.mp_pose_batch_size,
    )
