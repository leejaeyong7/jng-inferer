import sys

sys.path.append(".")

from pathlib import Path
import os
import shutil
import numpy as np
from PIL import Image

from inferer import OnlineJNGInferer
from inferer.multiprocess_pipeline import infer_online_multiprocess
from inferer.utils.model_path_op import resolve_pose_preprocess_model_files
from inferer.utils.prenms_op import ensure_prenms_detector_model
from inferer.utils.dll_op import configure_dll_path
from inferer.utils.perf_op import StageProfiler, format_stage_report
from inferer.utils.progress_op import build_progress_callback
from modules.preprocessor import iter_online_processed_frames


SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR / "binaries"


def _resolve_ffmpeg_path():
    env_path = os.getenv("JNG_FFMPEG_PATH")
    if env_path:
        return env_path
    local_unix = BIN_DIR / "ffmpeg"
    if local_unix.exists():
        return str(local_unix)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    local_win = BIN_DIR / "ffmpeg.exe"
    if local_win.exists():
        return str(local_win)
    return "ffmpeg"


FFMPEG_PATH = _resolve_ffmpeg_path()


def _resolve_binary_model_paths(use_prenms_detector=True, require_prenms_detector=False):
    detection_model_path, keypoint_model_path = resolve_pose_preprocess_model_files(BIN_DIR)
    if use_prenms_detector:
        detection_model_path = ensure_prenms_detector_model(
            detection_model_path,
            required=require_prenms_detector,
        )
    return str(detection_model_path), str(keypoint_model_path)


def infer_online(
    model_folder,
    video_file,
    output_folder,
    model_type="simultaneous",
    device="auto",
    save_processed=False,
    show_progress=False,
    decode_mode="stream",
    realtime=False,
    profile_stages=False,
    dll_path=None,
    pipeline_mode="sequential",
    use_prenms_detector=True,
    require_prenms_detector=False,
    det_keyframe_interval=3,
    mp_queue_size=8,
    mp_pose_batch_size=1,
):
    configure_dll_path(dll_path)

    if realtime:
        save_processed = False
        decode_mode = "stream"

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_processed:
        (output_path / "processed" / "images").mkdir(parents=True, exist_ok=True)
        (output_path / "processed" / "keypoints").mkdir(parents=True, exist_ok=True)

    detection_model_path, keypoint_model_path = _resolve_binary_model_paths(
        use_prenms_detector=use_prenms_detector,
        require_prenms_detector=require_prenms_detector,
    )

    if pipeline_mode == "multiprocess":
        counts = infer_online_multiprocess(
            model_folder=model_folder,
            video_file=video_file,
            output_folder=output_folder,
            model_type=model_type,
            detection_model_path=detection_model_path,
            keypoint_model_path=keypoint_model_path,
            ffmpeg_path=FFMPEG_PATH,
            device=device,
            save_processed=save_processed,
            show_progress=show_progress,
            decode_mode=decode_mode,
            det_keyframe_interval=det_keyframe_interval,
            queue_size=mp_queue_size,
            pose_batch_size=mp_pose_batch_size,
            profile_stages=profile_stages,
        )
        print(f"Counts saved to {output_path / 'counts.txt'}")
        return counts

    stage_profiler = StageProfiler() if profile_stages else None
    inferer = OnlineJNGInferer(
        model_folder=model_folder,
        model_type=model_type,
        device=device,
        stage_profiler=stage_profiler,
    )
    callback_ptr, close_progress = build_progress_callback(show_progress)

    try:
        for frame_id, _, crop_img, keypoints in iter_online_processed_frames(
            ffmpeg_path=FFMPEG_PATH,
            detection_model_path=detection_model_path,
            keypoint_model_path=keypoint_model_path,
            video_path=Path(video_file),
            callback_ptr=callback_ptr,
            device=device,
            det_keyframe_interval=det_keyframe_interval,
            decode_mode=decode_mode,
            stage_profiler=stage_profiler,
        ):
            inferer.step(crop_img, keypoints)

            if save_processed:
                prefix = f"{frame_id:06d}"
                Image.fromarray(crop_img).save(
                    output_path / "processed" / "images" / f"{prefix}.jpg"
                )
                np.save(output_path / "processed" / "keypoints" / f"{prefix}.npy", keypoints)
    finally:
        close_progress()

    counts = inferer.finalize()
    with open(output_path / "counts.txt", "w") as f:
        f.write("\n".join([str(c) for c in counts]))

    if profile_stages:
        print(format_stage_report(stage_profiler, total_frames=len(counts)))
    print(f"Counts saved to {output_path / 'counts.txt'}")
    return counts


def infer(
    model_folder,
    vid_file,
    output_folder,
    model_type="simultaneous",
    decode_mode="stream",
    realtime=False,
    profile_stages=False,
    dll_path=None,
    pipeline_mode="sequential",
    use_prenms_detector=True,
    require_prenms_detector=False,
    det_keyframe_interval=3,
    mp_queue_size=8,
    mp_pose_batch_size=1,
):
    return infer_online(
        model_folder=model_folder,
        video_file=vid_file,
        output_folder=output_folder,
        model_type=model_type,
        decode_mode=decode_mode,
        realtime=realtime,
        profile_stages=profile_stages,
        dll_path=dll_path,
        pipeline_mode=pipeline_mode,
        use_prenms_detector=use_prenms_detector,
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
        default="alternating",
        choices=["simultaneous", "alternating"],
    )
    parser.add_argument("--video_file", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output dir")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "nnapi", "cuda"],
        help="ONNX Runtime provider preference. 'nnapi' falls back to CPU; 'cuda' uses CUDA if available.",
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
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show preprocessing/inference progress",
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
        help="Force real-time mode (stream decode, no processed frame dumps).",
    )
    parser.add_argument(
        "--profile_stages",
        action="store_true",
        help="Print decode/det/pose/feature/score stage timing summary.",
    )
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        default="sequential",
        choices=["sequential", "multiprocess"],
        help="Inference pipeline mode: sequential in-process or queue-based multiprocess pipeline.",
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
    infer_online(
        model_folder=args.model_folder,
        video_file=args.video_file,
        output_folder=args.output_folder,
        model_type=args.model_type,
        device=args.device,
        save_processed=args.save_processed,
        show_progress=args.show_progress,
        decode_mode=args.decode_mode,
        realtime=args.realtime,
        profile_stages=args.profile_stages,
        dll_path=args.dll_path,
        pipeline_mode=args.pipeline_mode,
        use_prenms_detector=not args.disable_prenms_detector,
        require_prenms_detector=args.require_prenms_detector,
        det_keyframe_interval=args.det_keyframe_interval,
        mp_queue_size=args.mp_queue_size,
        mp_pose_batch_size=args.mp_pose_batch_size,
    )
