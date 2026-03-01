import sys

sys.path.append(".")

from pathlib import Path
from collections import OrderedDict
from time import perf_counter
import subprocess
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inferer import OnlineJNGInferer
from inferer.utils.model_path_op import resolve_pose_preprocess_model_files
from inferer.utils.prenms_op import ensure_prenms_detector_model
from inferer.utils.dll_op import configure_dll_path
from inferer.utils.perf_op import StageProfiler, format_stage_report
from inferer.utils.progress_op import build_progress_callback
from modules.preprocessor import iter_online_processed_frames
from modules.preprocessor.preprocess.utils.video_ops import create_video


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
FONT_PATH = str(BIN_DIR / "superbasic.ttf")


def _resolve_binary_model_paths(use_prenms_detector=True, require_prenms_detector=False):
    detection_model_path, keypoint_model_path = resolve_pose_preprocess_model_files(BIN_DIR)
    if use_prenms_detector:
        detection_model_path = ensure_prenms_detector_model(
            detection_model_path,
            required=require_prenms_detector,
        )
    return str(detection_model_path), str(keypoint_model_path)


def _load_font(size=96):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _save_counted_frame(frame, count, output_path, draw_font):
    img = _render_counted_frame(frame, count, draw_font)
    img.save(output_path)


def _render_counted_frame(frame, count, draw_font):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"Count: {count}", font=draw_font, fill=(0, 255, 0))
    return img


class _FFmpegVideoWriter:
    def __init__(self, ffmpeg_bin, output_file, fps=24):
        self.ffmpeg_bin = ffmpeg_bin
        self.output_file = str(output_file)
        self.fps = int(fps)
        self.proc = None
        self.width = None
        self.height = None

    def _start(self, width, height):
        self.width = int(width)
        self.height = int(height)
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            self.output_file,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_rgb):
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("Expected frame with shape HxWx3 in RGB.")
        if self.proc is None:
            self._start(frame_rgb.shape[1], frame_rgb.shape[0])
        if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
            raise ValueError("All frames must have the same size for video writing.")
        self.proc.stdin.write(frame_rgb.tobytes())

    def close(self):
        if self.proc is None:
            return
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        stderr_text = ""
        if self.proc.stderr is not None:
            stderr_text = self.proc.stderr.read().decode("utf-8", errors="ignore")
            self.proc.stderr.close()
        retcode = self.proc.wait()
        self.proc = None
        if retcode != 0:
            raise RuntimeError(
                f"ffmpeg video writer failed with code {retcode}: {stderr_text.strip()}"
            )


def infer_and_write(
    model_folder,
    video_file,
    output_folder,
    model_type="simultaneous",
    device="auto",
    show_progress=False,
    visualize_tracking=False,
    decode_mode="stream",
    realtime=False,
    profile_stages=False,
    dll_path=None,
    use_prenms_detector=True,
    require_prenms_detector=False,
    det_keyframe_interval=3,
):
    configure_dll_path(dll_path)

    if realtime and visualize_tracking:
        print("Realtime mode disables tracking frame dumps.")
        visualize_tracking = False
    if realtime:
        decode_mode = "stream"

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    use_stream_writer = bool(realtime)
    count_frame_path = None
    if not use_stream_writer:
        count_frame_path = output_path / "counts_frames"
        count_frame_path.mkdir(parents=True, exist_ok=True)
    tracking_frame_path = None
    if visualize_tracking:
        tracking_frame_path = output_path / "tracking_frames"
        tracking_frame_path.mkdir(parents=True, exist_ok=True)

    stage_profiler = StageProfiler() if profile_stages else None
    inferer = OnlineJNGInferer(
        model_folder=model_folder,
        model_type=model_type,
        device=device,
        stage_profiler=stage_profiler,
    )
    detection_model_path, keypoint_model_path = _resolve_binary_model_paths(
        use_prenms_detector=use_prenms_detector,
        require_prenms_detector=require_prenms_detector,
    )
    draw_font = _load_font()
    callback_ptr, close_progress = build_progress_callback(show_progress)
    total_frames_hint = {"value": None}
    if callback_ptr is not None:
        base_callback = callback_ptr()

        def wrapped_callback(desc, frame_id, total_frames):
            total_frames_hint["value"] = total_frames
            base_callback(desc, frame_id, total_frames)

        callback_ptr = lambda: wrapped_callback

    write_progress = None
    if show_progress:
        try:
            from tqdm import tqdm

            write_progress = tqdm(total=0, dynamic_ncols=True, desc="Writing frames")
        except Exception:
            write_progress = None

    pending_frames = OrderedDict()
    next_frame_to_write = 0
    counts = []
    written_count = 0
    write_start = perf_counter()
    output_video = output_path / "output.mp4"
    output_video.unlink(missing_ok=True)
    video_writer = _FFmpegVideoWriter(FFMPEG_PATH, output_video, fps=24) if use_stream_writer else None

    def report_write_progress():
        nonlocal written_count
        if not show_progress:
            return
        total = total_frames_hint["value"]
        elapsed = max(perf_counter() - write_start, 1e-6)
        fps = written_count / elapsed
        if write_progress is not None:
            if total is not None and write_progress.total != total:
                write_progress.total = total
            write_progress.update(1)
            write_progress.set_postfix({"fps": f"{fps:.2f}"})
            return

        if written_count == 1 or written_count % 50 == 0 or (total is not None and written_count == total):
            if total is None:
                print(f"Writing frames {written_count}/? ({fps:.2f} FPS)")
            else:
                print(f"Writing frames {written_count}/{total} ({fps:.2f} FPS)")

    try:
        for frame_id, raw_frame, crop_img, keypoints in iter_online_processed_frames(
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
            pending_frames[frame_id] = (
                raw_frame,
                crop_img if visualize_tracking else None,
            )
            inferer.step(crop_img, keypoints)

            while next_frame_to_write < len(inferer.finalized_counts):
                frame_pair = pending_frames.pop(next_frame_to_write, None)
                if frame_pair is None:
                    raise RuntimeError(
                        f"Missing buffered raw frame for finalized frame {next_frame_to_write}."
                    )
                raw_frame, tracking_frame = frame_pair
                write_stage_start = perf_counter()
                if use_stream_writer:
                    counted_img = _render_counted_frame(
                        raw_frame,
                        inferer.finalized_counts[next_frame_to_write],
                        draw_font,
                    )
                    video_writer.write(np.asarray(counted_img))
                else:
                    _save_counted_frame(
                        raw_frame,
                        inferer.finalized_counts[next_frame_to_write],
                        count_frame_path / f"{next_frame_to_write:06d}.jpg",
                        draw_font,
                    )
                if tracking_frame_path is not None:
                    _save_counted_frame(
                        tracking_frame,
                        inferer.finalized_counts[next_frame_to_write],
                        tracking_frame_path / f"{next_frame_to_write:06d}.jpg",
                        draw_font,
                    )
                if stage_profiler is not None:
                    stage_profiler.add("write", perf_counter() - write_stage_start)
                written_count += 1
                report_write_progress()
                next_frame_to_write += 1

        counts = inferer.finalize()

        while next_frame_to_write < len(counts):
            frame_pair = pending_frames.pop(next_frame_to_write, None)
            if frame_pair is None:
                raise RuntimeError(
                    f"Missing buffered raw frame for tail frame {next_frame_to_write}."
                )
            raw_frame, tracking_frame = frame_pair
            write_stage_start = perf_counter()
            if use_stream_writer:
                counted_img = _render_counted_frame(
                    raw_frame,
                    counts[next_frame_to_write],
                    draw_font,
                )
                video_writer.write(np.asarray(counted_img))
            else:
                _save_counted_frame(
                    raw_frame,
                    counts[next_frame_to_write],
                    count_frame_path / f"{next_frame_to_write:06d}.jpg",
                    draw_font,
                )
            if tracking_frame_path is not None:
                _save_counted_frame(
                    tracking_frame,
                    counts[next_frame_to_write],
                    tracking_frame_path / f"{next_frame_to_write:06d}.jpg",
                    draw_font,
                )
            if stage_profiler is not None:
                stage_profiler.add("write", perf_counter() - write_stage_start)
            written_count += 1
            report_write_progress()
            next_frame_to_write += 1
    finally:
        close_progress()
        if write_progress is not None:
            write_progress.close()
        if video_writer is not None:
            video_writer.close()

    if not use_stream_writer:
        create_video(FFMPEG_PATH, count_frame_path, output_video)

    with open(output_path / "counts.txt", "w") as f:
        f.write("\n".join([str(c) for c in counts]))

    if profile_stages:
        print(format_stage_report(stage_profiler, total_frames=len(counts)))
    print(f"Counts saved to {output_path / 'counts.txt'}")
    print(f"Output video saved to {output_video}")
    return counts


def infer(
    model_folder,
    vid_file,
    output_folder,
    model_type="simultaneous",
    show_progress=False,
    visualize_tracking=False,
    decode_mode="stream",
    realtime=False,
    profile_stages=False,
    dll_path=None,
    use_prenms_detector=True,
    require_prenms_detector=False,
    det_keyframe_interval=3,
):
    return infer_and_write(
        model_folder=model_folder,
        video_file=vid_file,
        output_folder=output_folder,
        model_type=model_type,
        show_progress=show_progress,
        visualize_tracking=visualize_tracking,
        decode_mode=decode_mode,
        realtime=realtime,
        profile_stages=profile_stages,
        dll_path=dll_path,
        use_prenms_detector=use_prenms_detector,
        require_prenms_detector=require_prenms_detector,
        det_keyframe_interval=det_keyframe_interval,
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
        "--visualize_tracking",
        action="store_true",
        help="Also write inference tracking frames to output_folder/tracking_frames",
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
        help="Force low-latency mode (stream decode + stream video write, no intermediate jpgs).",
    )
    parser.add_argument(
        "--profile_stages",
        action="store_true",
        help="Print decode/det/pose/feature/score/write stage timing summary.",
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
        "--dll_path",
        type=str,
        default=None,
        help="Optional directory containing native runtime/delegate DLLs to preload before inference.",
    )
    args = parser.parse_args()
    infer_and_write(
        model_folder=args.model_folder,
        video_file=args.video_file,
        output_folder=args.output_folder,
        model_type=args.model_type,
        device=args.device,
        show_progress=args.show_progress,
        visualize_tracking=args.visualize_tracking,
        decode_mode=args.decode_mode,
        realtime=args.realtime,
        profile_stages=args.profile_stages,
        dll_path=args.dll_path,
        use_prenms_detector=not args.disable_prenms_detector,
        require_prenms_detector=args.require_prenms_detector,
        det_keyframe_interval=args.det_keyframe_interval,
    )
