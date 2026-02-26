import sys

sys.path.append(".")

from pathlib import Path
from collections import OrderedDict
from time import perf_counter
from PIL import Image, ImageDraw, ImageFont

from inferer import OnlineJNGInferer
from inferer.utils.progress_op import build_progress_callback
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


def _save_counted_frame(frame, count, output_path, draw_font):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"Count: {count}", font=draw_font, fill=(0, 255, 0))
    img.save(output_path)


def infer_and_write(
    model_folder,
    video_file,
    output_folder,
    model_type="simultaneous",
    device="auto",
    show_progress=False,
    visualize_tracking=False,
):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    count_frame_path = output_path / "counts_frames"
    count_frame_path.mkdir(parents=True, exist_ok=True)
    tracking_frame_path = None
    if visualize_tracking:
        tracking_frame_path = output_path / "tracking_frames"
        tracking_frame_path.mkdir(parents=True, exist_ok=True)

    inferer = OnlineJNGInferer(
        model_folder=model_folder,
        model_type=model_type,
        device=device,
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
            detection_model_path=DETECTION_MODEL_PATH,
            keypoint_model_path=KEYPOINT_MODEL_PATH,
            video_path=Path(video_file),
            callback_ptr=callback_ptr,
            device=device,
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
            written_count += 1
            report_write_progress()
            next_frame_to_write += 1
    finally:
        close_progress()
        if write_progress is not None:
            write_progress.close()

    output_video = output_path / "output.mp4"
    output_video.unlink(missing_ok=True)
    create_video(FFMPEG_PATH, count_frame_path, output_video)

    with open(output_path / "counts.txt", "w") as f:
        f.write("\n".join([str(c) for c in counts]))

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
):
    return infer_and_write(
        model_folder=model_folder,
        video_file=vid_file,
        output_folder=output_folder,
        model_type=model_type,
        show_progress=show_progress,
        visualize_tracking=visualize_tracking,
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
    args = parser.parse_args()
    infer_and_write(
        model_folder=args.model_folder,
        video_file=args.video_file,
        output_folder=args.output_folder,
        model_type=args.model_type,
        device=args.device,
        show_progress=args.show_progress,
        visualize_tracking=args.visualize_tracking,
    )
