from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
import traceback
from pathlib import Path
from queue import Empty, Full
from time import perf_counter, sleep

import numpy as np
from PIL import Image

from .online_inferer import OnlineJNGInferer
from modules.preprocessor.preprocess.utils.bbox_ops import crop_bbox
from modules.preprocessor.preprocess.utils.rtm_ops import (
    load_keypoint_model,
    load_multi_detection_model,
)
from modules.preprocessor.preprocess.utils.video_ops import (
    enumerate_vid,
    extract_video,
    get_vid_count,
    iter_video_stream,
)
from modules.preprocessor.preprocess.preprocess import _select_online_bbox


_SENTINEL = None
_POLL_SEC = 0.1


def _record_stage(stage_stats, stage, seconds):
    total, count = stage_stats.get(stage, (0.0, 0))
    stage_stats[stage] = (total + float(seconds), count + 1)


def _put_or_wait(queue, item, stop_event):
    while True:
        try:
            queue.put(item, timeout=_POLL_SEC)
            return
        except Full:
            if stop_event.is_set():
                continue


def _get_or_wait(queue, stop_event):
    while True:
        try:
            return queue.get(timeout=_POLL_SEC)
        except Empty:
            if stop_event.is_set():
                return _SENTINEL


def _report_worker_error(stage, err_queue, stop_event):
    stop_event.set()
    err_queue.put(
        {
            "stage": stage,
            "traceback": traceback.format_exc(),
        }
    )


def _frame_feeder_worker(
    ffmpeg_path,
    video_path,
    decode_mode,
    frame_queue,
    stage_queue,
    err_queue,
    stop_event,
):
    stage_stats = {}
    tmp_dir = None
    try:
        video_path = Path(video_path)
        if decode_mode == "stream":
            for frame_id, frame, total_frames_hint in iter_video_stream(
                ffmpeg_bin=ffmpeg_path,
                video_file_path=video_path,
                fps=24,
            ):
                if stop_event.is_set():
                    break
                put_start = perf_counter()
                _put_or_wait(frame_queue, (frame_id, frame, total_frames_hint), stop_event)
                _record_stage(stage_stats, "decode", perf_counter() - put_start)
        elif decode_mode == "extract":
            tmp_dir = Path(tempfile.mkdtemp(prefix="jng_mp_decode_"))
            extract_video(ffmpeg_path, video_path, tmp_dir)
            total_frames = get_vid_count(tmp_dir)
            for frame_id, frame in enumerate_vid(tmp_dir, total=total_frames):
                if stop_event.is_set():
                    break
                put_start = perf_counter()
                _put_or_wait(frame_queue, (frame_id, frame, total_frames), stop_event)
                _record_stage(stage_stats, "decode", perf_counter() - put_start)
        else:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")
    except Exception:  # noqa: BLE001
        _report_worker_error("frame_feeder", err_queue, stop_event)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        _put_or_wait(frame_queue, _SENTINEL, stop_event)
        stage_queue.put(("frame_feeder", stage_stats))


def _detector_crop_worker(
    detection_model_path,
    device,
    crop_buffer,
    det_keyframe_interval,
    top_k,
    iou_weight,
    frame_queue,
    crop_queue,
    stage_queue,
    err_queue,
    stop_event,
):
    stage_stats = {}
    prev_bbox = None
    det_keyframe_interval = max(1, int(det_keyframe_interval))
    try:
        detect_people = load_multi_detection_model(detection_model_path, device=device)
        while not stop_event.is_set():
            item = _get_or_wait(frame_queue, stop_event)
            if item is _SENTINEL:
                break
            frame_id, frame, total_frames_hint = item
            is_det_frame = (prev_bbox is None) or (frame_id % det_keyframe_interval == 0)
            if is_det_frame:
                det_start = perf_counter()
                detections = detect_people(frame)
                _record_stage(stage_stats, "det", perf_counter() - det_start)
                bbox = _select_online_bbox(
                    detections,
                    prev_bbox=prev_bbox,
                    frame_shape=frame.shape,
                    top_k=top_k,
                    iou_weight=iou_weight,
                )
            else:
                bbox = np.asarray(prev_bbox, dtype=np.float32)
            prev_bbox = bbox

            crop_start = perf_counter()
            crop_img = crop_bbox(frame, bbox, buffer=crop_buffer)
            _record_stage(stage_stats, "crop", perf_counter() - crop_start)
            _put_or_wait(crop_queue, (frame_id, crop_img, total_frames_hint), stop_event)
    except Exception:  # noqa: BLE001
        _report_worker_error("detector_crop", err_queue, stop_event)
    finally:
        _put_or_wait(crop_queue, _SENTINEL, stop_event)
        stage_queue.put(("detector_crop", stage_stats))


def _save_processed_frame(output_path, frame_id, crop_img, keypoints):
    prefix = f"{frame_id:06d}"
    Image.fromarray(crop_img).save(output_path / "processed" / "images" / f"{prefix}.jpg")
    np.save(output_path / "processed" / "keypoints" / f"{prefix}.npy", keypoints)


def _pose_score_worker(
    model_folder,
    model_type,
    keypoint_model_path,
    device,
    save_processed,
    output_folder,
    pose_batch_size,
    crop_queue,
    result_queue,
    stage_queue,
    err_queue,
    stop_event,
):
    stage_stats = {}
    next_emit_idx = 0
    buffer = []
    output_path = Path(output_folder)
    pose_batch_size = max(1, int(pose_batch_size))

    def flush_buffer(inferer, detect_keypoints):
        nonlocal buffer
        nonlocal next_emit_idx
        for frame_id, crop_img, total_frames_hint in buffer:
            pose_start = perf_counter()
            keypoints = detect_keypoints(crop_img)
            _record_stage(stage_stats, "pose", perf_counter() - pose_start)

            if save_processed:
                _save_processed_frame(output_path, frame_id, crop_img, keypoints)

            feature_score_start = perf_counter()
            inferer.step(crop_img, keypoints)
            _record_stage(stage_stats, "feature_score", perf_counter() - feature_score_start)

            while next_emit_idx < len(inferer.finalized_counts):
                _put_or_wait(
                    result_queue,
                    (
                        "count",
                        next_emit_idx,
                        int(inferer.finalized_counts[next_emit_idx]),
                        np.asarray(inferer.finalized_scores[next_emit_idx], dtype=np.float32),
                        total_frames_hint,
                    ),
                    stop_event,
                )
                next_emit_idx += 1
        buffer = []

    try:
        detect_keypoints = load_keypoint_model(keypoint_model_path, device=device)
        inferer = OnlineJNGInferer(
            model_folder=model_folder,
            model_type=model_type,
            device=device,
            stage_profiler=None,
        )

        while not stop_event.is_set():
            item = _get_or_wait(crop_queue, stop_event)
            if item is _SENTINEL:
                break
            buffer.append(item)
            if len(buffer) >= pose_batch_size:
                flush_buffer(inferer, detect_keypoints)

        if buffer:
            flush_buffer(inferer, detect_keypoints)

        finalize_start = perf_counter()
        counts = inferer.finalize()
        _record_stage(stage_stats, "finalize", perf_counter() - finalize_start)
        while next_emit_idx < len(counts):
            _put_or_wait(
                result_queue,
                (
                    "count",
                    next_emit_idx,
                    int(counts[next_emit_idx]),
                    np.asarray(inferer.finalized_scores[next_emit_idx], dtype=np.float32),
                    None,
                ),
                stop_event,
            )
            next_emit_idx += 1

        _put_or_wait(result_queue, ("done", len(counts)), stop_event)
    except Exception:  # noqa: BLE001
        _report_worker_error("pose_score", err_queue, stop_event)
        _put_or_wait(result_queue, ("done", 0), stop_event)
    finally:
        stage_queue.put(("pose_score", stage_stats))


def _result_sink_worker(
    output_folder,
    show_progress,
    result_queue,
    done_queue,
    stage_queue,
    err_queue,
    stop_event,
):
    stage_stats = {}
    counts_by_frame = {}
    next_frame = 0
    total_hint = None
    expected_total = None
    started = perf_counter()

    try:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        while not stop_event.is_set():
            msg = _get_or_wait(result_queue, stop_event)
            if msg is _SENTINEL:
                continue
            tag = msg[0]
            if tag == "count":
                _, frame_id, count, _, frame_total_hint = msg
                if frame_total_hint is not None:
                    total_hint = int(frame_total_hint)
                counts_by_frame[int(frame_id)] = int(count)
                while next_frame in counts_by_frame:
                    next_frame += 1
                    if show_progress and (next_frame == 1 or next_frame % 50 == 0):
                        elapsed = max(perf_counter() - started, 1e-6)
                        fps = next_frame / elapsed
                        if total_hint is None:
                            print(f"Result sink: {next_frame}/? ({fps:.2f} FPS)")
                        else:
                            print(f"Result sink: {next_frame}/{total_hint} ({fps:.2f} FPS)")
            elif tag == "done":
                expected_total = int(msg[1])
                break

        if expected_total is None:
            expected_total = next_frame

        missing = [idx for idx in range(expected_total) if idx not in counts_by_frame]
        if missing:
            raise RuntimeError(
                f"Missing computed counts for frames: {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )

        write_start = perf_counter()
        counts = [counts_by_frame[idx] for idx in range(expected_total)]
        with open(output_path / "counts.txt", "w") as file_obj:
            file_obj.write("\n".join([str(c) for c in counts]))
        _record_stage(stage_stats, "write", perf_counter() - write_start)

        done_queue.put({"ok": True, "frames": expected_total})
    except Exception:  # noqa: BLE001
        _report_worker_error("result_sink", err_queue, stop_event)
        done_queue.put({"ok": False})
    finally:
        stage_queue.put(("result_sink", stage_stats))


def _merge_stage_stats(stage_stats_items):
    totals = {}
    counts = {}
    for _, stats in stage_stats_items:
        for stage_name, (total, count) in stats.items():
            totals[stage_name] = totals.get(stage_name, 0.0) + float(total)
            counts[stage_name] = counts.get(stage_name, 0) + int(count)
    return totals, counts


def _format_multiprocess_stage_report(stage_stats_items, total_frames):
    totals, counts = _merge_stage_stats(stage_stats_items)
    if not totals:
        return "Performance profile (multiprocess): no stage samples collected."

    wall = sum(totals.values())
    lines = ["Performance profile (multiprocess):"]
    if total_frames is not None and wall > 0:
        lines.append(f"  frames={total_frames}, approx_stage_time={wall:.3f}s")
    preferred_order = ["decode", "det", "crop", "pose", "feature_score", "finalize", "write"]
    known = [name for name in preferred_order if name in totals]
    extra = sorted([name for name in totals.keys() if name not in preferred_order])
    for stage in known + extra:
        total = totals[stage]
        count = max(1, counts.get(stage, 0))
        avg_ms = 1000.0 * total / count
        lines.append(f"  {stage:>12}: total={total:.3f}s avg={avg_ms:.2f}ms count={count}")
    return "\n".join(lines)


def infer_online_multiprocess(
    model_folder,
    video_file,
    output_folder,
    model_type,
    detection_model_path,
    keypoint_model_path,
    ffmpeg_path,
    device="auto",
    save_processed=False,
    show_progress=False,
    decode_mode="stream",
    crop_buffer=1.3,
    det_keyframe_interval=3,
    top_k=5,
    iou_weight=0.4,
    queue_size=8,
    pose_batch_size=1,
    profile_stages=False,
):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    if save_processed:
        (output_path / "processed" / "images").mkdir(parents=True, exist_ok=True)
        (output_path / "processed" / "keypoints").mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=max(1, int(queue_size)))
    crop_queue = ctx.Queue(maxsize=max(1, int(queue_size)))
    result_queue = ctx.Queue(maxsize=max(1, int(queue_size)))
    stage_queue = ctx.Queue()
    err_queue = ctx.Queue()
    done_queue = ctx.Queue(maxsize=1)
    stop_event = ctx.Event()

    processes = [
        ctx.Process(
            target=_frame_feeder_worker,
            name="jng-frame-feeder",
            args=(
                str(ffmpeg_path),
                str(video_file),
                decode_mode,
                frame_queue,
                stage_queue,
                err_queue,
                stop_event,
            ),
        ),
        ctx.Process(
            target=_detector_crop_worker,
            name="jng-detector-crop",
            args=(
                str(detection_model_path),
                device,
                float(crop_buffer),
                int(det_keyframe_interval),
                int(top_k),
                float(iou_weight),
                frame_queue,
                crop_queue,
                stage_queue,
                err_queue,
                stop_event,
            ),
        ),
        ctx.Process(
            target=_pose_score_worker,
            name="jng-pose-score",
            args=(
                str(model_folder),
                model_type,
                str(keypoint_model_path),
                device,
                bool(save_processed),
                str(output_folder),
                int(pose_batch_size),
                crop_queue,
                result_queue,
                stage_queue,
                err_queue,
                stop_event,
            ),
        ),
        ctx.Process(
            target=_result_sink_worker,
            name="jng-result-sink",
            args=(
                str(output_folder),
                bool(show_progress),
                result_queue,
                done_queue,
                stage_queue,
                err_queue,
                stop_event,
            ),
        ),
    ]

    for process in processes:
        process.start()

    done_payload = None
    try:
        while done_payload is None:
            if not err_queue.empty():
                stop_event.set()
                break
            try:
                done_payload = done_queue.get(timeout=0.2)
                break
            except Empty:
                if all(not process.is_alive() for process in processes):
                    break
                sleep(0.05)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get())
    if errors:
        details = "\n".join([f"[{e['stage']}]\n{e['traceback']}" for e in errors])
        raise RuntimeError(f"Multiprocess inference failed:\n{details}")
    if not done_payload or not done_payload.get("ok"):
        raise RuntimeError("Multiprocess inference did not complete successfully.")

    counts_path = output_path / "counts.txt"
    if not counts_path.exists():
        raise RuntimeError(f"Expected output file not found: {counts_path}")
    with open(counts_path, "r") as file_obj:
        counts = [int(line.strip()) for line in file_obj if line.strip()]

    if profile_stages:
        stage_items = []
        while not stage_queue.empty():
            stage_items.append(stage_queue.get())
        print(_format_multiprocess_stage_report(stage_items, total_frames=len(counts)))

    return counts
