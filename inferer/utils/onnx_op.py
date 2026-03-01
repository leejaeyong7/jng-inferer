import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np


_REPORTED_CUDA_FALLBACK_MODELS = set()


def _read_int_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _read_bool_env(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _import_onnxruntime():
    try:
        import onnxruntime as ort

        return ort
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "onnxruntime is required for ONNX inference but was not importable. "
            "Install it with: pip install onnxruntime"
        ) from exc


def _normalize_ort_device(device):
    requested = (device or "auto").lower()
    if requested not in {"auto", "cpu", "nnapi", "cuda"}:
        raise ValueError(
            f"Unsupported device '{device}'. Supported values: auto, cpu, nnapi, cuda."
        )
    if requested == "nnapi":
        print("Warning: ONNX Runtime does not support NNAPI in this setup. Falling back to CPU.")
        return "cpu"
    if requested == "auto":
        return "cpu"
    return requested


def _onnx_type_to_dtype(tensor_type):
    type_map = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int8)": np.int8,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(bool)": np.bool_,
    }
    return np.dtype(type_map.get(tensor_type, np.float32))


def _onnx_shape_tuple(shape):
    dims = []
    for dim in shape:
        if isinstance(dim, int):
            dims.append(int(dim))
        else:
            dims.append(-1)
    return tuple(dims)


def _build_probe_input(meta):
    shape = [dim if dim > 0 else 1 for dim in _onnx_shape_tuple(meta.shape)]
    dtype = _onnx_type_to_dtype(meta.type)
    return np.zeros(shape, dtype=dtype)


def _probe_cpu_fallback_ops(ort, model_path, providers, intra_threads, inter_threads):
    probe_options = ort.SessionOptions()
    probe_options.enable_profiling = True
    if intra_threads > 0:
        probe_options.intra_op_num_threads = int(intra_threads)
    if inter_threads > 0:
        probe_options.inter_op_num_threads = int(inter_threads)

    probe_session = ort.InferenceSession(
        str(model_path),
        sess_options=probe_options,
        providers=providers,
    )

    feed = {}
    for input_meta in probe_session.get_inputs():
        feed[input_meta.name] = _build_probe_input(input_meta)
    probe_session.run(None, feed)
    profile_path = probe_session.end_profiling()

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    cpu_ops = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        duration = event.get("dur")
        if duration is None:
            continue
        args = event.get("args", {})
        provider = args.get("provider")
        if provider != "CPUExecutionProvider":
            continue
        op_name = args.get("op_name") or args.get("op_type") or "UNKNOWN"
        cpu_ops[op_name] = cpu_ops.get(op_name, 0.0) + float(duration)
    return sorted(cpu_ops.items(), key=lambda item: item[1], reverse=True)


def _maybe_report_cuda_fallback_ops(
    ort,
    model_path,
    resolved_device,
    session,
    intra_threads,
    inter_threads,
):
    if resolved_device != "cuda":
        return

    active_providers = tuple(session.get_providers())
    model_name = Path(model_path).name
    if "CUDAExecutionProvider" not in active_providers:
        print(
            f"[ORT][CUDA] {model_name}: CUDA provider inactive. "
            f"Active providers: {list(active_providers)}"
        )
        return

    if not _read_bool_env("JNG_ORT_PRINT_CPU_FALLBACK_OPS", True):
        return

    report_key = (str(Path(model_path).resolve()), active_providers)
    if report_key in _REPORTED_CUDA_FALLBACK_MODELS:
        return
    _REPORTED_CUDA_FALLBACK_MODELS.add(report_key)

    try:
        cpu_ops = _probe_cpu_fallback_ops(
            ort=ort,
            model_path=model_path,
            providers=list(active_providers),
            intra_threads=intra_threads,
            inter_threads=inter_threads,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[ORT][CUDA fallback] {model_name}: failed to collect fallback op report ({exc})."
        )
        return

    if not cpu_ops:
        print(f"[ORT][CUDA fallback] {model_name}: no CPU fallback ops detected.")
        return

    total_us = sum(duration for _, duration in cpu_ops)
    preview = ", ".join(
        [f"{op}({duration / 1000.0:.3f}ms)" for op, duration in cpu_ops[:10]]
    )
    print(
        f"[ORT][CUDA fallback] {model_name}: {len(cpu_ops)} CPU op types, "
        f"total={total_us / 1000.0:.3f}ms in probe. Top: {preview}"
    )


@dataclass
class _TensorInfo:
    name: str
    type: str
    index: int
    dtype: np.dtype
    shape: tuple[int, ...]
    quantization: tuple[float, int]


class _OrtSession:
    def __init__(self, session):
        self._session = session
        self._lock = threading.Lock()
        self._inputs = list(session.get_inputs())
        self._outputs = list(session.get_outputs())
        self._input_by_name = {meta.name: meta for meta in self._inputs}
        self._output_by_name = {meta.name: meta for meta in self._outputs}

    def _check_shape(self, meta, actual_shape):
        expected_shape = _onnx_shape_tuple(meta.shape)
        if len(expected_shape) != len(actual_shape):
            raise ValueError(
                f"Input '{meta.name}' expects rank {len(expected_shape)} (shape {expected_shape}), "
                f"got shape {actual_shape}."
            )
        for expected_dim, actual_dim in zip(expected_shape, actual_shape):
            if expected_dim >= 0 and expected_dim != int(actual_dim):
                raise ValueError(
                    f"Input '{meta.name}' expects shape {expected_shape}, got {actual_shape}."
                )

    def run(self, output_names, feed_dict):
        with self._lock:
            prepared_inputs = {}
            for input_name, input_value in feed_dict.items():
                meta = self._input_by_name.get(input_name)
                if meta is None:
                    valid = ", ".join(sorted(self._input_by_name.keys()))
                    raise KeyError(
                        f"Unknown model input '{input_name}'. Available inputs: {valid}"
                    )
                arr = np.asarray(input_value, dtype=_onnx_type_to_dtype(meta.type))
                self._check_shape(meta, tuple(int(dim) for dim in arr.shape))
                prepared_inputs[input_name] = arr

            if output_names is None:
                selected_output_names = [meta.name for meta in self._outputs]
            else:
                selected_output_names = []
                for output_name in output_names:
                    meta = self._output_by_name.get(output_name)
                    if meta is None:
                        valid = ", ".join(sorted(self._output_by_name.keys()))
                        raise KeyError(
                            f"Unknown model output '{output_name}'. Available outputs: {valid}"
                        )
                    selected_output_names.append(meta.name)

            return self._session.run(selected_output_names, prepared_inputs)

    def get_inputs(self):
        infos = []
        for meta in self._inputs:
            infos.append(
                _TensorInfo(
                    name=meta.name,
                    type=meta.type,
                    index=-1,
                    dtype=_onnx_type_to_dtype(meta.type),
                    shape=_onnx_shape_tuple(meta.shape),
                    quantization=(0.0, 0),
                )
            )
        return infos

    def get_outputs(self):
        infos = []
        for meta in self._outputs:
            infos.append(
                _TensorInfo(
                    name=meta.name,
                    type=meta.type,
                    index=-1,
                    dtype=_onnx_type_to_dtype(meta.type),
                    shape=_onnx_shape_tuple(meta.shape),
                    quantization=(0.0, 0),
                )
            )
        return infos


def _create_ort_session(model_path, device):
    ort = _import_onnxruntime()
    resolved_device = _normalize_ort_device(device)

    available_providers = set(ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    if resolved_device == "cuda":
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print(
                "Warning: CUDAExecutionProvider not available in onnxruntime. "
                "Falling back to CPU."
            )

    session_options = ort.SessionOptions()
    intra_threads = _read_int_env("JNG_ORT_INTRA_OP_THREADS", max(1, min(4, os.cpu_count() or 1)))
    inter_threads = _read_int_env("JNG_ORT_INTER_OP_THREADS", 1)
    if intra_threads > 0:
        session_options.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        session_options.inter_op_num_threads = inter_threads

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )
    _maybe_report_cuda_fallback_ops(
        ort=ort,
        model_path=model_path,
        resolved_device=resolved_device,
        session=session,
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )
    return session


def load_onnx_model(onnx_model_path, device="auto"):
    model_path = Path(onnx_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError(
            f"ONNX-only runtime expects a .onnx model file, got: {model_path}"
        )

    session = _create_ort_session(model_path=model_path, device=device)
    sess = _OrtSession(session)
    return sess, sess.get_inputs(), sess.get_outputs()


def _infer_image_layout_from_input_shape(input_shape):
    if len(input_shape) != 4:
        raise ValueError(
            f"Expected 4D image input shape [N, C/H, H/W, W/C], got {input_shape}."
        )
    if input_shape[1] == 3 and input_shape[3] != 3:
        return "nchw"
    if input_shape[3] == 3:
        return "nhwc"
    raise ValueError(f"Could not infer image layout from input shape {input_shape}.")


def _align_image_for_model_input(img, input_info):
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {arr.shape}.")

    src_is_chw = arr.shape[0] == 3 and arr.shape[2] != 3
    src_is_hwc = arr.shape[2] == 3
    if not src_is_chw and not src_is_hwc:
        raise ValueError(
            f"Could not infer source image layout from shape {arr.shape}. Expected CHW or HWC."
        )

    layout = _infer_image_layout_from_input_shape(tuple(int(dim) for dim in input_info.shape))
    if layout == "nchw":
        return arr if src_is_chw else arr.transpose(2, 0, 1)
    return arr if src_is_hwc else arr.transpose(1, 2, 0)


def extract_frame_feature(ffe_sess, ffe_inputs, ffe_outputs, inputs):
    img, kp = inputs
    aligned_img = _align_image_for_model_input(img, ffe_inputs[0])
    return ffe_sess.run(
        [ffe_outputs[0].name],
        {
            ffe_inputs[0].name: aligned_img[None],
            ffe_inputs[1].name: kp[None],
        },
    )[0]


def extract_frame_features(ffe_sess, ffe_inputs, ffe_outputs, imgs, kps, callback=None):
    inputs = list(zip(imgs, kps))
    extract_frame_feature_partial = partial(
        extract_frame_feature, ffe_sess, ffe_inputs, ffe_outputs
    )
    total = len(inputs)
    latents_list = []
    latents_dict = {}
    msg = "프레임 특징 추출 중..."
    with ThreadPoolExecutor(max_workers=16) as executor:
        task_dict = {}

        for iid, i in enumerate(inputs):
            task = executor.submit(extract_frame_feature_partial, i)
            task_dict[task] = iid

        for task in as_completed(task_dict):
            iid = task_dict[task]
            latents_dict[iid] = task.result()

            if callback:
                callback(msg, iid, total)

    for i in range(total):
        latents_list.append(latents_dict[i])

    latents = np.concatenate(latents_list, axis=0)
    zero_img = np.zeros_like(imgs[0])
    zero_kp = np.zeros_like(kps[0])
    try:
        zero_latent = extract_frame_feature(
            ffe_sess, ffe_inputs, ffe_outputs, (zero_img, zero_kp)
        )
    except Exception:  # noqa: BLE001
        zero_latent = np.zeros((1, latents.shape[-1]), dtype=latents.dtype)

    return zero_latent, latents


def batchfiy_latents(frame_offsets, zero_latent, latents):
    B, D = latents.shape
    min_offset = min(frame_offsets)
    max_offset = max(frame_offsets)
    pre_zero_latents = zero_latent.repeat(-min_offset, 0)  # (-min_offset, D)
    post_zero_latents = zero_latent.repeat(max_offset, 0)
    padded_latents = np.concatenate([pre_zero_latents, latents, post_zero_latents])
    indices = np.arange(len(latents)) + (-min_offset)
    frame_indices = (indices[:, None] + np.array(frame_offsets)[None, :]).reshape(-1)
    frame_latents = padded_latents[frame_indices].reshape(B, len(frame_offsets), D)
    return frame_latents


def compute_score(sco_sess, sco_inputs, sco_outputs, latent):
    scores = sco_sess.run(
        [sco_outputs[0].name],
        {
            sco_inputs[0].name: latent[None],
        },
    )
    return scores[0]


def compute_scores(sco_sess, sco_inputs, sco_outputs, latents, callback=None):
    scores = []
    compute_score_partial = partial(compute_score, sco_sess, sco_inputs, sco_outputs)
    total = len(latents)
    score_dict = {}
    msg = "FSD 점수 계산 중..."
    with ThreadPoolExecutor(max_workers=16) as executor:
        task_dict = {}
        for i, latent in enumerate(latents):
            task = executor.submit(compute_score_partial, latent)
            task_dict[task] = i

        for task in as_completed(task_dict):
            i = task_dict[task]
            score_dict[i] = task.result()

            if callback:
                callback(msg, i, total)
    for i in range(total):
        scores.append(score_dict[i])
    return np.concatenate(scores)


def load_frame_feature_extractor(ffe_model_path, device="auto"):
    ffe_sess, ffe_inputs, ffe_outputs = load_onnx_model(ffe_model_path, device=device)
    return partial(extract_frame_features, ffe_sess, ffe_inputs, ffe_outputs)


def load_score_computer(sco_model_path, device="auto"):
    sco_sess, sco_inputs, sco_outputs = load_onnx_model(sco_model_path, device=device)
    return partial(compute_scores, sco_sess, sco_inputs, sco_outputs)
