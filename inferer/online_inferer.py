from pathlib import Path
import numpy as np
from time import perf_counter

from .utils.onnx_op import load_onnx_model
from .utils.model_path_op import resolve_feature_score_model_files
from .utils.score_op import SimultaneousCounter, AlternatingCounter


DEFAULT_FRAME_OFFSETS = (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8)


def _tensor_dtype(model_input):
    scale, _ = getattr(model_input, "quantization", (0.0, 0))
    if float(scale) > 0.0:
        # Quantized TFLite tensors still receive float preprocessing input in this pipeline.
        return np.float32
    if model_input.type == "tensor(float16)":
        return np.float16
    return np.float32


def _prepare_frame_input(img):
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected RGB frame with shape HxWx3.")
    arr = arr.astype(np.float32) / 255.0
    return arr


def _infer_model_image_layout(shape_without_batch):
    if len(shape_without_batch) != 3:
        raise ValueError(
            f"Expected 3D image shape without batch, got {shape_without_batch}."
        )
    if shape_without_batch[0] == 3 and shape_without_batch[2] != 3:
        return "nchw"
    if shape_without_batch[2] == 3:
        return "nhwc"
    raise ValueError(f"Could not infer image layout from shape {shape_without_batch}.")


def _align_image_layout(img, target_layout):
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got {arr.shape}.")

    src_is_chw = arr.shape[0] == 3 and arr.shape[2] != 3
    src_is_hwc = arr.shape[2] == 3
    if not src_is_chw and not src_is_hwc:
        raise ValueError(
            f"Could not infer image layout from shape {arr.shape}. Expected CHW or HWC."
        )

    if target_layout == "nchw":
        return arr if src_is_chw else arr.transpose(2, 0, 1)
    return arr if src_is_hwc else arr.transpose(1, 2, 0)


class OnlineJNGInferer:
    """
    Stateful online inferer that updates count per frame with bounded history.
    """

    def __init__(
        self,
        model_folder,
        model_type="simultaneous",
        frame_offsets=DEFAULT_FRAME_OFFSETS,
        device="auto",
        stage_profiler=None,
    ):
        self.model_folder = Path(model_folder)
        self.frame_offsets = tuple(frame_offsets)
        self.min_offset = min(self.frame_offsets)
        self.max_offset = max(self.frame_offsets)

        ffe_model_path, sco_model_path = resolve_feature_score_model_files(self.model_folder)
        self.ffe_sess, self.ffe_inputs, self.ffe_outputs = load_onnx_model(
            ffe_model_path,
            device=device,
        )
        self.sco_sess, self.sco_inputs, self.sco_outputs = load_onnx_model(
            sco_model_path,
            device=device,
        )
        self.ffe_dtype = _tensor_dtype(self.ffe_inputs[0])
        self.kp_dtype = _tensor_dtype(self.ffe_inputs[1])
        self.sco_dtype = _tensor_dtype(self.sco_inputs[0])
        self.ffe_image_shape = tuple(int(dim) for dim in self.ffe_inputs[0].shape[1:])
        self.ffe_image_layout = _infer_model_image_layout(self.ffe_image_shape)
        self.kp_shape = tuple(int(dim) for dim in self.ffe_inputs[1].shape[1:])
        self.model_type = model_type
        self.stage_profiler = stage_profiler

        if model_type == "alternating":
            self.counter = AlternatingCounter()
        else:
            self.counter = SimultaneousCounter()

        # Absolute frame index -> latent vector. Kept bounded by context window.
        self.latent_history = {}
        self.latest_frame_idx = -1
        self.next_score_frame_idx = 0
        self.finalized_counts = []
        self.finalized_scores = []
        self.live_count = 0
        self.zero_latent = self._compute_zero_latent()
        self._validate_model_type_against_scorer()

    def _validate_model_type_against_scorer(self):
        probe_context = np.stack([self.zero_latent for _ in self.frame_offsets], axis=0)
        probe_context = probe_context.astype(self.sco_dtype, copy=False)
        probe_score = self.sco_sess.run(
            [self.sco_outputs[0].name], {self.sco_inputs[0].name: probe_context[None]}
        )[0][0]
        score_dim = int(np.asarray(probe_score).shape[-1])
        if self.model_type == "alternating" and score_dim != 5:
            raise ValueError(
                f"Alternating model_type expects scorer output dim 5, but got {score_dim}. "
                "Use matching --model_type and model weights."
            )
        if self.model_type == "simultaneous" and score_dim != 4:
            raise ValueError(
                f"Simultaneous model_type expects scorer output dim 4, but got {score_dim}. "
                "Use matching --model_type and model weights."
            )

    def _compute_zero_latent(self):
        zero_img = np.zeros(self.ffe_image_shape, dtype=self.ffe_dtype)
        zero_kp = np.zeros(self.kp_shape, dtype=self.kp_dtype)
        try:
            return self._extract_latent(zero_img, zero_kp)
        except RuntimeError:
            # Keep inference robust if model-side probing fails unexpectedly.
            latent_dim = int(self.sco_inputs[0].shape[-1])
            return np.zeros((latent_dim,), dtype=self.sco_dtype)

    def _extract_latent(self, img_tensor, keypoints):
        aligned_img = _align_image_layout(img_tensor, self.ffe_image_layout)
        kp = np.asarray(keypoints)
        if tuple(kp.shape) != self.kp_shape:
            raise ValueError(
                f"Keypoint shape mismatch. Expected {self.kp_shape}, got {tuple(kp.shape)}."
            )
        latent = self.ffe_sess.run(
            [self.ffe_outputs[0].name],
            {
                self.ffe_inputs[0].name: aligned_img[None].astype(self.ffe_dtype),
                self.ffe_inputs[1].name: kp[None].astype(self.kp_dtype),
            },
        )[0][0]
        return latent.astype(self.sco_dtype, copy=False)

    def _latent_for_index(self, frame_idx, use_zero_for_missing):
        # Match offline batching behavior: pre-sequence frames are zero-padded.
        if frame_idx < 0:
            return self.zero_latent
        latent = self.latent_history.get(frame_idx)
        if latent is not None:
            return latent
        if use_zero_for_missing:
            return self.zero_latent
        raise KeyError(frame_idx)

    def _build_context(self, target_frame_idx, use_zero_for_missing):
        context = []
        for offset in self.frame_offsets:
            src_idx = target_frame_idx + offset
            context.append(self._latent_for_index(src_idx, use_zero_for_missing))
        return np.stack(context, axis=0).astype(self.sco_dtype, copy=False)

    def _score_frame(self, frame_idx, use_zero_for_missing):
        context = self._build_context(frame_idx, use_zero_for_missing=use_zero_for_missing)
        score = self.sco_sess.run(
            [self.sco_outputs[0].name], {self.sco_inputs[0].name: context[None]}
        )[0][0]
        self.live_count = int(self.counter.update(score))
        self.finalized_scores.append(score)
        self.finalized_counts.append(self.live_count)
        return score

    def _gc_history(self):
        keep_from = self.next_score_frame_idx + self.min_offset
        for idx in [k for k in self.latent_history.keys() if k < keep_from]:
            del self.latent_history[idx]

    def step(self, crop_img, keypoints):
        img_tensor = _prepare_frame_input(crop_img).astype(self.ffe_dtype)
        keypoints = np.asarray(keypoints, dtype=self.kp_dtype)
        feature_start = perf_counter()
        latent = self._extract_latent(img_tensor, keypoints)
        if self.stage_profiler is not None:
            self.stage_profiler.add("feature", perf_counter() - feature_start)

        self.latest_frame_idx += 1
        self.latent_history[self.latest_frame_idx] = latent

        # Score only frames with full future context already available.
        ready_until = self.latest_frame_idx - self.max_offset
        while self.next_score_frame_idx <= ready_until:
            score_start = perf_counter()
            self._score_frame(
                self.next_score_frame_idx,
                use_zero_for_missing=False,
            )
            if self.stage_profiler is not None:
                self.stage_profiler.add("score", perf_counter() - score_start)
            self.next_score_frame_idx += 1

        self._gc_history()
        return self.live_count

    def finalize(self):
        """
        Flush remaining tail frames by zero-padding missing future context,
        mirroring the original offline batch behavior at sequence end.
        """
        while self.next_score_frame_idx <= self.latest_frame_idx:
            score_start = perf_counter()
            self._score_frame(
                self.next_score_frame_idx,
                use_zero_for_missing=True,
            )
            if self.stage_profiler is not None:
                self.stage_profiler.add("score", perf_counter() - score_start)
            self.next_score_frame_idx += 1
        self._gc_history()
        return list(self.finalized_counts)

    def get_finalized_scores(self):
        if len(self.finalized_scores) == 0:
            return np.empty((0, 0), dtype=self.sco_dtype)
        return np.stack(self.finalized_scores, axis=0)
