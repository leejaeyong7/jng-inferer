import numpy as np
import onnxruntime as ort
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed


def _resolve_providers(device="auto"):
    available = set(ort.get_available_providers())
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this runtime.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    ordered = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "NNAPIExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [p for p in ordered if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def load_onnx_model(onnx_model_path, device="auto"):
    sess = ort.InferenceSession(str(onnx_model_path), providers=_resolve_providers(device))
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    return sess, inputs, outputs


def extract_frame_feature(ffe_sess, ffe_inputs, ffe_outputs, inputs):
    img, kp = inputs
    return ffe_sess.run(
        [ffe_outputs[0].name],
        {
            ffe_inputs[0].name: img[None],
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
    zero_latent = extract_frame_feature(
        ffe_sess, ffe_inputs, ffe_outputs, (zero_img, zero_kp)
    )

    return zero_latent, latents


def batchfiy_latents(frame_offsets, zero_latent, latents):
    B, D = latents.shape
    min_offset = min(frame_offsets)
    max_offset = max(frame_offsets)
    pre_zero_latents = zero_latent.repeat(-min_offset, 0)  # (-min_offset, D)
    post_zero_latents = zero_latent.repeat(max_offset, 0)  # (
    padded_latents = np.concatenate(
        [pre_zero_latents, latents, post_zero_latents]
    )  # (B, D)
    indices = np.arange(len(latents)) + (-min_offset)
    frame_indices = (indices[:, None] + np.array(frame_offsets)[None, :]).reshape(-1)
    frame_latents = padded_latents[frame_indices].reshape(
        B, len(frame_offsets), D
    )  # FxBxD
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
    """
    callback: function(i, total) called after each score is computed
    """
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
