import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from functools import partial
from concurrent.futures import ThreadPoolExecutor


def load_onnx_model(onnx_model_path):
    sess = ort.InferenceSession(str(onnx_model_path))
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


def extract_frame_features(ffe_sess, ffe_inputs, ffe_outputs, imgs, kps):
    latents_list = []
    inputs = list(zip(imgs, kps))
    with ThreadPoolExecutor(max_workers=16) as executor:
        latents_list = list(
            tqdm(
                executor.map(
                    partial(extract_frame_feature, ffe_sess, ffe_inputs, ffe_outputs),
                    inputs,
                ),
                total=len(inputs),
                desc="Extracting frame features",
            )
        )
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


def compute_scores(sco_sess, sco_inputs, sco_outputs, latents):
    scores = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        scores = list(
            tqdm(
                executor.map(
                    partial(compute_score, sco_sess, sco_inputs, sco_outputs),
                    latents,
                ),
                total=len(latents),
                desc="Computing scores",
            )
        )
    return np.concatenate(scores)


def load_frame_feature_extractor(ffe_model_path):
    ffe_sess, ffe_inputs, ffe_outputs = load_onnx_model(ffe_model_path)
    return partial(extract_frame_features, ffe_sess, ffe_inputs, ffe_outputs)


def load_score_computer(sco_model_path):
    sco_sess, sco_inputs, sco_outputs = load_onnx_model(sco_model_path)
    return partial(compute_scores, sco_sess, sco_inputs, sco_outputs)
