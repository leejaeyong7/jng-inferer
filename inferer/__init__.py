from pathlib import Path
from .utils.inference_op import parallel_load_data
from .utils.score_op import scores_to_counts_simu, scores_to_counts_alt
from .utils.onnx_op import (
    load_frame_feature_extractor,
    load_score_computer,
    batchfiy_latents,
)


def compute_scores(
    model_folder, processed_path, model_type="simultaneous", callback_ptr=None
):
    model_path = Path(model_folder)
    processed_path = Path(processed_path)
    # output_path contains images / keypoints folder
    imgs, kps = parallel_load_data(processed_path)

    # run inference with the loaded imgs and kps
    ffe = load_frame_feature_extractor(model_path / "jng.feature.onnx")
    sco = load_score_computer(model_path / "jng.scorer.onnx")
    if callback_ptr is not None:
        callback = callback_ptr()
    else:
        callback = None

    zero_latent, latents = ffe(imgs, kps, callback=callback)
    frame_offsets = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8]
    batch_latents = batchfiy_latents(frame_offsets, zero_latent, latents)
    if callback_ptr is not None:
        callback = callback_ptr()
    else:
        callback = None
    scores = sco(batch_latents)

    if model_type == "simultaneous":
        scores_to_counts = scores_to_counts_simu
    else:
        scores_to_counts = scores_to_counts_alt
    counts = scores_to_counts(scores)
    return counts
