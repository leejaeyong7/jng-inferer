from pathlib import Path
from .utils.inference_op import parallel_load_data
from .utils.video_ops import extract_video, create_video, enumerate_vid, get_vid_count
from .utils.score_op import scores_to_counts
from .utils.onnx_op import (
    load_frame_feature_extractor,
    load_score_computer,
    batchfiy_latents,
)


def compute_scores(model_folder, processed_path):
    model_path = Path(model_folder)
    processed_path = Path(processed_path)
    # output_path contains images / keypoints folder
    imgs, kps = parallel_load_data(processed_path)

    # run inference with the loaded imgs and kps
    ffe = load_frame_feature_extractor(model_path / "jng.feature.onnx")
    sco = load_score_computer(model_path / "jng.scorer.onnx")

    zero_latent, latents = ffe(imgs, kps)
    frame_offsets = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8]
    batch_latents = batchfiy_latents(frame_offsets, zero_latent, latents)
    scores = sco(batch_latents)

    counts = scores_to_counts(scores)
    return counts
