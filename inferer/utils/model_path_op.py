from pathlib import Path


POSE_DETECTION_MODEL_FILENAMES = (
    "jng.det.prenms.fp16.onnx",
    "jng.det.prenms.onnx",
    "jng.det.fp16.onnx",
    "jng.det.onnx",
    "jng.det.inferred.onnx",
)
POSE_KEYPOINT_MODEL_FILENAMES = (
    "jng.pose.fp16.onnx",
    "jng.pose.onnx",
)
FEATURE_MODEL_FILENAMES = (
    "jng.feature.fp16.onnx",
    "jng.feature.onnx",
)
SCORER_MODEL_FILENAMES = (
    "jng.scorer.fp16.onnx",
    "jng.scorer.onnx",
)


def _unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _candidate_model_dirs(model_dir):
    model_dir = Path(model_dir)
    candidates = [
        model_dir,
        model_dir / "onnx",
        model_dir.parent / "onnx",
    ]

    parts = model_dir.parts
    if "int8_weights" in parts:
        idx = parts.index("int8_weights")
        alt_parts = list(parts)
        alt_parts[idx] = "weights"
        alt_dir = Path(*alt_parts)
        candidates.extend(
            [
                alt_dir,
                alt_dir / "onnx",
                alt_dir.parent / "onnx",
            ]
        )

    return _unique_preserve_order(candidates)


def _resolve_required_model_file(model_dir, model_filenames, model_label):
    model_dirs = _candidate_model_dirs(model_dir)
    searched = []
    for model_filename in model_filenames:
        for candidate_dir in model_dirs:
            candidate = Path(candidate_dir) / model_filename
            searched.append(candidate)
            if candidate.exists():
                return candidate

    searched_lines = "\n".join([f"  - {path}" for path in searched])
    raise FileNotFoundError(
        f"Could not find required {model_label} model under {Path(model_dir)}.\n"
        f"Searched:\n{searched_lines}"
    )


def resolve_pose_preprocess_model_files(model_dir):
    detection_model_path = _resolve_required_model_file(
        model_dir,
        POSE_DETECTION_MODEL_FILENAMES,
        "pose detection",
    )
    keypoint_model_path = _resolve_required_model_file(
        model_dir,
        POSE_KEYPOINT_MODEL_FILENAMES,
        "pose keypoint",
    )
    return detection_model_path, keypoint_model_path


def resolve_feature_score_model_files(model_dir):
    feature_model_path = _resolve_required_model_file(
        model_dir,
        FEATURE_MODEL_FILENAMES,
        "feature extractor",
    )
    scorer_model_path = _resolve_required_model_file(
        model_dir,
        SCORER_MODEL_FILENAMES,
        "scorer",
    )
    return feature_model_path, scorer_model_path
