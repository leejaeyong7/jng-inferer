from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def _import_onnx():
    try:
        import onnx

        return onnx
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Preparing a pre-NMS detector requires the 'onnx' package. "
            "Install it with: pip install onnx"
        ) from exc


def _infer_prenms_output_path(detector_model_path: Path) -> Path:
    if detector_model_path.name.startswith("jng.det"):
        return detector_model_path.with_name("jng.det.prenms.onnx")
    return detector_model_path.with_name(f"{detector_model_path.stem}.prenms.onnx")


def _find_value_info(model, tensor_name):
    for info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if info.name == tensor_name:
            return deepcopy(info)
    return None


def _make_fallback_value_info(onnx_module, tensor_name):
    return onnx_module.helper.make_tensor_value_info(
        tensor_name,
        onnx_module.TensorProto.FLOAT,
        None,
    )


def ensure_prenms_detector_model(detector_model_path, required=False):
    """
    Ensure a pre-NMS detector ONNX exists and return its path.

    If `detector_model_path` is already pre-NMS (name contains 'prenms'), it is returned as-is.
    If conversion fails and `required=False`, the original model path is returned.
    If conversion fails and `required=True`, an exception is raised.
    """
    detector_model_path = Path(detector_model_path)
    if "prenms" in detector_model_path.name.lower():
        return detector_model_path

    prenms_path = _infer_prenms_output_path(detector_model_path)
    if prenms_path.exists():
        return prenms_path

    try:
        onnx = _import_onnx()
        model = onnx.load(str(detector_model_path))
        nms_nodes = [node for node in model.graph.node if node.op_type == "NonMaxSuppression"]
        if not nms_nodes:
            raise RuntimeError(
                f"No NonMaxSuppression node found in detector model: {detector_model_path}"
            )
        nms_node = nms_nodes[0]
        if len(nms_node.input) < 2:
            raise RuntimeError(
                f"NonMaxSuppression node has insufficient inputs in model: {detector_model_path}"
            )

        boxes_name = nms_node.input[0]
        scores_name = nms_node.input[1]
        boxes_vi = _find_value_info(model, boxes_name)
        scores_vi = _find_value_info(model, scores_name)
        if boxes_vi is None:
            boxes_vi = _make_fallback_value_info(onnx, boxes_name)
        if scores_vi is None:
            scores_vi = _make_fallback_value_info(onnx, scores_name)

        model.graph.ClearField("output")
        model.graph.output.extend([boxes_vi, scores_vi])

        # Keep validation best-effort because some exported graphs have partial shape metadata.
        try:
            onnx.checker.check_model(model)
        except Exception:
            pass

        onnx.save(model, str(prenms_path))
        return prenms_path
    except Exception as exc:  # noqa: BLE001
        if required:
            raise RuntimeError(
                f"Failed to prepare pre-NMS detector from {detector_model_path}: {exc}"
            ) from exc
        print(
            "Warning: Could not prepare pre-NMS detector. "
            f"Falling back to model outputs from {detector_model_path}. Reason: {exc}"
        )
        return detector_model_path
