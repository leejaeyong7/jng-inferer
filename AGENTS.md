# jng-inferer Contributor Notes

## Purpose
`jng-inferer` runs ONNX-based jump-rope inference from videos and outputs counts/scores.

## Main Areas
- `run.py`: end-to-end preprocessing + scoring.
- `run_inference.py`: inference flow without video write.
- `infer_and_write.py`: inference plus output video rendering.
- `modules/preprocessor/`: local preprocessing dependency.

## Run Commands
Install dependencies and run from repository root:

```bash
pip install -r jng-inferer/requirements.txt
python jng-inferer/run_inference.py --model_folder <dir> --model_type <str> --video_file <video> --output_folder <out>
python jng-inferer/infer_and_write.py --model_folder <dir> --model_type <str> --video_file <video> --output_folder <out>
```

## Development Guidance
- Keep CLI arguments backward-compatible; scripts are used in batch jobs.
- Validate outputs (`counts.txt`, score arrays, rendered videos) on short clips before merging.
- Keep model/binary paths configurable; do not hardcode deployment-only paths.
