"""Measure ball sizes from sampled video frames and save one Excel file per video.

Run summarize_ball_sizes.py afterward to create ball_sizes_means.csv.
"""
import argparse
from pathlib import Path
import re

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
VIDEO_DIR = BASE_DIR / "video"  # Used when --video is not provided.
OUTPUT_DIR = BASE_DIR / "ball_size_measurements"
SAMPLES_PER_VIDEO = 100
DETECTION_CONFIDENCE = 0.25
DETECTION_NMS_IOU = 0.2
VIDEO_PATTERN = re.compile(r"^(\d+)r_(\d+)\.mp4$", re.IGNORECASE)

def measure_video(video_path: Path, model: YOLO,
                  confidence: float = DETECTION_CONFIDENCE,
                  nms_iou: float = DETECTION_NMS_IOU, *,
                  device: str | None = None, output_dir: Path = OUTPUT_DIR,
                  max_frames: int = 20000,
                  require_ball_count: bool = False) -> tuple[int, int]:
    """Sample up to 100 frames; optionally reject frames with the wrong ball count."""
    if max_frames < 0:
        raise ValueError("max_frames must be 0 or a positive integer")
    expected_balls = None
    if require_ball_count:
        match = VIDEO_PATTERN.fullmatch(video_path.name)
        if match is None or int(match.group(2)) <= 0:
            raise ValueError("Ball count check requires a <rpm>r_<balls>.mp4 name with positive ball count")
        expected_balls = int(match.group(2))
    output_dir = output_dir.expanduser().resolve()
    inference_kwargs = {"conf": confidence, "iou": nms_iou, "verbose": False}
    if device is not None:
        inference_kwargs["device"] = device
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video frame count unavailable: {video_path}")
        frame_limit = min(max_frames, total_frames) if max_frames else total_frames
        sample_count = min(SAMPLES_PER_VIDEO, frame_limit)
        # OpenCV seeks with zero-based indices. Preserve the original equal-interval
        # sampling rule, now applied to the selected prefix of the video.
        frame_indices = [
            (i * frame_limit) // sample_count
            for i in range(sample_count)
        ]
        sizes = []
        valid_frames = 0
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Cannot read frame {frame_index} from {video_path}"
                )
            # Match the preprocessing used by extract_velocities.py.
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            detections = model(frame, **inference_kwargs)[0].boxes.data.cpu().numpy()
            balls = [detection for detection in detections if int(detection[-1]) == 0]
            if require_ball_count and len(balls) != expected_balls:
                continue  # Discard the entire sampled frame; do not replace it.
            frame_sizes = []
            for detection in balls:
                x1, y1, x2, y2 = detection[:4]  # Keep YOLO coordinates in pixels.
                width = x2 - x1
                height = y2 - y1
                if width > 0 and height > 0:
                    frame_sizes.append((width, height))
            if require_ball_count and len(frame_sizes) != expected_balls:
                continue  # A frame with invalid box dimensions cannot be complete.
            if frame_sizes:
                sizes.extend(frame_sizes)
                valid_frames += 1
    finally:
        cap.release()

    count_check = f"on (expected {expected_balls})" if require_ball_count else "off"
    summary = (f"{video_path.name}: frames 1-{frame_limit} (inclusive), "
               f"{sample_count} sampled frames, {valid_frames} valid frames, "
               f"{sample_count - valid_frames} discarded frames, {len(sizes)} ball boxes; "
               f"ball-count check={count_check}")
    if not sizes:
        raise ValueError(f"No valid ball detections to save. {summary}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_ball_sizes.xlsx"
    pd.DataFrame(
        sizes, columns=["Width (pixels)", "Height (pixels)"]
    ).to_excel(output_path, index=False)
    print(f"{summary} -> {output_path}")
    return sample_count, len(sizes)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Measure ball sizes from video frames")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--video", type=Path,
        help="Process one video file",
    )
    source.add_argument(
        "--video-dir", type=Path,
        help="Process all matching MP4 files directly inside this directory",
    )
    parser.add_argument("--weights", type=Path, default=MODEL_PATH,
                        help="Model weights (default: best.pt beside this script)")
    parser.add_argument("--device", default=None,
                        help="Ultralytics device, e.g. 0 for the first CUDA GPU (default: auto)")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 convolution (default: disabled, matching velocity extraction)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for Excel files (default: 2_GetResults/ball_size_measurements)")
    parser.add_argument("--max-frames", type=int, default=20000,
                        help="Use only the first N frames for sampling (default: 20000; "
                             "0 uses the full video); sample at most 100 frames within this limit")
    parser.add_argument("--conf", type=float, default=DETECTION_CONFIDENCE,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--nms-iou", type=float, default=DETECTION_NMS_IOU,
                        help="YOLO NMS overlap threshold (default: 0.2)")
    parser.add_argument("--require-ball-count", action="store_true",
                        help="Keep a sampled frame only when its ball count matches the filename; "
                             "discard mismatched frames without replacement (default: disabled)")
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1 or not 0 <= args.nms_iou <= 1:
        parser.error("--conf and --nms-iou must be between 0 and 1")
    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or a positive integer")

    if args.video is not None:
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            parser.error(f"Video file not found: {video_path}")
        if not VIDEO_PATTERN.fullmatch(video_path.name):
            parser.error("Video name must match <rpm>r_<balls>.mp4")
        videos = [video_path]
    else:
        video_dir = args.video_dir.expanduser().resolve() if args.video_dir else VIDEO_DIR
        if not video_dir.is_dir():
            parser.error(f"Video directory not found: {video_dir}")
        videos = sorted(
            (path for path in video_dir.glob("*.mp4")
             if path.is_file() and VIDEO_PATTERN.fullmatch(path.name)),
            key=lambda path: tuple(map(int, VIDEO_PATTERN.fullmatch(path.name).groups())),
        )
        if not videos:
            parser.error(f"No <rpm>r_<balls>.mp4 videos found in {video_dir}")

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        parser.error(f"Weights file not found: {weights_path}")
    output_dir = args.output_dir.expanduser().resolve()
    # Match extract_velocities.py before constructing the model or running inference.
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    print(f"Model weights: {weights_path}")
    print(f"YOLO inference: conf={args.conf:g}, NMS IoU={args.nms_iou:g}; "
          f"device={args.device if args.device is not None else 'auto'}; "
          f"conv precision={'TF32' if args.tf32 else 'strict fp32'}")
    model = YOLO(str(weights_path))
    for video_path in videos:
        try:
            measure_video(video_path, model, confidence=args.conf, nms_iou=args.nms_iou,
                          device=args.device, output_dir=output_dir,
                          max_frames=args.max_frames, require_ball_count=args.require_ball_count)
        except (ValueError, RuntimeError) as exc:
            parser.exit(2, f"Error: {exc}\n")
    print(f"Measured {len(videos)} video(s). Run summarize_ball_sizes.py to create the CSV.")


if __name__ == "__main__":
    main()
