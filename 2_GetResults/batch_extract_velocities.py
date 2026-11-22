"""Run extract_velocities.py separately for every video in a folder."""

import argparse
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "processed_data"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract velocities from all videos in one folder")
    parser.add_argument("video_dir", nargs="?", type=Path, default=SCRIPT_DIR / "video",
                        help="Folder containing videos (default: video beside this script)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for CSV files (default: 2_GetResults/processed_data)")
    parser.add_argument("--max-frames", type=int, default=20000,
                        help="Maximum frames per video (default: 20000; 0 processes full videos)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--nms-iou", type=float, default=0.2,
                        help="YOLO NMS overlap threshold (default: 0.2)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Frames per YOLO call (default: 16); 1 uses one frame per inference call")
    parser.add_argument("--prep-workers", type=int, default=4,
                        help="Background threads that rotate/convert frames (default: 4)")
    parser.add_argument("--serial", action="store_true",
                        help="Run each video frame by frame on one thread; "
                             "--batch-size and --prep-workers are ignored")
    parser.add_argument("--weights", type=Path, default=SCRIPT_DIR / "best.pt",
                        help="Model weights for every video (default: best.pt beside this script)")
    parser.add_argument("--device", default=None,
                        help="Ultralytics device for every video, e.g. 0 (default: auto)")
    parser.add_argument("--sort-iou", type=float, default=0.3,
                        help="SORT matching IoU for every video "
                             "(default: 0.3; use --sort-iou 0.5 for stricter matching)")
    parser.add_argument("--enforce-ball-count", action="store_true",
                        help="Keep at most the expected ball count per frame by confidence; "
                             "frames with fewer balls are retained "
                             "(each video reads its own count from its file name)")
    parser.add_argument("--ball-size-gate", type=float, default=0.0,
                        help="Lower box-size cutoff relative to the running median for every video "
                             "(0 disables it)")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 convolution on compatible GPUs for every video "
                             "(default: disabled)")
    parser.add_argument("--expected-balls", type=int,
                        help="Override the ball count for every video; otherwise infer each filename")
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or a positive integer")
    if not 0 <= args.conf <= 1 or not 0 <= args.nms_iou <= 1:
        parser.error("--conf and --nms-iou must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    if args.prep_workers < 0:
        parser.error("--prep-workers must be 0 or a positive integer")
    if not 0 <= args.ball_size_gate < 1:
        parser.error("--ball-size-gate must be in [0, 1)")
    if not 0 < args.sort_iou <= 1:
        parser.error("--sort-iou must be in (0, 1]")
    if args.expected_balls is not None and args.expected_balls <= 0:
        parser.error("--expected-balls must be positive")

    # Resolve relative weights before starting children with cwd=SCRIPT_DIR.
    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        parser.error(f"Weights file not found: {weights_path}")
    video_dir = args.video_dir.expanduser().resolve()
    if not video_dir.is_dir():
        parser.error(f"Video folder not found: {video_dir}")
    videos = sorted((p for p in video_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                    key=lambda p: p.name.lower())
    if not videos:
        parser.error(f"No supported video files in: {video_dir}")

    # Output names are based on stems; reject collisions before processing anything.
    stems = [p.stem.lower() for p in videos]
    if len(stems) != len(set(stems)):
        parser.error("Two videos have the same base name and would overwrite one CSV")

    output_dir = args.output_dir.expanduser().resolve()
    failed = []
    for index, video in enumerate(videos, 1):
        print(f"\n[{index}/{len(videos)}] Processing {video.name}", flush=True)
        command = [sys.executable, str(SCRIPT_DIR / "extract_velocities.py"),
                   "--video", str(video), "--output-dir", str(output_dir),
                   "--weights", str(weights_path),
                   "--max-frames", str(args.max_frames),
                   "--conf", str(args.conf), "--nms-iou", str(args.nms_iou),
                   "--sort-iou", str(args.sort_iou),
                   "--batch-size", str(args.batch_size),
                   "--prep-workers", str(args.prep_workers)]
        if args.serial:
            command.append("--serial")
        if args.device is not None:
            command.extend(["--device", str(args.device)])
        if args.tf32:
            command.append("--tf32")
        if args.ball_size_gate:
            command.extend(["--ball-size-gate", str(args.ball_size_gate)])
        if args.enforce_ball_count:
            command.append("--enforce-ball-count")
        if args.expected_balls is not None:
            command.extend(["--expected-balls", str(args.expected_balls)])
        result = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            check=False,
        )
        if result.returncode:
            failed.append(video.name)
            print(f"Failed: {video.name} (exit code {result.returncode})", flush=True)

    print(f"\nFinished: {len(videos) - len(failed)}/{len(videos)} videos succeeded.")
    if failed:
        print("Failed videos: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
