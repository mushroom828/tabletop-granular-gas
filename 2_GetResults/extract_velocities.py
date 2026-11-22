"""Extract ball and container velocities from one video.

Two execution modes:

* parallel (default) - decoding and frame preparation run on background threads
  while the GPU infers on the previous batch, and tracking runs on its own
  thread.
* serial (``--serial``) - one frame in flight at a time on a single thread, the
  way the first published version of this script worked.

Both modes process frames in order with the same tracking steps. Numerical
results can depend on hardware, precision settings and inference batch size.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import csv
import queue
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from container_motion import ContainerMotionTracker
from container_velocity import fit_container_velocities
from extraction_diagnostics import infer_expected_balls, make_ball_filter_event, report_frame_counts
from sort import Sort

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "processed_data"
DETECTION_CONFIDENCE = 0.25
DETECTION_NMS_IOU = 0.2  # Detection deduplication, separate from SORT's matching IoU.


# Object detection using YOLOv8
def split_detections(data):
    """Split one frame's rows into ball boxes, the best container box, and its count."""
    balls_detection = []
    container_xyxy = None
    container_detection_count = 0
    best_container_confidence = -1.0
    for object in data:
        category = object[-1]
        if category == 0:  # ball position
            xyxyc = object[:-1]  # xyxyc: [x1, y1, x2, y2, conf]
            balls_detection.append(xyxyc)
        if category == 1:
            container_detection_count += 1
            if object[4] > best_container_confidence:  # container position
                container_xyxy = object[:4]  # container_xyxy: [x1, y1, x2, y2]
                best_container_confidence = object[4]
    return balls_detection, container_xyxy, container_detection_count


class BallSizeGate:
    """Apply an optional lower size bound relative to this video's typical ball.

    Removed boxes are rule-based exclusions, not verified misdetections.
    """

    def __init__(self, ratio=0.85, window=2000, min_samples=100):
        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio must be between 0 and 1")
        self.ratio = float(ratio)
        self.min_samples = int(min_samples)
        self._sizes = deque(maxlen=int(window))
        self.accepted = 0
        self.dropped = 0
        self.last_threshold = None

    @staticmethod
    def _diameter(box):
        width = float(box[2]) - float(box[0])
        height = float(box[3]) - float(box[1])
        return (max(width, 0.0) * max(height, 0.0)) ** 0.5

    @property
    def threshold(self):
        """Cutoff in pixels, or None while the median is still bootstrapping."""
        if len(self._sizes) < self.min_samples:
            return None
        return self.ratio * float(np.median(self._sizes))

    def __call__(self, balls):
        if not balls:
            self.last_threshold = None
            return balls
        threshold = self.threshold
        self.last_threshold = threshold
        kept = []
        for box in balls:
            if threshold is not None and self._diameter(box) < threshold:
                self.dropped += 1
                continue
            kept.append(box)
            self._sizes.append(self._diameter(box))
        self.accepted += len(kept)
        return kept


def prepare_frame(item):
    """Rotate one decoded frame and derive the grayscale copy used by tracking."""
    index, timestamp, frame = item
    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return index, timestamp, rotated, cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)


class VideoFrameBatcher:
    """Yield [(index, timestamp, rotated_bgr, gray), ...] batches in frame order.

    With threaded=True decoding runs on its own thread and frame preparation on
    a small ordered pool, so both overlap the caller's GPU work. With
    threaded=False the class degenerates to a plain read-prepare-yield loop on
    the caller's thread, which is what --serial selects.

    index is the one-based frame number and timestamp is the presentation time
    read from CAP_PROP_POS_MSEC right after the frame is decoded, so both match
    a plain sequential cap.read() loop exactly.
    """

    _END = object()

    def __init__(self, video_path, max_frames=0, batch_size=16, prep_workers=2,
                 threaded=True):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if prep_workers < 0:
            raise ValueError("prep_workers must be zero or positive")
        self.video_path = str(video_path)
        self.max_frames = int(max_frames)
        self.batch_size = int(batch_size)
        self.prep_workers = int(prep_workers)
        self.threaded = bool(threaded)
        self._stop = threading.Event()
        self._error = None

    def __iter__(self):
        if not self.threaded:
            return self._iterate_synchronous()
        return self._iterate_threaded()

    def _iterate_synchronous(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                index += 1
                yield [prepare_frame((index, timestamp, frame))]
                if self.max_frames and index >= self.max_frames:
                    break
        finally:
            cap.release()

    def _deliver(self, raw_queue, item, force=False):
        """Block until the queue accepts item; give up once the consumer stops."""
        while not self._stop.is_set():
            try:
                raw_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        if force:
            try:
                raw_queue.put_nowait(item)
            except queue.Full:
                pass
        return False

    def _decode(self, raw_queue):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._error = RuntimeError(f"Cannot open video: {self.video_path}")
            self._deliver(raw_queue, self._END)
            return
        index = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                index += 1
                if self.prep_workers == 0:
                    item = prepare_frame((index, timestamp, frame))
                else:
                    item = (index, timestamp, frame)
                if not self._deliver(raw_queue, item):
                    return
                if self.max_frames and index >= self.max_frames:
                    break
        except BaseException as exc:  # surfaced on the consumer side
            self._error = exc
        finally:
            cap.release()
            self._deliver(raw_queue, self._END, force=True)

    def _iterate_threaded(self):
        self._stop.clear()
        self._error = None
        raw_queue = queue.Queue(maxsize=self.batch_size * 2)
        decoder = threading.Thread(target=self._decode, args=(raw_queue,),
                                   name="video-decode", daemon=True)
        decoder.start()
        pool = None
        if self.prep_workers:
            pool = ThreadPoolExecutor(max_workers=self.prep_workers,
                                      thread_name_prefix="frame-prep")
        try:
            finished = False
            while not finished:
                chunk = []
                while len(chunk) < self.batch_size:
                    item = raw_queue.get()
                    if item is self._END:
                        finished = True
                        break
                    chunk.append(item)
                if not chunk:
                    break
                yield list(pool.map(prepare_frame, chunk)) if pool else chunk
        finally:
            self._stop.set()
            if pool is not None:
                pool.shutdown(wait=True)
            decoder.join(timeout=10)
        # Raised after cleanup and never from the finally block, so an early
        # close() cannot turn into "generator ignored GeneratorExit".
        if self._error is not None:
            error, self._error = self._error, None
            raise error


class Object_SpeedTracker:
    def __init__(self):
        self.tracking_history = {}  # recent positions, timestamps, and frame indices
        self.speed_history = {}     # track speeds
    
    def update(self, track_id, bbox, frame_index, timestamp):
        """
        Update the position information of the tracked object and calculate speed.
        
        :param track_id: ID of the tracked object
        :param bbox: bounding box [x1, y1, x2, y2]
        :param frame_index: one-based video frame number
        :param timestamp: video presentation timestamp in seconds
        """
        if track_id not in self.tracking_history:
            self.tracking_history[track_id] = deque(maxlen=2)
            self.speed_history[track_id] = []

        values = np.asarray(bbox, dtype=float)
        history = self.tracking_history[track_id]
        if (values.shape != (4,) or not np.isfinite(values).all()
                or values[2] <= values[0] or values[3] <= values[1]):
            history.clear()
            return
        x1, y1, x2, y2 = values
        center_position = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
        if history and frame_index != history[-1][2] + 1:
            history.clear()  # SORT may keep an ID across missing detections.
        if not np.all(np.isfinite(center_position)) or not np.isfinite(timestamp):
            history.clear()
            return
        history.append((center_position, float(timestamp), frame_index))
        if len(history) < 2:
            return

        previous_position, previous_time, _ = history[0]
        time_diff = timestamp - previous_time
        if time_diff <= 0:
            history.clear()
            return
        # Position difference across consecutive frames, divided by video time (px/s).
        speed = (center_position - previous_position) / time_diff
        self.speed_history[track_id].append((speed, frame_index))


def track_worker(track_queue, errors, process_frame):
    """Consume inference results on a second thread so tracking overlaps the next batch."""
    try:
        while True:
            item = track_queue.get()
            if item is None:
                break
            gray_frames, results = item
            for (frame_number, timestamp, gray), result in zip(gray_frames, results):
                # Copy detections straight off the GPU here so inference never waits on it.
                process_frame(frame_number, timestamp, gray, result.boxes.data.to("cpu").numpy())
    except BaseException as exc:    # surfaced from the main thread
        errors.append(exc)


def put_tracking_item(track_queue, worker, item):
    """Wait for queue space only while the tracking thread can still consume it."""
    while worker.is_alive():
        try:
            track_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def main(argv=None):
    """Parse arguments and extract one video; runtime state belongs to this call."""
    parser = argparse.ArgumentParser(description="Extract ball and container velocities from one video")
    parser.add_argument("--video", type=Path, required=True,
                        help="Input video file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the velocity CSV (default: 2_GetResults/processed_data)")
    parser.add_argument("--max-frames", type=int, default=20000,
                        help="Maximum frames to process (default: 20000; 0 processes the full video)")
    parser.add_argument("--conf", type=float, default=DETECTION_CONFIDENCE,
                        help="YOLO confidence threshold (default: 0.25, the Ultralytics predict default)")
    parser.add_argument("--nms-iou", type=float, default=DETECTION_NMS_IOU,
                        help="YOLO NMS overlap threshold (default: 0.2; not SORT matching IoU)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Frames per YOLO call (default: 16); 1 uses one frame per inference call")
    parser.add_argument("--prep-workers", type=int, default=4,
                        help="Background threads that rotate and convert frames while the GPU "
                             "runs (default: 4; 0 prepares frames on the decoder thread)")
    parser.add_argument("--serial", action="store_true",
                        help="Process one frame at a time on a single thread; "
                             "--batch-size and --prep-workers are ignored")
    parser.add_argument("--weights", type=Path, default=SCRIPT_DIR / "best.pt",
                        help="Model weights (default: best.pt beside this script)")
    parser.add_argument("--device", default=None,
                        help="Ultralytics device, e.g. 0 for the first CUDA GPU (default: auto)")
    parser.add_argument("--sort-iou", type=float, default=0.3,
                        help="Minimum IoU between ball detections and predicted tracks "
                             "(default: 0.3; use --sort-iou 0.5 for stricter matching)")
    parser.add_argument("--enforce-ball-count", action="store_true",
                        help="Keep at most the expected ball count per frame by dropping the "
                             "lowest-confidence surplus boxes; frames with fewer balls are retained. "
                             "Needs a known count: --expected-balls, or a name like 120r_10")
    parser.add_argument("--ball-size-gate", type=float, default=0.0,
                        help="Discard ball boxes whose size sqrt(width*height) is below this "
                             "fraction of the running median, e.g. 0.85; "
                             "0 disables the size-based check (default: 0)")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 convolution on compatible GPUs (default: disabled); "
                             "numerical and performance effects depend on hardware and batch size")
    parser.add_argument("--expected-balls", type=int,
                        help="Physical ball count for diagnostics; inferred from <rpm>r_<balls> if omitted")
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
    # Disable TF32 convolution by default; configure it before the first inference call.
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"Video file not found: {video_path}")
    video_name = video_path.stem
    output_dir = args.output_dir.expanduser().resolve()
    expected_balls = args.expected_balls if args.expected_balls is not None else infer_expected_balls(video_name)
    expected_source = ("--expected-balls" if args.expected_balls is not None
                       else "filename" if expected_balls is not None else "unknown")
    if expected_balls is not None and expected_balls <= 0:
        parser.error("The expected ball count must be positive")
    if args.enforce_ball_count and expected_balls is None:
        parser.error("--enforce-ball-count needs a known ball count: pass --expected-balls or "
                     "use a video name such as 120r_10")
    print(f"YOLO inference: conf={args.conf:g}, NMS IoU={args.nms_iou:g}; "
          f"SORT matching IoU={args.sort_iou:g}; "
          f"device={args.device if args.device is not None else 'auto'}; "
          f"conv precision={'TF32' if args.tf32 else 'strict fp32'}")
    print("Container tracking: two left corners with rim masks; "
          f"minimum template match score={ContainerMotionTracker.MIN_SCORE:g}")
    if args.ball_size_gate:
        print(f"Ball size gate: below {args.ball_size_gate:g}x the running median diameter")
    if args.enforce_ball_count:
        print(f"Ball count enforcement: keeping at most {expected_balls} balls per frame")
    if args.serial:
        print("Frame pipeline: serial (one frame at a time, single thread)")
    else:
        print(f"Frame pipeline: batch-size={args.batch_size}, prep-workers={args.prep_workers}")

    # Load the YOLOv8 model
    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        parser.error(f"Weights file not found: {weights_path}")
    print(f"Model weights: {weights_path}")
    model = YOLO(str(weights_path))

    # Initialize the SORT tracker
    mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=args.sort_iou)
    # Create the speed tracker
    speedtracker = Object_SpeedTracker()
    container_tracker = ContainerMotionTracker(calibration_mode="auto")
    # Optional lower size cutoff relative to this video's running median.
    # Removed boxes are rule-based exclusions; detection errors require review.
    ball_gate = BallSizeGate(args.ball_size_gate) if args.ball_size_gate else None
    # Opt-in count constraint: retain the highest-confidence candidates. The
    # diagnostic records allow review of removals without assuming they are false positives.
    enforcement = {"trimmed": 0, "frames": 0}

    # Load video. In the default parallel mode decoding and frame preparation run on
    # background threads while the GPU infers on the previous batch. --serial keeps
    # a single frame in flight instead: no threads and no batching.
    batcher = VideoFrameBatcher(
        video_path,
        max_frames=args.max_frames,
        batch_size=1 if args.serial else args.batch_size,
        prep_workers=0 if args.serial else args.prep_workers,
        threaded=not args.serial,
    )
    inference_kwargs = {"conf": args.conf, "iou": args.nms_iou, "verbose": False}
    if args.device is not None:
        inference_kwargs["device"] = args.device

    # Frame state shared with the tracking thread. It is only read after that thread joins.
    frame_index = 0
    last_timestamp = None
    frame_timestamps = {}
    container_positions = []
    frame_count_records = []
    ball_filter_events = []
    reported_overcounts = set()

    def process_frame(frame_number, timestamp, gray, detections):
        """Update every tracker for one frame and record its diagnostics."""
        nonlocal frame_index, last_timestamp
        if not np.isfinite(timestamp) or (last_timestamp is not None and timestamp <= last_timestamp):
            raise RuntimeError(f"Invalid video timestamp at frame {frame_number}; "
                               "cannot calculate velocity in px/s")
        last_timestamp = timestamp
        frame_index = frame_number
        frame_timestamps[frame_number] = timestamp
        # Detect balls and container positions
        balls_detection, container_xyxy, container_detection_count = split_detections(detections)
        raw_ball_detections = balls_detection
        if ball_gate is not None:
            balls_detection = ball_gate(balls_detection)
        size_gate_threshold = ball_gate.last_threshold if ball_gate is not None else None
        after_size_gate = balls_detection
        # Record the size-filtered count before applying the ball-count constraint.
        # Raw detections and removals from both rules are recorded separately below.
        detected_after_gate = len(balls_detection)
        if args.enforce_ball_count and detected_after_gate > expected_balls:
            enforcement["trimmed"] += detected_after_gate - expected_balls
            enforcement["frames"] += 1
            balls_detection = sorted(balls_detection, key=lambda box: box[4],
                                     reverse=True)[:expected_balls]
        event = make_ball_filter_event(
            video_name, frame_number, timestamp, raw_ball_detections,
            after_size_gate, balls_detection, expected_balls,
            args.ball_size_gate, size_gate_threshold, args.enforce_ball_count,
        )
        if event is not None:
            ball_filter_events.append(event)
        # Use SORT algorithm to track balls
        tracks = mot_tracker.update(dets=np.asarray(balls_detection, dtype=float).reshape(-1, 5))
        if expected_balls is not None:
            for stage, count in (("YOLO detections", len(balls_detection)), ("SORT tracks", len(tracks))):
                if count > expected_balls and stage not in reported_overcounts:
                    print(f"WARNING: first {stage} over-count at frame {frame_number}: "
                          f"{count} balls, expected {expected_balls}. "
                          "Complete frame counts will be reported after extraction.", flush=True)
                    reported_overcounts.add(stage)
        if frame_number == 1 or frame_number % 1000 == 0:
            print(f"Processed {frame_number} frames; latest YOLO balls={len(balls_detection)}, "
                  f"SORT tracks={len(tracks)}", flush=True)
        # YOLO only initializes/reacquires the container; retain reliable image positions.
        # The grayscale frame from the pipeline is reused instead of converting again.
        tracked_container = container_tracker.update(gray, container_xyxy, timestamp)
        container_position_valid = tracked_container is not None and container_tracker.last_status["sample_valid"]
        if container_position_valid:
            x1, y1, x2, y2 = tracked_container
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
            container_positions.append((frame_number, timestamp, center))
        frame_count_records.append({
            "frame": frame_number, "timestamp": timestamp,
            "raw_detected_balls": len(raw_ball_detections),
            "detected_balls": detected_after_gate, "tracked_balls": len(tracks),
            "size_gate_removed": len(raw_ball_detections) - detected_after_gate,
            "count_filter_removed": detected_after_gate - len(balls_detection),
            "size_gate_threshold_px": size_gate_threshold,
            "enforced_balls": len(balls_detection),
            "detected_containers": container_detection_count,
            "container_position_valid": bool(container_position_valid),
        })
        # Update ball tracking information
        for track in tracks:    # track: [x1, y1, x2, y2, track_id]
            track_id = int(track[4])
            speedtracker.update(track_id, track[:4], frame_number, timestamp)

    track_queue = queue.Queue(maxsize=2)
    track_errors = []
    tracker_thread = None
    if not args.serial:
        tracker_thread = threading.Thread(
            target=track_worker, args=(track_queue, track_errors, process_frame),
            name="velocity-tracking", daemon=True)
        tracker_thread.start()

    frame_stream = iter(batcher)
    try:
        for prepared in frame_stream:
            # Batched inference: one call for several frames instead of one call per frame
            # (a batch of one in serial mode).
            results = model([item[2] for item in prepared], **inference_kwargs)
            if track_errors:
                raise track_errors[0]
            # Tracking only needs grayscale, so drop the colour frames here.
            gray_frames = [(item[0], item[1], item[3]) for item in prepared]
            prepared = None
            if args.serial:
                for (frame_number, timestamp, gray), result in zip(gray_frames, results):
                    process_frame(frame_number, timestamp, gray,
                                  result.boxes.data.to("cpu").numpy())
            else:
                if not put_tracking_item(track_queue, tracker_thread, (gray_frames, results)):
                    if track_errors:
                        raise track_errors[0]
                    raise RuntimeError("Tracking thread exited before consuming all frames")
    finally:
        # Closing the stream stops the decoder thread even when the loop above raised.
        frame_stream.close()
        if tracker_thread is not None:
            put_tracking_item(track_queue, tracker_thread, None)
            tracker_thread.join()

    if track_errors:
        raise track_errors[0]

    print(f"Total frames: {frame_index}")
    if ball_gate is not None:
        print(f"Ball size gate: dropped {ball_gate.dropped} boxes, kept {ball_gate.accepted}")
    if args.enforce_ball_count:
        print(f"Ball count enforcement: trimmed {enforcement['trimmed']} boxes "
              f"on {enforcement['frames']} frames")
    # Fit four valid container positions around each ball's two-frame interval.
    # A window [n-2, n-1, n, n+1] produces container velocity at frame n.
    container_speeds, container_quality = fit_container_velocities(container_positions)
    speedtracker.speed_history["container"] = container_speeds
    print(f"Container quality: {container_quality}")

    # Observe the number of IDs and the number of frames per ID
    for obj_id in speedtracker.speed_history.keys():
        print(f"id: {obj_id}: {len(speedtracker.speed_history[obj_id])} frames")

    # Group velocities by integer frame number; timestamps remain physical seconds.
    common_frames = range(1, frame_index + 1)
    vx_vy_at_frame = {frame: {} for frame in common_frames}
    # Iterate over each object's speed_history and fill the dictionary
    for obj_id, obj_history in speedtracker.speed_history.items():
        for entry in obj_history:
            vx_vy, frame = entry
            vx_vy_at_frame[frame][obj_id] = vx_vy

    # Save speed tracking results to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame, obj_data in vx_vy_at_frame.items():
        for obj_id, (vx, vy) in obj_data.items():
            rows.append([frame, frame_timestamps[frame], obj_id, vx, vy])
    # Write CSV
    output_path = output_dir / f"vx_vy_data_{video_name}.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "obj_id", "vx", "vy"])  # seconds, px/s
        writer.writerows(rows)

    print(f"Speed data saved to {output_path}")
    report_frame_counts(
        frame_count_records, vx_vy_at_frame, expected_balls, video_name, output_dir,
        confidence=args.conf, nms_iou=args.nms_iou, expected_source=expected_source,
        filter_events=ball_filter_events, ball_size_gate=args.ball_size_gate,
        enforce_ball_count=args.enforce_ball_count,
    )


if __name__ == "__main__":
    main()
