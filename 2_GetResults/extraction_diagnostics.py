"""Report per-frame ball counts without filtering detections or velocities."""

from collections import Counter
import csv
import json
from pathlib import Path
import re


def infer_expected_balls(video_stem):
    """Read the physical ball count from a name such as ``120r_10``."""
    match = re.fullmatch(r"\d+r_(\d+)", video_stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _frame_ranges(frames):
    ranges = []
    for frame in frames:
        if ranges and frame == ranges[-1][1] + 1:
            ranges[-1][1] = frame
        else:
            ranges.append([frame, frame])
    return ranges


def _count_summary(rows, field, expected_balls, first_frame=1):
    eligible = [row for row in rows if row["frame"] >= first_frame]
    histogram = Counter(row[field] for row in eligible)
    result = {
        "evaluated_frames": len(eligible),
        "first_eligible_frame": first_frame,
        "histogram": dict(sorted(histogram.items())),
        "maximum_count": max(histogram, default=0),
    }
    if expected_balls is not None:
        over = [row["frame"] for row in eligible if row[field] > expected_balls]
        result.update({
            "below_expected_frames": sum(row[field] < expected_balls for row in eligible),
            "equal_expected_frames": histogram[expected_balls],
            "above_expected_frames": len(over),
            "above_expected_percent": 100.0 * len(over) / len(eligible) if eligible else 0.0,
            "excess_count_total": sum(max(0, row[field] - expected_balls) for row in eligible),
            "above_expected_frame_ranges": _frame_ranges(over),
        })
    return result


def make_ball_filter_event(video_name, frame, timestamp, raw_detections,
                           after_size_gate, kept_detections, expected_balls,
                           size_gate_ratio, size_gate_threshold_px,
                           enforce_ball_count):
    """Describe a frame with actual removals; never change boxes or their order.

    Filtering retains the original box objects, so object identity also handles
    duplicate boxes and confidence ties. Indices are one-based within this
    frame's post-confidence/NMS ball detections, before SORT assigns track IDs.
    """
    if len(raw_detections) == len(after_size_gate) == len(kept_detections):
        return None
    size_kept = {id(box) for box in after_size_gate}
    final_kept = {id(box) for box in kept_detections}
    records = []
    for index, box in enumerate(raw_detections, 1):
        x1, y1, x2, y2, confidence = map(float, box)
        width, height = x2 - x1, y2 - y1
        reason = ("size_gate" if id(box) not in size_kept
                  else "ball_count" if id(box) not in final_kept else None)
        records.append({
            "detection_index": index, "xyxy_px": [x1, y1, x2, y2],
            "width_px": width, "height_px": height,
            "diameter_px": (max(width, 0.0) * max(height, 0.0)) ** 0.5,
            "confidence": confidence, "kept": reason is None,
            "removal_reason": reason,
        })
    def scores(reason):
        return sorted((row["confidence"] for row in records
                       if row["removal_reason"] == reason), reverse=True)
    kept_scores = scores(None)
    size_removed = scores("size_gate")
    count_removed = scores("ball_count")
    min_kept = min(kept_scores) if kept_scores else None
    max_removed = max(count_removed) if count_removed else None
    return {
        "schema_version": 1, "video": video_name, "frame": int(frame),
        "timestamp": float(timestamp), "expected_balls": expected_balls,
        "raw_detected_balls": len(raw_detections),
        "after_size_gate": len(after_size_gate),
        "sent_to_tracker": len(kept_detections),
        "size_gate": {"enabled": bool(size_gate_ratio), "ratio": size_gate_ratio,
                      "threshold_px": size_gate_threshold_px},
        "count_enforcement_enabled": bool(enforce_ball_count),
        "all_confidences_desc": sorted((row["confidence"] for row in records), reverse=True),
        "count_candidates_confidences_desc": sorted((float(box[4]) for box in after_size_gate), reverse=True),
        "kept_confidences_desc": kept_scores,
        "size_removed_confidences_desc": size_removed,
        "count_removed_confidences_desc": count_removed,
        "min_kept_confidence": min_kept,
        "min_count_removed_confidence": min(count_removed) if count_removed else None,
        "max_count_removed_confidence": max_removed,
        "count_confidence_gap": (min_kept - max_removed
                                 if min_kept is not None and max_removed is not None else None),
        "detections": records,
    }


def report_frame_counts(frame_records, velocities_by_frame, expected_balls,
                        video_name, output_dir, confidence, nms_iou,
                        expected_source="unknown", *, filter_events=None,
                        ball_size_gate=0.0, enforce_ball_count=False):
    """Print histograms and save complete count diagnostics beside the velocity CSV.

    Ball counts always exclude the container. The velocity histogram starts at
    frame 2 because a two-frame velocity cannot exist at the first frame.
    These diagnostics never remove a record or force the number of tracks.
    """
    rows = []
    for record in frame_records:
        row = dict(record)
        row.setdefault("enforced_balls", row["detected_balls"])
        row.setdefault("raw_detected_balls", row["detected_balls"])
        row.setdefault("size_gate_removed", 0)
        row.setdefault("count_filter_removed", row["detected_balls"] - row["enforced_balls"])
        row.setdefault("size_gate_threshold_px", None)
        velocities = velocities_by_frame.get(row["frame"], {})
        row["velocity_balls"] = sum(obj_id != "container" for obj_id in velocities)
        row["container_velocity_valid"] = "container" in velocities
        row["expected_balls"] = expected_balls if expected_balls is not None else ""
        for field, flag in (("detected_balls", "detection_over_expected"),
                            ("tracked_balls", "tracking_over_expected"),
                            ("velocity_balls", "velocity_over_expected")):
            row[flag] = row[field] > expected_balls if expected_balls is not None else ""
        rows.append(row)

    stages = {
        "detected_balls": _count_summary(rows, "detected_balls", expected_balls),
        "tracked_balls": _count_summary(rows, "tracked_balls", expected_balls),
        "velocity_balls": _count_summary(rows, "velocity_balls", expected_balls, first_frame=2),
    }
    container_histogram = dict(sorted(Counter(row["detected_containers"] for row in rows).items()))
    filter_events = [] if filter_events is None else filter_events
    filtering = {
        "unit": "detection_boxes_per_frame",
        "order": ["size_gate", "ball_count"],
        "input_detections": sum(row["raw_detected_balls"] for row in rows),
        "after_size_gate": sum(row["detected_balls"] for row in rows),
        "sent_to_tracker": sum(row["enforced_balls"] for row in rows),
        "event_frames": len(filter_events),
        "size_gate": {"enabled": bool(ball_size_gate), "ratio": ball_size_gate,
                      "removed_boxes": sum(row["size_gate_removed"] for row in rows),
                      "affected_frames": sum(row["size_gate_removed"] > 0 for row in rows)},
        "ball_count": {"enabled": bool(enforce_ball_count), "expected_balls": expected_balls,
                       "removed_boxes": sum(row["count_filter_removed"] for row in rows),
                       "affected_frames": sum(row["count_filter_removed"] > 0 for row in rows)},
    }
    summary = {
        "video": video_name,
        "processed_frames": len(rows),
        "expected_balls": expected_balls,
        "expected_balls_source": expected_source,
        "inference": {"conf": confidence, "nms_iou": nms_iou},
        "ball_counts": stages,
        "ball_filtering": filtering,
        "container_detection_histogram": container_histogram,
        "reliable_container_positions": sum(row["container_position_valid"] for row in rows),
        "container_velocity_records": sum(row["container_velocity_valid"] for row in rows),
        "ball_filter_events": filter_events,
    }

    print("\n=== Per-frame count diagnostics (container excluded from ball counts) ===")
    print(f"Video: {video_name}; processed frames: {len(rows)}")
    print(f"YOLO settings: conf={confidence:g}, NMS IoU={nms_iou:g}")
    if expected_balls is None:
        print("Expected ball count: unknown; use --expected-balls to enable count comparisons.")
    else:
        print(f"Expected balls: {expected_balls} ({expected_source})")
    print("Ball velocities are counted on frames 2 onward; frame 1 has no two-frame velocity.")
    print(f"{'Balls':>7} {'YOLO frames':>14} {'SORT frames':>14} {'Velocity frames':>18}")
    observed = set().union(*(stage["histogram"] for stage in stages.values()))
    if expected_balls is not None:
        observed.add(expected_balls)
    for count in sorted(observed):
        values = [stage["histogram"].get(count, 0) for stage in stages.values()]
        print(f"{count:>7} {values[0]:>14} {values[1]:>14} {values[2]:>18}")

    labels = {"detected_balls": "YOLO detections", "tracked_balls": "SORT tracks",
              "velocity_balls": "Ball velocity records"}
    if expected_balls is not None:
        print("\nComparison with expected ball count:")
        for field, stage in stages.items():
            print(f"  {labels[field]}: below={stage['below_expected_frames']}, "
                  f"equal={stage['equal_expected_frames']}, above={stage['above_expected_frames']}, "
                  f"max={stage['maximum_count']}")
            if stage["above_expected_frames"]:
                ranges = stage["above_expected_frame_ranges"]
                preview = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges[:12])
                if len(ranges) > 12:
                    preview += ", ... (complete ranges in the diagnostics JSON)"
                print(f"  WARNING: {labels[field]} exceed {expected_balls} on "
                      f"{stage['above_expected_frames']} frames "
                      f"({stage['above_expected_percent']:.3f}%); "
                      f"total excess count={stage['excess_count_total']}.")
                print(f"    Frames: {preview}")

    print(f"Container detections per frame (count: frames): {container_histogram}")
    print(f"Reliable container positions: {summary['reliable_container_positions']}; "
          f"container velocity records: {summary['container_velocity_records']}")
    print("Ball filter totals (detection boxes): "
          f"size gate={filtering['size_gate']['removed_boxes']} on "
          f"{filtering['size_gate']['affected_frames']} frames; "
          f"ball count={filtering['ball_count']['removed_boxes']} on "
          f"{filtering['ball_count']['affected_frames']} frames")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"frame_counts_{video_name}.csv"
    fields = ["frame", "timestamp", "raw_detected_balls", "detected_balls",
              "size_gate_removed", "count_filter_removed", "size_gate_threshold_px",
              "enforced_balls", "tracked_balls",
              "velocity_balls",
              "detected_containers", "container_position_valid", "container_velocity_valid",
              "expected_balls", "detection_over_expected", "tracking_over_expected",
              "velocity_over_expected"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / f"frame_counts_{video_name}.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Per-frame counts saved to {csv_path}")
    print(f"Count summary saved to {json_path}")
    print(f"Ball filter details: {len(filter_events)} event frames in ball_filter_events")
    return summary
