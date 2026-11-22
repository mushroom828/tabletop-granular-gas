"""Locate the dark inner rim once, then track its motion between detections.

The YOLO rectangle is an approximate search region, not the measured rim.
The measured rectangle is kept rigid while patches near its left corners
track motion. Fixed masks retain the outer rim and exclude most of the inner
particle area. Measuring all four sides anew on every frame would add
edge-localization noise directly to the velocity estimate.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _Corner:
    template: np.ndarray
    origin: np.ndarray
    position: np.ndarray
    misses: int = 0
    mask: np.ndarray | None = None


class ContainerMotionTracker:
    """Estimate container translation from stable image patches near its rim corners."""

    MIN_SCORE = 0.85
    MAX_MISSES = 20

    def __init__(self, calibration_mode="auto"):
        if calibration_mode not in ("auto", "skip_rim"):
            raise ValueError("calibration_mode must be 'auto' or 'skip_rim'")
        self.calibration_mode = calibration_mode
        self.reset()

    @property
    def last_status(self):
        """Quality of the most recent update; callers cannot mutate tracker state."""
        return self._status.copy()

    def _set_status(self, source, state, valid, score=None):
        self._status = {
            "calibration_source": source,
            "tracking_state": state,
            "sample_valid": bool(valid),
            "min_match_score": None if score is None else float(score),
        }

    def reset(self):
        self._corners = {}
        self._reference_box = None
        self._frame_shape = None
        self._last_timestamp = None
        self._lost_frames = 0
        self._failed_frames = 0
        self.last_angle = None
        self._calibration_source = "none"
        self._set_status("none", "uninitialized", False)

    @staticmethod
    def _gray(frame):
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _subpixel_peak(scores, peak):
        x, y = peak
        offsets = []
        for before, center, after in (
            (scores[y, x - 1], scores[y, x], scores[y, x + 1])
            if 0 < x < scores.shape[1] - 1 else (0.0, 0.0, 0.0),
            (scores[y - 1, x], scores[y, x], scores[y + 1, x])
            if 0 < y < scores.shape[0] - 1 else (0.0, 0.0, 0.0),
        ):
            denominator = float(before - 2.0 * center + after)
            offset = 0.5 * float(before - after) / denominator if abs(denominator) > 1e-6 else 0.0
            offsets.append(float(np.clip(offset, -1.0, 1.0)))
        return np.array([x + offsets[0], y + offsets[1]], dtype=float)

    @staticmethod
    def _corner_rectangles(box):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        left = (x1 - 0.01 * width, x1 + 0.115 * width)
        top = (y1 - 0.025 * height, y1 + 0.155 * height)
        bottom = (y2 - 0.135 * height, y2 + 0.045 * height)
        return {
            "tl": (left[0], top[0], left[1], top[1]),
            "bl": (left[0], bottom[0], left[1], bottom[1]),
        }

    @staticmethod
    def _rim_mask(template, corner_name):
        """Keep the outer side of the dark rim, plus a narrow boundary band.

        The left edge of each patch lies outside the inner rim. Its bright
        connected component separates the metal rim from the particle area.
        Keep this mask fixed with its template until reacquisition. If the
        selected support is too small or flat, use the full textured patch.
        """
        threshold = float(np.clip(np.percentile(template, 10) + 10, 45, 85))
        _, labels = cv2.connectedComponents((template >= threshold).astype(np.uint8))
        row = 2 if corner_name == "tl" else template.shape[0] - 3
        label = int(labels[row, 2])
        if label == 0:
            candidates = labels[:, 0]
            candidates = candidates[candidates != 0]
            label = int(np.bincount(candidates).argmax()) if candidates.size else 0
        outside = ((labels == label).astype(np.uint8) if label
                   else np.ones(template.shape, dtype=np.uint8))
        mask = cv2.dilate(outside, np.ones((5, 5), dtype=np.uint8))
        if mask.sum() < 100 or float(template[mask > 0].std()) < 4.0:
            mask = np.ones(template.shape, dtype=np.uint8)
        return mask

    @staticmethod
    def _rim_coordinate(gray, box, side):
        """Find a dark, continuous inner-rim line near one YOLO box side.

        A median over the middle of each edge suppresses particles, highlights,
        and the rounded corners. Both sides of the trough must be brighter so
        that a uniformly dark wall is not mistaken for an edge.
        """
        x1, y1, x2, y2 = box
        box_width, box_height = x2 - x1, y2 - y1
        horizontal = side in ("top", "bottom")
        center = y1 if side == "top" else y2 if side == "bottom" else x1 if side == "left" else x2
        span = box_height if horizontal else box_width
        radius = int(np.clip(round(0.055 * span), 20, 50))
        limit = gray.shape[0] if horizontal else gray.shape[1]
        lo = max(0, int(round(center)) - radius)
        hi = min(limit, int(round(center)) + radius + 1)
        along_start = x1 if horizontal else y1
        along_span = box_width if horizontal else box_height
        first = max(0, int(round(along_start + 0.22 * along_span)))
        last = min(gray.shape[1] if horizontal else gray.shape[0],
                   int(round(along_start + 0.78 * along_span)))
        if hi - lo < 20 or last - first < 20:
            return None
        strip = gray[lo:hi, first:last] if horizontal else gray[first:last, lo:hi]
        profile = np.median(strip, axis=1 if horizontal else 0)
        # A light three-pixel smoothing avoids selecting one noisy pixel while
        # preserving the narrow black seam.
        profile = np.convolve(profile, (0.25, 0.5, 0.25), mode="same")
        # The convolution endpoints have incomplete support.
        candidates = profile[3:-3]
        peak = int(np.argmin(candidates)) + 3
        if peak < 8 or peak > len(profile) - 9:
            return None
        before = float(np.median(profile[max(0, peak - 18):peak - 7]))
        after = float(np.median(profile[peak + 8:min(len(profile), peak + 19)]))
        if min(before, after) - float(profile[peak]) < 12.0:
            return None
        # A painted or synthetic seam can have a flat, several-pixel trough.
        # Its center is the edge position, not the first minimum pixel.
        left = right = peak
        trough_limit = float(profile[peak]) + 0.5
        while left > 0 and profile[left - 1] <= trough_limit:
            left -= 1
        while right < len(profile) - 1 and profile[right + 1] <= trough_limit:
            right += 1
        return float(lo + (left + right) / 2.0)

    @classmethod
    def _align_to_inner_rim(cls, gray, detection):
        """Calibrate the four straight rim sides without changing frame motion."""
        aligned = detection.copy()
        sides = ("left", "top", "right", "bottom")
        measurements = [cls._rim_coordinate(gray, detection, side) for side in sides]
        if any(value is None for value in measurements):
            return detection
        aligned[:] = measurements
        width, height = detection[2] - detection[0], detection[3] - detection[1]
        if (aligned[2] - aligned[0] < 0.9 * width or
                aligned[3] - aligned[1] < 0.9 * height):
            return detection
        return aligned

    def _initialize(self, gray, box, timestamp):
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.all(np.isfinite(box)):
            self._set_status("none", "lost", False)
            return None
        if box[2] - box[0] < 80 or box[3] - box[1] < 60:
            self._set_status("none", "lost", False)
            return None
        if self.calibration_mode == "auto":
            aligned = self._align_to_inner_rim(gray, box)
            source = "inner_rim" if aligned is not box else "yolo"
            box = aligned
        else:
            source = "yolo"
        height, width = gray.shape[:2]
        corners = {}
        for name, rect in self._corner_rectangles(box).items():
            x1, y1, x2, y2 = (int(round(value)) for value in rect)
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            template = gray[y1:y2, x1:x2].copy()
            if float(template.std()) < 4.0:
                continue
            origin = np.array([x1, y1], dtype=float)
            corners[name] = _Corner(template, origin.copy(), origin.copy(),
                                    mask=self._rim_mask(template, name))
        if len(corners) < 2:
            self._set_status("none", "lost", False)
            return None
        self._corners = corners
        self._reference_box = box.copy()
        self._calibration_source = source
        self._frame_shape = gray.shape
        self._last_timestamp = timestamp
        self._lost_frames = 0
        self._failed_frames = 0
        self.last_angle = 0.0
        self._set_status(source, "tracking", True)
        return box.tolist()

    def _failed_sample(self, gray, detection, timestamp, match_score=None):
        """Count every unreliable geometry outcome, then reacquire from YOLO."""
        self._failed_frames += 1
        self.last_angle = None
        self._set_status(self._calibration_source, "lost", False, match_score)
        if self._failed_frames >= self.MAX_MISSES and detection is not None:
            self.reset()
            return self._initialize(gray, detection, timestamp)
        return None

    def _match(self, gray, corner, radius):
        template = corner.template
        width, height = template.shape[1], template.shape[0]
        px, py = np.rint(corner.position).astype(int)
        x0, y0 = max(0, px - radius), max(0, py - radius)
        x1 = min(gray.shape[1], px + width + radius)
        y1 = min(gray.shape[0], py + height + radius)
        if x1 - x0 < width or y1 - y0 < height:
            return None, None
        scores = cv2.matchTemplate(gray[y0:y1, x0:x1], template,
                                   cv2.TM_CCOEFF_NORMED, mask=corner.mask)
        # Masked correlation can be undefined for a flat candidate region.
        # Such candidates must fail the score gate, never become a peak.
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, peak = cv2.minMaxLoc(scores)
        if score < self.MIN_SCORE:
            return None, score
        return np.array([x0, y0], dtype=float) + self._subpixel_peak(scores, peak), score

    def update(self, frame, detection, timestamp=None):
        """Return a fixed-size container rectangle at the rigid center, or ``None`` if lost."""
        gray = self._gray(frame)
        if self._reference_box is None:
            if detection is None:
                self._set_status("none", "uninitialized", False)
                return None
            return self._initialize(gray, detection, timestamp)
        if gray.shape != self._frame_shape or (
            timestamp is not None and self._last_timestamp is not None
            and timestamp <= self._last_timestamp
        ):
            self.reset()
            if detection is None:
                self._set_status("none", "lost", False)
                return None
            return self._initialize(gray, detection, timestamp)
        self._last_timestamp = timestamp

        box_width = self._reference_box[2] - self._reference_box[0]
        radius = max(6, int(round(12.0 * box_width / 1100.0)))
        matches = {}
        scores = []
        for name, corner in self._corners.items():
            search_radius = min(radius + corner.misses * radius, radius * 5)
            position, score = self._match(gray, corner, search_radius)
            if score is not None:
                scores.append(score)
            if position is not None:
                matches[name] = position
        min_score = min(scores) if scores else None
        if len(matches) == 1:
            # Keep the occluded corner's search window near the moving container.
            name, position = next(iter(matches.items()))
            shift = position - self._corners[name].position
            self._corners[name].position = position
            self._corners[name].misses = 0
            other = "bl" if name == "tl" else "tl"
            self._corners[other].position += shift
            self._corners[other].misses += 1
            self._lost_frames = 0
            return self._failed_sample(gray, detection, timestamp, min_score)
        if not matches:
            for corner in self._corners.values():
                corner.misses += 1
            self._lost_frames += 1
            return self._failed_sample(gray, detection, timestamp, min_score)

        top = matches["tl"]
        bottom = matches["bl"]
        ref_top = self._corners["tl"].origin
        ref_bottom = self._corners["bl"].origin
        reference_edge = ref_bottom - ref_top
        current_edge = bottom - top
        edge_length = float(np.linalg.norm(reference_edge))
        length_error = abs(float(np.linalg.norm(current_edge)) - edge_length)
        angle = float(np.arctan2(current_edge[1], current_edge[0])
                      - np.arctan2(reference_edge[1], reference_edge[0]))
        angle = float(np.arctan2(np.sin(angle), np.cos(angle)))
        if length_error > max(4.0, 0.015 * edge_length) or abs(angle) > np.deg2rad(3.0):
            return self._failed_sample(gray, detection, timestamp, min_score)

        for name, position in matches.items():
            self._corners[name].position = position
            self._corners[name].misses = 0
        self._lost_frames = 0
        self._failed_frames = 0
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        reference_center = np.array([
            (self._reference_box[0] + self._reference_box[2]) / 2.0,
            (self._reference_box[1] + self._reference_box[3]) / 2.0,
        ])
        reference_left_center = (ref_top + ref_bottom) / 2.0
        current_left_center = (top + bottom) / 2.0
        current_center = current_left_center + rotation @ (reference_center - reference_left_center)
        shift = current_center - reference_center
        refined = self._reference_box + np.array([shift[0], shift[1], shift[0], shift[1]])
        self.last_angle = angle
        self._set_status(self._calibration_source, "tracking", True, min_score)
        return refined.tolist()
