"""
Cricket ball tracker.

Uses:
    - YOLO detections from BallDetector
    - Kalman filter for position/velocity prediction
    - Distance gating to reject unrelated detections
    - Short-term missed-detection tolerance
    - Delivery trajectory collection

This module intentionally does NOT depend on trajectory.py or utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List


Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Kalman Ball Tracker
# State:
#   x, y, vx, vy
#
# Measurement:
#   x, y
# ---------------------------------------------------------------------------

class KalmanBallTracker:
    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
    ):
        self.x: Optional[float] = None
        self.y: Optional[float] = None

        self.vx: float = 0.0
        self.vy: float = 0.0

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        self.initialized = False

    def initialize(self, point: Point) -> None:
        self.x = float(point[0])
        self.y = float(point[1])
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = True

    def predict(self, dt: float = 1.0) -> Point:
        if not self.initialized:
            raise RuntimeError("Kalman tracker is not initialized.")

        self.x += self.vx * dt
        self.y += self.vy * dt

        return self.x, self.y

    def update(self, measurement: Point, dt: float = 1.0) -> Point:
        measurement_x = float(measurement[0])
        measurement_y = float(measurement[1])

        if not self.initialized:
            self.initialize(measurement)
            return self.x, self.y

        # Prediction.
        predicted_x = self.x + self.vx * dt
        predicted_y = self.y + self.vy * dt

        # Simple adaptive measurement correction.
        alpha = 0.65

        corrected_x = (
            alpha * measurement_x
            + (1.0 - alpha) * predicted_x
        )

        corrected_y = (
            alpha * measurement_y
            + (1.0 - alpha) * predicted_y
        )

        # Estimate velocity from corrected position.
        new_vx = (corrected_x - self.x) / max(dt, 1e-6)
        new_vy = (corrected_y - self.y) / max(dt, 1e-6)

        velocity_alpha = 0.55

        self.vx = (
            velocity_alpha * new_vx
            + (1.0 - velocity_alpha) * self.vx
        )

        self.vy = (
            velocity_alpha * new_vy
            + (1.0 - velocity_alpha) * self.vy
        )

        self.x = corrected_x
        self.y = corrected_y

        return self.x, self.y

    def get_position(self) -> Optional[Point]:
        if not self.initialized:
            return None

        return self.x, self.y

    def reset(self) -> None:
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False


# ---------------------------------------------------------------------------
# Ball Tracker
# ---------------------------------------------------------------------------

class BallTracker:
    """
    Tracks one cricket ball through a delivery clip.

    Expected detector output:

        (x, y)

    or:

        None

    The tracker keeps:
        - current position
        - trajectory
        - frame numbers
        - number of detections
        - missed-frame count
    """

    MAX_TRAIL = 120
    MAX_MISSED = 8
    GATE_RADIUS = 120.0

    def __init__(
        self,
        max_trail: int = MAX_TRAIL,
        max_missed: int = MAX_MISSED,
        gate_radius: float = GATE_RADIUS,
    ):
        self.max_trail = int(max_trail)
        self.max_missed = int(max_missed)
        self.gate_radius = float(gate_radius)

        self.kalman = KalmanBallTracker()

        self.trajectory: List[Point] = []
        self.frame_numbers: List[int] = []

        self.current_position: Optional[Point] = None

        self.missed_frames = 0
        self.total_detections = 0

        self.in_delivery = False

        # Complete delivery trajectory.
        self.delivery_positions: List[Tuple[Point, int]] = []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        dx = float(p1[0]) - float(p2[0])
        dy = float(p1[1]) - float(p2[1])

        return (dx * dx + dy * dy) ** 0.5

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        detection: Optional[Point],
        frame_no: int,
    ) -> Optional[Point]:

        frame_no = int(frame_no)

        # --------------------------------------------------------------
        # No detection
        # --------------------------------------------------------------

        if detection is None:

            if self.kalman.initialized:
                predicted = self.kalman.predict()

                self.missed_frames += 1

                if self.missed_frames <= self.max_missed:
                    self.current_position = predicted

                    self._append_trajectory(
                        predicted,
                        frame_no,
                    )

                    return predicted

            self.current_position = None

            return None

        # --------------------------------------------------------------
        # Convert detection
        # --------------------------------------------------------------

        detection_point: Point = (
            float(detection[0]),
            float(detection[1]),
        )

        # --------------------------------------------------------------
        # First detection
        # --------------------------------------------------------------

        if not self.kalman.initialized:

            self.kalman.initialize(detection_point)

            self.current_position = detection_point

            self.missed_frames = 0
            self.total_detections += 1
            self.in_delivery = True

            self._append_trajectory(
                detection_point,
                frame_no,
            )

            return detection_point

        # --------------------------------------------------------------
        # Predict
        # --------------------------------------------------------------

        predicted = self.kalman.predict()

        distance = self._distance(
            detection_point,
            predicted,
        )

        # --------------------------------------------------------------
        # Gating
        #
        # Reject a detection that is too far from the predicted ball
        # position.
        # --------------------------------------------------------------

        if distance > self.gate_radius:

            self.missed_frames += 1

            if self.missed_frames <= self.max_missed:

                self.current_position = predicted

                self._append_trajectory(
                    predicted,
                    frame_no,
                )

                return predicted

            # Too many misses.
            # Reinitialize from the new detection.
            self.kalman.initialize(detection_point)

        else:

            self.kalman.update(
                detection_point,
            )

        # --------------------------------------------------------------
        # Accepted detection
        # --------------------------------------------------------------

        self.current_position = self.kalman.get_position()

        self.missed_frames = 0
        self.total_detections += 1
        self.in_delivery = True

        if self.current_position is not None:
            self._append_trajectory(
                self.current_position,
                frame_no,
            )

        return self.current_position

    # ------------------------------------------------------------------
    # Store trajectory
    # ------------------------------------------------------------------

    def _append_trajectory(
        self,
        point: Point,
        frame_no: int,
    ) -> None:

        self.trajectory.append(
            (
                float(point[0]),
                float(point[1]),
            )
        )

        self.frame_numbers.append(
            int(frame_no)
        )

        self.delivery_positions.append(
            (
                (
                    float(point[0]),
                    float(point[1]),
                ),
                int(frame_no),
            )
        )

        # Keep recent trajectory bounded.
        if len(self.trajectory) > self.max_trail:
            self.trajectory.pop(0)

        if len(self.frame_numbers) > self.max_trail:
            self.frame_numbers.pop(0)

    # ------------------------------------------------------------------
    # Reset delivery
    # ------------------------------------------------------------------

    def reset_delivery(self) -> None:

        self.kalman.reset()

        self.trajectory.clear()
        self.frame_numbers.clear()
        self.delivery_positions.clear()

        self.current_position = None

        self.missed_frames = 0
        self.total_detections = 0

        self.in_delivery = False

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_trajectory(self) -> List[Point]:
        return list(self.trajectory)

    def get_delivery_positions(self) -> List[Tuple[Point, int]]:
        return list(self.delivery_positions)

    def get_current_position(self) -> Optional[Point]:
        return self.current_position

    def get_detection_count(self) -> int:
        return self.total_detections

    def is_tracking(self) -> bool:
        return (
            self.kalman.initialized
            and self.missed_frames <= self.max_missed
        )