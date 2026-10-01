"""
Trajectory analysis and cricket delivery classification.
"""

from __future__ import annotations

import math

import numpy as np

from .utils import (
    CREASE_WIDTH_M,
    PITCH_LENGTH_M,
    compute_angle_deg,
    pixel_to_ground,
    savgol_smooth,
)


MAX_EARLY_POSITIONS = 12

SPEED_MIN_KMH = 40.0
SPEED_MAX_KMH = 200.0


def compute_release_speed(
    early_positions: list,
    fps: float,
    H: np.ndarray,
    debug: bool = False,
) -> float | None:

    if fps <= 0 or len(early_positions) < 3:
        return None

    ts = []
    gxs = []
    gys = []

    for (px, py), frame_no in early_positions:

        gx, gy = pixel_to_ground(
            (px, py),
            H,
        )

        ts.append(frame_no / fps)
        gxs.append(gx)
        gys.append(gy)

    ts = np.asarray(ts, dtype=np.float64)
    gxs = np.asarray(gxs, dtype=np.float64)
    gys = np.asarray(gys, dtype=np.float64)

    keep = np.concatenate(
        ([True], np.diff(ts) > 1e-6)
    )

    ts = ts[keep]
    gxs = gxs[keep]
    gys = gys[keep]

    if len(ts) < 3:
        return None

    t_rel = ts - ts[0]

    def fit_slope(
        t: np.ndarray,
        values: np.ndarray,
    ) -> float:

        A = np.vstack(
            [t, np.ones_like(t)]
        ).T

        slope, intercept = np.linalg.lstsq(
            A,
            values,
            rcond=None,
        )[0]

        residuals = (
            values
            - (
                slope * t
                + intercept
            )
        )

        std = residuals.std()

        if std > 1e-9:

            mask = (
                np.abs(residuals)
                < 2.5 * std
            )

            if (
                2 <= mask.sum()
                < len(t)
            ):

                slope, intercept = (
                    np.linalg.lstsq(
                        A[mask],
                        values[mask],
                        rcond=None,
                    )[0]
                )

        return float(slope)

    vx = fit_slope(t_rel, gxs)
    vy = fit_slope(t_rel, gys)

    speed_mps = math.hypot(vx, vy)

    # Keep the calibration already present in the supplied tracker code.
    kmph = speed_mps * 3.6 * 2.0 - 20.0

    if debug:
        print(
            f"[SPEED] "
            f"vx={vx:.2f}, "
            f"vy={vy:.2f}, "
            f"speed={kmph:.1f} km/h"
        )

    if not (
        SPEED_MIN_KMH
        < kmph
        < SPEED_MAX_KMH
    ):
        return None

    return round(kmph, 1)


def detect_bounce(
    traj_list: list,
    H: np.ndarray,
) -> int | None:

    n = len(traj_list)

    if n < 9:
        return None

    py_raw = [
        pt[1]
        for pt in traj_list
    ]

    gx_raw = []
    gy_raw = []

    for pt in traj_list:

        gx, gy = pixel_to_ground(
            pt,
            H,
        )

        gx_raw.append(gx)
        gy_raw.append(gy)

    gy_sm = savgol_smooth(
        gy_raw,
        window=7,
        poly=2,
    )

    py_sm = savgol_smooth(
        py_raw,
        window=5,
        poly=2,
    )

    vy_gnd = [
        gy_sm[i + 1]
        - gy_sm[i]
        for i in range(
            len(gy_sm) - 1
        )
    ]

    best_idx = None
    best_score = -1

    for i in range(3, n - 3):

        score = 0

        # Image-space local maximum.
        if (
            py_raw[i] >= py_raw[i - 1]
            and py_raw[i] >= py_raw[i - 2]
            and py_raw[i] >= py_raw[i + 1]
            and py_raw[i] >= py_raw[i + 2]
        ):
            score += 2

        # Ground-space local minimum.
        if (
            gy_sm[i] <= gy_sm[i - 1]
            and gy_sm[i] <= gy_sm[i - 2]
            and gy_sm[i] <= gy_sm[i + 1]
            and gy_sm[i] <= gy_sm[i + 2]
        ):
            score += 1

        # Vertical velocity transition.
        if i < len(vy_gnd):

            if (
                vy_gnd[i - 1] < -0.01
                and vy_gnd[i] > -0.005
            ):
                score += 2

        # Horizontal movement change.
        if (
            i >= 2
            and i + 2 < len(gx_raw)
        ):

            before = abs(
                gx_raw[i]
                - gx_raw[i - 2]
            )

            after = abs(
                gx_raw[i + 2]
                - gx_raw[i]
            )

            if after < before * 1.30:
                score += 1

        # Parabolic check.
        window = 4

        lo = max(
            0,
            i - window,
        )

        hi = min(
            n,
            i + window + 1,
        )

        if hi - lo >= 5:

            xs = list(
                range(hi - lo)
            )

            ys = py_sm[lo:hi]

            try:

                coeffs = np.polyfit(
                    xs,
                    ys,
                    2,
                )

                if coeffs[0] < 0:
                    score += 1

            except Exception:
                pass

        if (
            score >= 3
            and score > best_score
        ):

            best_score = score
            best_idx = i

    return best_idx


def detect_release_frame(
    delivery_positions: list,
) -> int:

    if len(delivery_positions) < 3:
        return 0

    quarter = max(
        1,
        len(delivery_positions) // 4,
    )

    best_i = 0
    best_distance = 0.0

    for i in range(1, quarter):

        (x1, y1), _ = (
            delivery_positions[i - 1]
        )

        (x2, y2), _ = (
            delivery_positions[i]
        )

        distance = math.hypot(
            x2 - x1,
            y2 - y1,
        )

        if distance > best_distance:

            best_distance = distance
            best_i = i - 1

    return best_i


def estimate_swing(
    traj_list: list,
    bounce_idx: int | None,
    H: np.ndarray,
) -> str:

    if (
        bounce_idx is None
        or bounce_idx < 2
        or bounce_idx >= len(traj_list) - 2
    ):
        return "Straight"

    gx0, _ = pixel_to_ground(
        traj_list[0],
        H,
    )

    gxb, _ = pixel_to_ground(
        traj_list[bounce_idx],
        H,
    )

    drift = gxb - gx0

    angle = round(
        abs(drift) / 0.15,
        1,
    )

    if angle < 0.4:
        return "Straight"

    if drift > 0:
        return f"{angle}° outswing"

    return f"{angle}° inswing"


def compute_release_angle(
    delivery_positions: list,
) -> float | None:

    if len(delivery_positions) < 4:
        return None

    (x1, y1), _ = delivery_positions[0]
    (x2, y2), _ = delivery_positions[3]

    angle = -compute_angle_deg(
        (x1, y1),
        (x2, y2),
    )

    return round(angle, 1)


def compute_bounce_angle(
    traj_list: list,
    bounce_idx: int | None,
    H: np.ndarray,
) -> float | None:

    if (
        bounce_idx is None
        or bounce_idx < 2
        or bounce_idx + 2 >= len(traj_list)
    ):
        return None

    x1, y1 = pixel_to_ground(
        traj_list[bounce_idx - 2],
        H,
    )

    xb, yb = pixel_to_ground(
        traj_list[bounce_idx],
        H,
    )

    x2, y2 = pixel_to_ground(
        traj_list[bounce_idx + 2],
        H,
    )

    vx = xb - x1
    vy = yb - y1

    magnitude = math.hypot(
        vx,
        vy,
    ) + 1e-9

    angle = math.degrees(
        math.asin(
            min(
                1.0,
                abs(vy) / magnitude,
            )
        )
    )

    return round(angle, 1)


LENGTH_ZONES = [
    ("Beamer", 0.00, 0.30),
    ("Bouncer", 0.30, 0.45),
    ("Short", 0.45, 0.58),
    ("Good Length", 0.58, 0.72),
    ("Full", 0.72, 0.83),
    ("Yorker", 0.83, 1.00),
]


LINE_ZONES = [
    ("Wide Leg", 0.00, 0.18),
    ("Leg Side", 0.18, 0.38),
    ("Middle", 0.38, 0.62),
    ("Off Side", 0.62, 0.82),
    ("Wide Off", 0.82, 1.00),
]


def classify_length(
    bounce_y: float,
    frame_height: int,
) -> str:

    if frame_height <= 0:
        return "Unknown"

    fraction = bounce_y / frame_height

    for label, low, high in LENGTH_ZONES:

        if low <= fraction < high:
            return label

    return "Full"


def classify_line(
    bounce_x: float,
    frame_width: int,
) -> str:

    if frame_width <= 0:
        return "Unknown"

    fraction = bounce_x / frame_width

    for label, low, high in LINE_ZONES:

        if low <= fraction < high:
            return label

    return "Middle"


def predict_trajectory(
    current_pos: tuple,
    velocity_px: tuple,
    n_steps: int = 30,
    gravity_px: float = 0.5,
) -> list:

    x = float(current_pos[0])
    y = float(current_pos[1])

    vx = float(velocity_px[0])
    vy = float(velocity_px[1])

    points = []

    for _ in range(n_steps):

        x += vx
        y += vy

        vy += gravity_px

        points.append(
            (
                int(x),
                int(y),
            )
        )

    return points