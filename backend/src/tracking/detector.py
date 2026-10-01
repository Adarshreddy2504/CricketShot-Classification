"""
Cricket ball detector.

Loads:
    backend/models/best.pt

Returns:
    (x, y) ball centre in ORIGINAL frame coordinates.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt",
)

CONF_THRESHOLD = 0.30

RESIZE_W = 640
RESIZE_H = 360

MIN_DIAG_PX = 4
MAX_DIAG_PX = 80


class BallDetector:
    def __init__(self):
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Ball detection model not found:\n{MODEL_PATH}"
            )

        self.device = 0 if torch.cuda.is_available() else "cpu"

        print(
            f"[BALL] Loading YOLO model: {MODEL_PATH}",
            flush=True,
        )
        print(
            f"[BALL] Device: {self.device}",
            flush=True,
        )

        self.model = YOLO(MODEL_PATH)

        print(
            "[BALL] YOLO model loaded",
            flush=True,
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[int, int] | None:

        if frame is None or frame.size == 0:
            return None

        h_full, w_full = frame.shape[:2]

        small = cv2.resize(
            frame,
            (RESIZE_W, RESIZE_H),
            interpolation=cv2.INTER_LINEAR,
        )

        results = self.model.predict(
            source=small,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=self.device,
        )

        if not results:
            return None

        result = results[0]

        if result.boxes is None:
            return None

        sx = w_full / RESIZE_W
        sy = h_full / RESIZE_H

        best_center = None
        best_conf = -1.0

        for box in result.boxes.data.tolist():

            if len(box) < 6:
                continue

            x1, y1, x2, y2, score, cls = box

            # Ball class.
            if int(cls) != 0:
                continue

            diagonal = (
                (x2 - x1) ** 2
                +
                (y2 - y1) ** 2
            ) ** 0.5

            if not (
                MIN_DIAG_PX
                <= diagonal
                <= MAX_DIAG_PX
            ):
                continue

            score = float(score)

            if score <= best_conf:
                continue

            best_conf = score

            best_center = (
                int(((x1 + x2) / 2) * sx),
                int(((y1 + y2) / 2) * sy),
            )

        return best_center