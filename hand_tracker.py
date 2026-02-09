"""
hand_tracker.py — Hand Detection Module
========================================
This module wraps MediaPipe's new Task API (v0.10.14+) to detect
21 hand landmarks from a webcam frame. It returns pixel-coordinate
landmarks and draws the hand skeleton overlay using plain OpenCV
(no dependency on mediapipe.drawing_utils — more portable & educational).

NOTE: MediaPipe >= 0.10.14 removed the legacy `mp.solutions` API and
replaced it with `mediapipe.tasks`.  This file uses the new API.

MediaPipe Hands landmark map (21 points per hand):
    0  = WRIST
    1  = THUMB_CMC,   2 = THUMB_MCP,  3 = THUMB_IP,   4 = THUMB_TIP
    5  = INDEX_MCP,   6 = INDEX_PIP,  7 = INDEX_DIP,  8 = INDEX_TIP
    9  = MIDDLE_MCP, 10 = MIDDLE_PIP,11 = MIDDLE_DIP,12 = MIDDLE_TIP
    13 = RING_MCP,   14 = RING_PIP,  15 = RING_DIP,  16 = RING_TIP
    17 = PINKY_MCP,  18 = PINKY_PIP, 19 = PINKY_DIP, 20 = PINKY_TIP

Each landmark has normalized (x, y, z) in [0..1] relative to the image.
We convert these to pixel coordinates so the rest of the pipeline can
work in pixel space.

First-time setup:
    The model file "hand_landmarker.task" (~7.5 MB) is downloaded
    automatically on the first run and cached in the project folder.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

# Google-hosted URL for the float16 hand-landmarker model.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
# We store the model next to this file for easy access.
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")


def _ensure_model():
    """Download the hand-landmarker model if it isn't already present."""
    if not os.path.exists(_MODEL_PATH):
        print(f"[hand_tracker] Downloading model -> {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        size_mb = os.path.getsize(_MODEL_PATH) / 1024 / 1024
        print(f"[hand_tracker] Done ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Hand skeleton connections (bone segments between landmark indices).
# We hard-code these so we don't depend on any drawing helper library.
#
# Each tuple (a, b) means "draw a line from landmark a to landmark b".
# ---------------------------------------------------------------------------
HAND_BONE_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky  +  palm base connections
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class HandTracker:
    """
    Detects a single hand and returns its 21 landmarks in pixel coords.

    Usage:
        tracker = HandTracker()
        landmarks, raw_result = tracker.find_hand_landmarks(frame)
        frame = tracker.draw_landmarks(frame, landmarks)
    """

    def __init__(self, max_hands=1, detection_confidence=0.7,
                 tracking_confidence=0.7):
        """
        Parameters
        ----------
        max_hands : int
            How many hands to detect (we only need 1 for this project).
        detection_confidence : float
            Minimum confidence [0..1] for the initial hand detection.
        tracking_confidence : float
            Minimum confidence [0..1] for landmark tracking across frames.
        """

        # Make sure the model file exists on disk.
        _ensure_model()

        # --- Build the HandLandmarker using the new Task API ---------------
        #
        # RunningMode.VIDEO is used because we feed frames from a webcam
        # one at a time with increasing timestamps.  (LIVE_STREAM would
        # require an async callback, which adds complexity.)
        base_options = mp_tasks.BaseOptions(
            model_asset_path=_MODEL_PATH
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Monotonically-increasing timestamp counter required by VIDEO mode.
        self._frame_ts = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_hand_landmarks(self, frame):
        """
        Process one BGR frame and extract hand landmarks.

        Parameters
        ----------
        frame : numpy.ndarray
            A BGR image from OpenCV (e.g. from cap.read()).

        Returns
        -------
        landmarks : list[(id, x_px, y_px)]
            21 entries — one per landmark — in *pixel* coordinates.
            Empty list if no hand is found.
        result : HandLandmarkerResult
            Raw result (kept for potential future use).
        """

        h, w, _ = frame.shape

        # Convert BGR -> RGB and wrap in a mediapipe.Image.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # detect_for_video() requires a strictly increasing timestamp (ms).
        self._frame_ts += 1
        result = self.detector.detect_for_video(mp_image, self._frame_ts)

        landmarks = []

        # result.hand_landmarks is a list of hands, each a list of 21
        # NormalizedLandmark objects with x, y in [0..1].
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]  # first hand only
            for idx, lm in enumerate(hand):
                px = int(lm.x * w)
                py = int(lm.y * h)
                landmarks.append((idx, px, py))

        return landmarks, result

    def draw_landmarks(self, frame, landmarks):
        """
        Draw the hand skeleton directly on `frame` using OpenCV.

        Instead of relying on mediapipe.drawing_utils (which can behave
        differently across versions and may resize the canvas), we draw
        everything ourselves with simple cv2.line() and cv2.circle().
        This guarantees the frame size is never modified.

        Parameters
        ----------
        frame : numpy.ndarray
            The BGR camera frame to draw on (modified in-place).
        landmarks : list[(id, x_px, y_px)]
            The 21 pixel-coordinate landmarks from find_hand_landmarks().
        """
        if not landmarks or len(landmarks) < 21:
            return frame

        # --- Draw bone connections (white lines) ---
        for (a, b) in HAND_BONE_CONNECTIONS:
            # landmarks[a] = (id, x, y)  →  we need (x, y) as a tuple.
            pt_a = (landmarks[a][1], landmarks[a][2])
            pt_b = (landmarks[b][1], landmarks[b][2])
            cv2.line(frame, pt_a, pt_b, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Draw joint circles ---
        for (idx, x, y) in landmarks:
            # Finger tips (4, 8, 12, 16, 20) get a larger green circle.
            if idx in (4, 8, 12, 16, 20):
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 3, (0, 200, 255), -1, cv2.LINE_AA)

        return frame
