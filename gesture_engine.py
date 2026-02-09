"""
gesture_engine.py — Gesture Detection Module
=============================================
All gesture recognition lives here. Every gesture is *rule-based* —
we simply compare landmark positions and distances. No ML training needed!

Gesture catalogue implemented in this file
------------------------------------------
1. Index finger pointing  → cursor control
2. Thumb + Index pinch     → left click / drag
3. Index + Middle pinch    → right click
4. Two-finger vertical     → scroll
5. Three-finger raise      → toggle virtual keyboard
6. Closed fist             → pause cursor

Key landmark indices used
-------------------------
    Thumb  tip=4  ip=3  mcp=2
    Index  tip=8  pip=6 mcp=5
    Middle tip=12 pip=10 mcp=9
    Ring   tip=16 pip=14 mcp=13
    Pinky  tip=20 pip=18 mcp=17
    Wrist=0
"""

import math


class GestureEngine:
    """
    Stateless gesture detector that works on a list of 21 landmarks.

    Each landmark is a tuple: (id, x_pixel, y_pixel).
    """

    # Distance (pixels) below which two finger tips count as "pinched".
    PINCH_THRESHOLD = 40

    def __init__(self):
        # Used by the scroll gesture to track vertical movement over time.
        self.prev_scroll_y = None

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def distance(p1, p2):
        """
        Euclidean distance between two landmark tuples (id, x, y).
        We ignore the id field and just use x, y.
        """
        return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    @staticmethod
    def is_finger_extended(landmarks, tip_id, pip_id):
        """
        A finger is considered "extended" when its TIP is above (lower y)
        its PIP joint.  In image coordinates y increases downward, so:
            extended  →  tip_y  <  pip_y
        """
        tip = landmarks[tip_id]
        pip_joint = landmarks[pip_id]
        return tip[2] < pip_joint[2]

    def is_thumb_extended(self, landmarks):
        """
        The thumb doesn't bend like the other four fingers — it moves
        *sideways*.  So we check whether the thumb tip (4) is farther
        from the palm base (thumb MCP = 2) than the thumb IP joint (3).
        """
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        return abs(thumb_tip[1] - thumb_mcp[1]) > abs(thumb_ip[1] - thumb_mcp[1])

    def get_extended_fingers(self, landmarks):
        """
        Return a list of five booleans — one per finger:
            [thumb, index, middle, ring, pinky]
        True means the finger is extended (open), False means curled.
        """
        if len(landmarks) < 21:
            return [False] * 5

        return [
            self.is_thumb_extended(landmarks),              # Thumb
            self.is_finger_extended(landmarks, 8, 6),       # Index
            self.is_finger_extended(landmarks, 12, 10),     # Middle
            self.is_finger_extended(landmarks, 16, 14),     # Ring
            self.is_finger_extended(landmarks, 20, 18),     # Pinky
        ]

    # ------------------------------------------------------------------
    # Individual gesture detectors
    # ------------------------------------------------------------------

    def detect_thumb_index_pinch(self, landmarks):
        """
        GESTURE: Thumb tip (4) touches Index tip (8).

        We do NOT require the index finger to be "extended" (tip above PIP)
        because when the user points downward at the virtual keyboard the
        index tip is naturally BELOW the PIP joint in screen-y coordinates.
        The old guard `is_finger_extended(8, 6)` blocked every pinch in
        keyboard mode — that was the root cause of the "no typing" bug.

        False-positive safety: closed fist is checked at Priority 1 in
        main.py, so we never reach this code when all fingers are curled.

        Returns
        -------
        is_pinching : bool
        distance    : float   (useful for debugging / UI display)
        """
        if len(landmarks) < 21:
            return False, 0.0

        # Pure distance check — no extension guard needed.
        dist = self.distance(landmarks[4], landmarks[8])
        return dist < self.PINCH_THRESHOLD, dist

    def detect_index_middle_pinch(self, landmarks):
        """
        GESTURE: Index tip (8) touches Middle tip (12) → right click.

        Both fingers should be extended; otherwise it's just a fist.

        Returns
        -------
        is_pinching : bool
        distance    : float
        """
        if len(landmarks) < 21:
            return False, 0.0

        index_ext = self.is_finger_extended(landmarks, 8, 6)
        middle_ext = self.is_finger_extended(landmarks, 12, 10)
        if not (index_ext and middle_ext):
            return False, 0.0

        dist = self.distance(landmarks[8], landmarks[12])
        return dist < self.PINCH_THRESHOLD, dist

    def detect_closed_fist(self, landmarks):
        """
        GESTURE: ALL fingers curled → pause cursor movement.

        Returns True when no finger is extended.
        """
        if len(landmarks) < 21:
            return False
        fingers = self.get_extended_fingers(landmarks)
        return not any(fingers)

    def detect_three_finger_raise(self, landmarks):
        """
        GESTURE: Index + Middle + Ring extended, Thumb + Pinky curled.
        Used to toggle the virtual keyboard on/off.
        """
        if len(landmarks) < 21:
            return False
        fingers = self.get_extended_fingers(landmarks)
        # fingers = [thumb, index, middle, ring, pinky]
        return (not fingers[0]       # thumb curled
                and fingers[1]       # index up
                and fingers[2]       # middle up
                and fingers[3]       # ring up
                and not fingers[4])  # pinky curled

    def detect_scroll(self, landmarks):
        """
        GESTURE: Index + Middle extended, all others curled.
        Vertical movement of the two finger tips drives scrolling.

        Returns
        -------
        is_scrolling : bool
        delta_y      : float   positive = fingers moved down, negative = up
        """
        if len(landmarks) < 21:
            self.prev_scroll_y = None
            return False, 0.0

        fingers = self.get_extended_fingers(landmarks)
        two_up = (not fingers[0]       # thumb curled
                  and fingers[1]       # index up
                  and fingers[2]       # middle up
                  and not fingers[3]   # ring curled
                  and not fingers[4])  # pinky curled

        if not two_up:
            self.prev_scroll_y = None
            return False, 0.0

        # Average y of the two extended finger tips.
        avg_y = (landmarks[8][2] + landmarks[12][2]) / 2.0

        if self.prev_scroll_y is None:
            # First frame of scrolling — just record, don't scroll yet.
            self.prev_scroll_y = avg_y
            return True, 0.0

        delta_y = avg_y - self.prev_scroll_y
        self.prev_scroll_y = avg_y
        return True, delta_y

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_index_finger_tip(self, landmarks):
        """
        Return the index finger tip landmark: (8, x, y).
        Returns None if landmarks are incomplete.
        """
        if len(landmarks) < 21:
            return None
        return landmarks[8]
