"""
virtual_keyboard.py — On-Screen Virtual Keyboard Module
=======================================================
Draws a QWERTY keyboard overlay on the OpenCV camera frame using
simple rectangles and text.  The user "presses" a key by hovering
the index finger tip over the key box and performing a pinch gesture.

Layout
------
Row 0:  Q W E R T Y U I O P
Row 1:  A S D F G H J K L
Row 2:  Z X C V B N M  [BKSP]
Row 3:  [          SPACE          ]

The keyboard is drawn semi-transparently so the user can still see the
camera feed underneath.
"""

import cv2
import time


class VirtualKeyboard:
    """
    Manages the virtual keyboard overlay.

    Usage:
        kb = VirtualKeyboard()
        kb.compute_layout(frame_w, frame_h)   # call once
        kb.toggle()                            # show / hide
        hovered = kb.get_hovered_key(fx, fy)   # check hover
        press   = kb.get_press_candidate()     # key to type on pinch
        frame   = kb.draw(frame)               # render overlay
    """

    def __init__(self):
        # ----- Key layout (list of rows, each row is a list of labels) -----
        self.rows = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM") + ["BKSP"],
            ["SPACE"],
        ]

        # ----- Key sizing (pixels) -----
        self.key_w = 50       # width of a normal key
        self.key_h = 50       # height of every key
        self.padding = 5      # gap between keys

        # ----- State -----
        self.visible = False          # toggled by the three-finger gesture
        self.hovered_key = None       # key currently under the finger (visual)

        # ----- Press-candidate cache -----
        #
        # BUG FIX: During a pinch the index fingertip shifts a few pixels
        # toward the thumb, often sliding off the key rectangle.  On that
        # exact frame the hovered_key is None, so the press never fires.
        #
        # Solution: we remember the LAST hovered key for a short grace
        # period.  If the pinch happens within that window, we still know
        # which key the user meant to press.
        self._press_candidate = None      # last key that was hovered
        self._press_candidate_time = 0.0  # when it was last hovered
        self.PRESS_GRACE_PERIOD = 0.35    # seconds — keep candidate alive

        # ----- Pre-computed rectangles -----
        # Each entry: (label, x1, y1, x2, y2)
        self.key_rects = []

    # ------------------------------------------------------------------
    # Layout computation
    # ------------------------------------------------------------------

    def compute_layout(self, frame_w, frame_h):
        """
        Calculate pixel positions for every key based on the frame size.
        Call this once after you know the camera resolution.

        The keyboard is centred horizontally and placed near the bottom
        of the frame.
        """
        self.key_rects = []

        # Total height the keyboard occupies.
        total_rows = len(self.rows)
        kb_height = total_rows * (self.key_h + self.padding) + self.padding

        # y position: sit near the bottom with a small margin.
        start_y = frame_h - kb_height - 20

        for row_idx, row in enumerate(self.rows):
            y1 = start_y + row_idx * (self.key_h + self.padding)
            y2 = y1 + self.key_h

            if row == ["SPACE"]:
                # The SPACE bar spans the same width as the longest row.
                row_width = len(self.rows[0]) * (self.key_w + self.padding)
                x1 = (frame_w - row_width) // 2
                x2 = x1 + row_width
                self.key_rects.append(("SPACE", x1, y1, x2, y2))
            else:
                # Centre this row horizontally.
                row_width = (len(row) * (self.key_w + self.padding)
                             - self.padding)
                row_start_x = (frame_w - row_width) // 2

                for col_idx, key in enumerate(row):
                    x1 = row_start_x + col_idx * (self.key_w + self.padding)
                    # BKSP is a bit wider so the label fits.
                    x2 = x1 + (self.key_w + 20 if key == "BKSP"
                               else self.key_w)
                    self.key_rects.append((key, x1, y1, x2, y2))

    # ------------------------------------------------------------------
    # Visibility toggle
    # ------------------------------------------------------------------

    def toggle(self):
        """Flip keyboard visibility on <-> off."""
        self.visible = not self.visible
        # Clear cached state when toggling.
        self.hovered_key = None
        self._press_candidate = None

    # ------------------------------------------------------------------
    # Hover detection
    # ------------------------------------------------------------------

    def get_hovered_key(self, finger_x, finger_y):
        """
        Check whether the finger tip is inside any key's bounding box.

        Also updates the _press_candidate cache so that a pinch that
        happens a few frames later still knows which key was targeted.

        Parameters
        ----------
        finger_x, finger_y : int
            Index finger tip position in camera-pixel coords.

        Returns
        -------
        str or None
            The label of the hovered key, or None if not hovering.
        """
        if not self.visible:
            return None

        self.hovered_key = None   # reset visual highlight each frame

        for label, x1, y1, x2, y2 in self.key_rects:
            if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                self.hovered_key = label
                # Update the press-candidate cache while hovering.
                self._press_candidate = label
                self._press_candidate_time = time.time()
                return label

        return None

    def get_press_candidate(self):
        """
        Return the key that should be typed on a pinch.

        Prefers the currently hovered key.  Falls back to the last
        hovered key if it's within the grace period (covers the case
        where the fingertip shifted off the key during the pinch).
        """
        if self.hovered_key:
            return self.hovered_key

        # Grace period fallback.
        if (self._press_candidate
                and time.time() - self._press_candidate_time
                < self.PRESS_GRACE_PERIOD):
            return self._press_candidate

        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, frame):
        """
        Render the keyboard overlay on `frame` (in-place).

        Uses cv2.addWeighted for a semi-transparent look so the camera
        feed is still visible underneath.
        """
        if not self.visible:
            return frame

        # Work on a copy so we can blend later.
        overlay = frame.copy()

        for label, x1, y1, x2, y2 in self.key_rects:
            # Hovered key -> green, default -> dark grey.
            if label == self.hovered_key:
                bg_color = (0, 200, 0)        # green highlight
                txt_color = (0, 0, 0)          # black text
            else:
                bg_color = (50, 50, 50)        # dark grey
                txt_color = (255, 255, 255)    # white text

            # Filled rectangle (background).
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
            # Thin border.
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), 1)

            # Centre the label inside the key box.
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45 if len(label) > 1 else 0.55
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(overlay, label, (tx, ty), font, scale,
                        txt_color, thickness, cv2.LINE_AA)

        # Blend: 70 % keyboard overlay, 30 % original feed.
        alpha = 0.70
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame
