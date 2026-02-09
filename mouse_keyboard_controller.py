"""
mouse_keyboard_controller.py — Mouse & Keyboard Control Module
===============================================================
Translates detected gestures into real OS-level actions using pyautogui:
    • Move the mouse cursor  (with coordinate mapping + smoothing)
    • Left / right click     (with cooldown timers)
    • Click-and-drag
    • Scroll
    • Send keyboard key presses

Coordinate pipeline:
    camera pixel (0..640, 0..480)
        ↓  map_coordinates()   — maps an inner "active zone" to full screen
        ↓  smooth_position()   — exponential moving average removes jitter
        → pyautogui.moveTo()

Cooldown timers prevent a single pinch gesture from firing dozens of
clicks across consecutive frames.
"""

import time
import numpy as np
import pyautogui

# ------------------------------------------------------------------
# pyautogui global settings
# ------------------------------------------------------------------
# FAILSAFE: moving the mouse to the top-left corner normally raises an
# exception and kills the program.  We disable it so the user doesn't
# accidentally crash the app.
pyautogui.FAILSAFE = False

# PAUSE: pyautogui adds a tiny sleep after every call by default.
# Set to 0 for maximum responsiveness.
pyautogui.PAUSE = 0


class MouseKeyboardController:
    """
    Bridges the gesture layer and the operating system.

    Usage:
        ctrl = MouseKeyboardController()
        ctrl.move_cursor(finger_x, finger_y, frame_w, frame_h)
        ctrl.left_click()
    """

    def __init__(self, smoothing_factor=0.3, cooldown_time=0.4):
        """
        Parameters
        ----------
        smoothing_factor : float  (0..1)
            Controls cursor smoothing.
            Lower  → smoother but sluggish.
            Higher → snappy but jittery.
            0.3 is a good starting point.
        cooldown_time : float  (seconds)
            Minimum interval between repeated click actions.
        """

        # Screen resolution (needed for coordinate mapping).
        self.screen_w, self.screen_h = pyautogui.size()

        # --- Smoothing state ---
        self.smoothing = smoothing_factor
        self.prev_x = self.screen_w / 2.0   # start in center
        self.prev_y = self.screen_h / 2.0

        # --- Cooldown timers (seconds) ---
        self.cooldown = cooldown_time
        self.last_click_time = 0.0
        self.last_right_click_time = 0.0
        self.last_scroll_time = 0.0
        self.last_toggle_time = 0.0
        # BUG FIX: key presses get their OWN timer so a mouse click in
        # mouse mode doesn't eat the keyboard cooldown (and vice versa).
        self.last_key_press_time = 0.0
        self.key_press_cooldown = 0.5  # seconds between virtual key presses

        # --- Drag state ---
        self.is_dragging = False
        self.pinch_start_time = 0.0
        # How long the pinch must be held before it becomes a drag
        # instead of a simple click.
        self.DRAG_HOLD_THRESHOLD = 0.3  # seconds

        # --- Pause flag (set by closed-fist gesture) ---
        self.paused = False

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def map_coordinates(self, finger_x, finger_y, frame_w, frame_h,
                        margin=0.15):
        """
        Map a finger position in *camera-pixel* space to *screen* space.

        We define an "active zone" inside the camera frame (shrunk by
        `margin` on each side) so the user doesn't have to reach the
        very edges of the camera view to hit the screen corners.

        NOTE: The camera frame is already horizontally flipped in
        main.py (cv2.flip), so finger_x already moves in the natural
        direction.  No extra flip is needed here.

        Steps:
            1. Clamp finger position to the active zone.
            2. Linearly interpolate from active zone → full screen.
        """
        # Active zone boundaries (in camera pixels).
        x_min = frame_w * margin
        x_max = frame_w * (1 - margin)
        y_min = frame_h * margin
        y_max = frame_h * (1 - margin)

        # Clamp so we never go outside the active zone.
        clamped_x = np.clip(finger_x, x_min, x_max)
        clamped_y = np.clip(finger_y, y_min, y_max)

        # Linear interpolation: active zone → screen.
        screen_x = np.interp(clamped_x, (x_min, x_max), (0, self.screen_w))
        screen_y = np.interp(clamped_y, (y_min, y_max), (0, self.screen_h))

        return float(screen_x), float(screen_y)

    # ------------------------------------------------------------------
    # Cursor smoothing
    # ------------------------------------------------------------------

    def smooth_position(self, target_x, target_y):
        """
        Exponential moving average (EMA) to reduce cursor jitter.

        Formula:  new = prev + α × (target − prev)

        Where α = self.smoothing.  A small α smooths a lot but adds lag;
        a large α reacts quickly but lets through more jitter.
        """
        smooth_x = self.prev_x + self.smoothing * (target_x - self.prev_x)
        smooth_y = self.prev_y + self.smoothing * (target_y - self.prev_y)

        # Remember for next frame.
        self.prev_x = smooth_x
        self.prev_y = smooth_y

        return int(smooth_x), int(smooth_y)

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def move_cursor(self, finger_x, finger_y, frame_w, frame_h):
        """
        Move the OS cursor to the position indicated by the index finger.

        Does nothing when self.paused is True (closed-fist gesture).
        """
        if self.paused:
            return

        # Step 1: camera pixels → screen pixels.
        raw_x, raw_y = self.map_coordinates(finger_x, finger_y,
                                            frame_w, frame_h)
        # Step 2: apply smoothing filter.
        smooth_x, smooth_y = self.smooth_position(raw_x, raw_y)

        # Step 3: move the real cursor.
        pyautogui.moveTo(smooth_x, smooth_y)

    # ------------------------------------------------------------------
    # Click actions
    # ------------------------------------------------------------------

    def left_click(self):
        """Perform a left click (respects cooldown)."""
        now = time.time()
        if now - self.last_click_time > self.cooldown:
            pyautogui.click()
            self.last_click_time = now
            return True
        return False

    def right_click(self):
        """Perform a right click (respects cooldown)."""
        now = time.time()
        if now - self.last_right_click_time > self.cooldown:
            pyautogui.rightClick()
            self.last_right_click_time = now
            return True
        return False

    # ------------------------------------------------------------------
    # Drag actions
    # ------------------------------------------------------------------

    def start_drag(self):
        """Press and hold the left mouse button to begin dragging."""
        if not self.is_dragging:
            pyautogui.mouseDown()
            self.is_dragging = True

    def stop_drag(self):
        """Release the left mouse button to end dragging."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def scroll(self, delta_y):
        """
        Scroll based on vertical finger movement.

        delta_y > 0  means fingers moved DOWN  → scroll down (negative).
        delta_y < 0  means fingers moved UP    → scroll up   (positive).
        """
        now = time.time()
        if now - self.last_scroll_time > 0.05:  # tiny cooldown for smoothness
            scroll_amount = int(-delta_y / 5)   # scale + invert
            if scroll_amount != 0:
                pyautogui.scroll(scroll_amount)
            self.last_scroll_time = now

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def press_key(self, key):
        """
        Send a single key press to the OS.

        Handles special keys (SPACE, BKSP) and normal letter keys.
        Uses its own cooldown timer (last_key_press_time) so it doesn't
        conflict with the mouse-click cooldown.
        """
        now = time.time()
        if now - self.last_key_press_time > self.key_press_cooldown:
            if key == "SPACE":
                pyautogui.press("space")
            elif key == "BKSP":
                pyautogui.press("backspace")
            else:
                pyautogui.press(key.lower())
            self.last_key_press_time = now
            # Debug: print to console so you can verify presses are firing.
            print(f"[KEY PRESSED] {key}")
            return True
        return False

    # ------------------------------------------------------------------
    # Toggle cooldown (for virtual keyboard show/hide)
    # ------------------------------------------------------------------

    def can_toggle(self):
        """Return True if enough time has passed since the last toggle."""
        now = time.time()
        if now - self.last_toggle_time > 0.8:  # longer cooldown for toggles
            self.last_toggle_time = now
            return True
        return False
