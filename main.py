"""
main.py — AirPointer: Touchless Mouse & Virtual Keyboard
=========================================================
This is the entry point. Run with:

    python main.py

Pipeline (every frame):
    1. Capture webcam frame
    2. Detect hand landmarks  (hand_tracker)
    3. Recognise gesture      (gesture_engine)
    4. Trigger OS action      (mouse_keyboard_controller)
    5. Draw UI overlays       (virtual_keyboard + status bar)
    6. Display frame  ← everything drawn on the SAME webcam frame

Press 'q' in the OpenCV window to quit.

Gesture cheat-sheet:
    Index finger        → move cursor
    Thumb+Index pinch   → left click  (quick) / drag (hold)
    Index+Middle pinch  → right click
    Two fingers up      → scroll (vertical movement)
    Three fingers up    → toggle virtual keyboard
    Closed fist         → pause cursor
"""

# ---------------------------------------------------------------------------
# FIX #1 — Windows DPI awareness.
#
# On Windows with display scaling (125 %, 150 %, etc.) OpenCV renders the
# frame at its native pixel size inside a window that is scaled up, leaving
# gray padding around the image.  Calling SetProcessDpiAwareness(2) tells
# Windows to give us raw pixel coordinates — the frame fills the window.
# ---------------------------------------------------------------------------
import ctypes
import sys

if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass  # older Windows or non-standard environment

import cv2
import time

# Import our four custom modules.
from hand_tracker import HandTracker
from gesture_engine import GestureEngine
from mouse_keyboard_controller import MouseKeyboardController
from virtual_keyboard import VirtualKeyboard


def print_gesture_summary():
    """Print a nice gesture reference table to the console at startup."""
    print()
    print("=" * 55)
    print("   AirPointer — Gesture Control Summary")
    print("=" * 55)
    print("  GESTURE                  ACTION")
    print("  -----------------------  -----------------------")
    print("  Index finger pointing    Move cursor")
    print("  Thumb + Index pinch      Left click (quick tap)")
    print("  Thumb + Index hold       Click-and-drag")
    print("  Index + Middle pinch     Right click")
    print("  Two fingers up + move    Scroll up / down")
    print("  Three fingers up (I/M/R) Toggle virtual keyboard")
    print("  Closed fist              Pause cursor movement")
    print("  Hover key + pinch        Type key (keyboard mode)")
    print("=" * 55)
    print("  Press 'q' in the window to quit.")
    print("=" * 55)
    print()


def main():
    # ==================================================================
    # 1.  INITIALISATION
    # ==================================================================

    print_gesture_summary()

    # --- Webcam ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    # Read one frame to get the ACTUAL resolution the camera provides.
    # Some cameras ignore the requested 640x480 and return a different size.
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        return
    FRAME_H, FRAME_W = test_frame.shape[:2]
    print(f"[INFO] Camera resolution: {FRAME_W}x{FRAME_H}")

    # ------------------------------------------------------------------
    # FIX #2 — Create the window BEFORE the loop with WINDOW_NORMAL.
    #
    # cv2.WINDOW_NORMAL lets us resize the window freely AND avoids
    # the DPI-scaling gray-border issue.  We then set its size to
    # match the actual camera resolution so the image fills entirely.
    # ------------------------------------------------------------------
    WINDOW_NAME = "AirPointer - Touchless Control"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_W, FRAME_H)

    # --- Modules ---
    tracker = HandTracker(
        max_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.7,
    )
    gesture = GestureEngine()
    controller = MouseKeyboardController(
        smoothing_factor=0.3,
        cooldown_time=0.4,
    )
    keyboard = VirtualKeyboard()

    # Compute the virtual-keyboard layout based on actual frame size.
    keyboard.compute_layout(FRAME_W, FRAME_H)

    # --- UI state ---
    prev_time = time.time()          # for FPS counter
    status_text = "Ready"            # top-bar status message
    gesture_text = "None"            # currently detected gesture
    last_pressed_key = ""            # for on-screen "KEY PRESSED: X"
    last_pressed_time = 0.0          # when it was pressed

    # ==================================================================
    # 2.  MAIN LOOP
    # ==================================================================

    while True:
        # --------------------------------------------------------------
        # STEP 1 — Capture frame
        # --------------------------------------------------------------
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame. Exiting.")
            break

        # Flip horizontally so the image acts like a mirror.
        # This means the user's right hand appears on the right side,
        # which feels natural.
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        # --------------------------------------------------------------
        # STEP 2 — Detect hand landmarks
        # --------------------------------------------------------------
        landmarks, results = tracker.find_hand_landmarks(frame)

        # ------------------------------------------------------------------
        # FIX #3 — draw_landmarks now takes `landmarks` (pixel-coord list)
        # instead of the raw MediaPipe result object.  This uses plain
        # cv2.line / cv2.circle — no mediapipe drawing_utils, so the frame
        # size is never accidentally changed.
        # ------------------------------------------------------------------
        frame = tracker.draw_landmarks(frame, landmarks)

        # --------------------------------------------------------------
        # STEP 3 — Gesture detection & action dispatch
        # --------------------------------------------------------------
        if landmarks:
            # ........................................................
            # Priority 1: CLOSED FIST → pause everything
            # ........................................................
            if gesture.detect_closed_fist(landmarks):
                gesture_text = "Fist (Paused)"
                controller.paused = True
                controller.stop_drag()

            # ........................................................
            # Priority 2: THREE-FINGER RAISE → toggle keyboard
            # ........................................................
            elif gesture.detect_three_finger_raise(landmarks):
                gesture_text = "3-Finger (Toggle KB)"
                controller.paused = False
                if controller.can_toggle():
                    keyboard.toggle()
                    status_text = ("Keyboard ON" if keyboard.visible
                                   else "Keyboard OFF")

            # ........................................................
            # Priority 3+: Pinch / scroll / point
            # ........................................................
            else:
                controller.paused = False
                index_tip = gesture.get_index_finger_tip(landmarks)

                # --- Detect pinches first (higher priority) ---
                thumb_index_pinch, pinch_dist = \
                    gesture.detect_thumb_index_pinch(landmarks)
                idx_mid_pinch, _ = \
                    gesture.detect_index_middle_pinch(landmarks)

                # --- Detect scroll only when no pinch is active ---
                if (not thumb_index_pinch and not idx_mid_pinch
                        and not keyboard.visible):
                    is_scrolling, scroll_delta = \
                        gesture.detect_scroll(landmarks)
                else:
                    is_scrolling = False
                    gesture.prev_scroll_y = None  # reset scroll state

                # --- Move cursor / hover keyboard ---
                if not is_scrolling and index_tip:
                    fx, fy = index_tip[1], index_tip[2]
                    if keyboard.visible:
                        # Hover-detect over virtual keys (no real cursor
                        # movement while keyboard is showing).
                        keyboard.get_hovered_key(fx, fy)
                    else:
                        # Move the real OS cursor.
                        controller.move_cursor(fx, fy, frame_w, frame_h)

                # ..................................................
                # ACTION: Thumb + Index PINCH
                # ..................................................
                if thumb_index_pinch:
                    # --- Virtual keyboard key press ---
                    # Use get_press_candidate() instead of hovered_key
                    # directly — it survives the few frames where the
                    # finger tip slides off the key during the pinch.
                    press_key = (keyboard.get_press_candidate()
                                 if keyboard.visible else None)
                    if keyboard.visible and press_key:
                        gesture_text = f"KB Press: {press_key}"
                        if controller.press_key(press_key):
                            status_text = f"Typed: {press_key}"
                            last_pressed_key = press_key
                            last_pressed_time = time.time()
                    else:
                        # --- Click / Drag logic ---
                        # Record when the pinch first started.
                        if controller.pinch_start_time == 0:
                            controller.pinch_start_time = time.time()

                        hold = time.time() - controller.pinch_start_time

                        if hold > controller.DRAG_HOLD_THRESHOLD:
                            # Held long enough → drag mode.
                            gesture_text = "Dragging"
                            controller.start_drag()
                            # Keep moving cursor while dragging.
                            if index_tip:
                                controller.move_cursor(
                                    index_tip[1], index_tip[2],
                                    frame_w, frame_h,
                                )
                        else:
                            gesture_text = "Pinch Hold..."

                # ..................................................
                # ACTION: Index + Middle PINCH → right click
                # ..................................................
                elif idx_mid_pinch and not keyboard.visible:
                    gesture_text = "Right Click"
                    controller.right_click()
                    controller.pinch_start_time = 0

                # ..................................................
                # ACTION: SCROLL
                # ..................................................
                elif is_scrolling:
                    gesture_text = "Scrolling"
                    if abs(scroll_delta) > 2:
                        controller.scroll(scroll_delta)

                # ..................................................
                # NO PINCH → handle release / default pointing
                # ..................................................
                else:
                    if controller.is_dragging:
                        # Pinch just released after a drag.
                        controller.stop_drag()
                        gesture_text = "Drag Released"
                    elif controller.pinch_start_time > 0:
                        # Pinch was short → treat as a left click.
                        controller.left_click()
                        gesture_text = "Left Click"
                    else:
                        gesture_text = "Pointing"

                    # Reset pinch timer.
                    controller.pinch_start_time = 0

        else:
            # ----------------------------------------------------------
            # No hand detected — reset everything
            # ----------------------------------------------------------
            gesture_text = "No Hand"
            controller.stop_drag()
            controller.pinch_start_time = 0
            gesture.prev_scroll_y = None

        # --------------------------------------------------------------
        # STEP 4 — Draw ALL UI overlays on the SAME frame
        #
        # Everything is drawn directly onto the webcam frame.
        # No separate canvas. No concatenation. No resizing.
        # --------------------------------------------------------------

        # 4a. Virtual keyboard overlay (only when visible).
        frame = keyboard.draw(frame)

        # 4b. "KEYBOARD MODE ON" large indicator when keyboard is active.
        if keyboard.visible:
            cv2.putText(frame, "KEYBOARD MODE ON", (frame_w // 2 - 150, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        cv2.LINE_AA)

        # 4b-2. Flash "KEY PRESSED: X" for 1 second after each key press.
        if last_pressed_key and time.time() - last_pressed_time < 1.0:
            cv2.putText(frame, f"KEY PRESSED: {last_pressed_key}",
                        (frame_w // 2 - 130, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA)

        # 4c. Draw a bright circle on the index finger tip so the user
        #     can see exactly where they are pointing / hovering.
        if landmarks and len(landmarks) >= 21:
            tip_x, tip_y = landmarks[8][1], landmarks[8][2]
            cv2.circle(frame, (tip_x, tip_y), 10, (255, 0, 255), 2,
                       cv2.LINE_AA)

        # 4d. FPS counter.
        curr_time = time.time()
        fps = (1.0 / (curr_time - prev_time)
               if (curr_time - prev_time) > 0 else 0)
        prev_time = curr_time

        # 4e. Status bar background (top of frame).
        cv2.rectangle(frame, (0, 0), (frame_w, 65), (0, 0, 0), -1)

        # FPS (green).
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Status message (yellow).
        cv2.putText(frame, f"Status: {status_text}", (130, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Current gesture (orange).
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

        # Mode indicator (pink).
        mode = "KEYBOARD" if keyboard.visible else "MOUSE"
        cv2.putText(frame, f"Mode: {mode}", (420, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)

        # Pinch-distance debug (small, bottom-left).
        if landmarks:
            _, pdist = gesture.detect_thumb_index_pinch(landmarks)
            cv2.putText(frame, f"Pinch dist: {int(pdist)}px",
                        (10, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --------------------------------------------------------------
        # STEP 5 — Show frame (single window, full resolution)
        # --------------------------------------------------------------
        cv2.imshow(WINDOW_NAME, frame)

        # 'q' key exits the loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==================================================================
    # 3.  CLEANUP
    # ==================================================================
    cap.release()
    cv2.destroyAllWindows()
    print("AirPointer shut down cleanly.")


# Standard Python entry point.
if __name__ == "__main__":
    main()
