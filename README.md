# AirPointer

### Touchless Mouse & Virtual Keyboard using Hand Gestures

Control your computer without touching it. AirPointer uses your webcam to track hand gestures in real-time and translates them into mouse movements, clicks, scrolling, and keyboard input — all using just your hand in the air.

Built with **MediaPipe** + **OpenCV** + **pyautogui**. No deep learning training required — everything is rule-based using hand landmark positions.

---

## Demo

```
+--------------------------------------------------+
|  FPS: 30  Status: Ready     Mode: MOUSE          |
|  Gesture: Pointing                                |
|                                                   |
|                 .---.                              |
|                / 8   \    <-- Index finger tip     |
|               |  |    |       controls cursor      |
|            6 -+  |    +- 12                        |
|               |  |    |                            |
|            5 -+--+----+- 9                         |
|               \ 4|   /                             |
|                \  |  / <-- Thumb+Index pinch       |
|                 +-+-+      = left click            |
|                 | 0 |                              |
|                 +---+                              |
|                                                   |
|  Pinch dist: 85px                                 |
+--------------------------------------------------+
```

---

## Architecture

```
                        AirPointer Pipeline
                        ===================

    +-------------+     +----------------+     +------------------+
    |   Webcam    | --> | hand_tracker   | --> | gesture_engine   |
    |  (OpenCV)   |     |  (MediaPipe)   |     |  (Rule-based)    |
    | cv2.capture |     | 21 landmarks   |     | Pinch / Fist /   |
    |  + flip     |     | in pixel coords|     | Scroll / 3-finger|
    +-------------+     +----------------+     +------------------+
                                                       |
                                                       v
    +-------------+     +-------------------+   +------------------+
    |  Display    | <-- | virtual_keyboard  |   | mouse_keyboard   |
    |  cv2.imshow |     | (OpenCV drawing)  |   | _controller      |
    |  (single    |     | QWERTY overlay    |   | (pyautogui)      |
    |   window)   |     | hover + press     |   | cursor / click / |
    +-------------+     +-------------------+   | scroll / type    |
                                                +------------------+

    Every frame: capture -> detect -> gesture -> action -> draw -> display
```

### File Structure

```
AirPointer/
|
|-- main.py                       Entry point — main loop pipeline
|-- hand_tracker.py               MediaPipe hand detection + landmark extraction
|-- gesture_engine.py             Rule-based gesture recognition (6 gestures)
|-- mouse_keyboard_controller.py  pyautogui OS control (cursor, clicks, keys)
|-- virtual_keyboard.py           OpenCV QWERTY keyboard overlay
|-- requirements.txt              Python dependencies
|-- hand_landmarker.task          MediaPipe model (auto-downloaded on first run)
|-- README.md                     This file
```

### Module Responsibilities

| Module | Role |
|---|---|
| `main.py` | Captures frames, orchestrates the full pipeline, draws UI overlays |
| `hand_tracker.py` | Wraps MediaPipe Tasks API, returns 21 landmarks in pixel coords, draws hand skeleton |
| `gesture_engine.py` | Detects 6 gestures using landmark distances and finger extension checks |
| `mouse_keyboard_controller.py` | Maps camera coords to screen, applies cursor smoothing (EMA), sends OS events via pyautogui |
| `virtual_keyboard.py` | Renders semi-transparent QWERTY keyboard, handles hover detection with press-candidate caching |

---

## Gesture Guide

### Mouse Mode (Default)

| Gesture | How To Do It | Action |
|---|---|---|
| **Point** | Raise only your index finger | Move cursor on screen |
| **Left Click** | Tap thumb tip to index tip (quick pinch) | Single left click |
| **Drag** | Pinch and hold for 0.3+ seconds | Click-and-drag |
| **Right Click** | Bring index and middle fingertips together | Single right click |
| **Scroll** | Raise index + middle fingers, move hand up/down | Scroll up / down |
| **Pause** | Close your fist (all fingers curled) | Freeze cursor in place |

### Keyboard Mode

| Gesture | How To Do It | Action |
|---|---|---|
| **Toggle Keyboard** | Raise 3 fingers (index + middle + ring) | Switch between Mouse / Keyboard mode |
| **Hover Key** | Point index finger over a key on the overlay | Key highlights green |
| **Press Key** | Hover over key, then thumb+index pinch | Types that key into the focused app |

### Visual Gesture Reference

```
    POINT (Move Cursor)         PINCH (Left Click)        FIST (Pause)

         |                           |
         |                          /|                     .---.
         |                         / |                    | ... |
         |                        /  |                    | ... |
    _____|____               ____x___|____               |_____|
   |          |             |             |             |         |
   |__________|             |_____________|             |_________|

   Index finger up          Thumb touches              All fingers
   Others curled            index tip                  curled in


    RIGHT CLICK               SCROLL                  3-FINGER TOGGLE

        ||                       ||                      |||
        ||                       ||                      |||
        ||                       ||                      |||
    ____||____               ____||____               ___|||___
   |          |             |          |             |          |
   |__________|             |__________|             |__________|

   Index + Middle            Index + Middle           Index + Middle
   tips together             up, move vertical        + Ring up
```

### Keyboard Layout (On-Screen)

```
  +---+---+---+---+---+---+---+---+---+---+
  | Q | W | E | R | T | Y | U | I | O | P |
  +---+---+---+---+---+---+---+---+---+---+
    +---+---+---+---+---+---+---+---+---+
    | A | S | D | F | G | H | J | K | L |
    +---+---+---+---+---+---+---+---+---+
      +---+---+---+---+---+---+---+------+
      | Z | X | C | V | B | N | M | BKSP |
      +---+---+---+---+---+---+---+------+
      +----------------------------------+
      |             SPACE                |
      +----------------------------------+
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- A working webcam
- Windows / macOS / Linux

### Setup

```bash
# Clone the repository
git clone https://github.com/aryanchauhan07/AirPointer.git
cd AirPointer

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

On the **first run**, the MediaPipe hand landmarker model (~7.5 MB) is downloaded automatically and cached in the project folder.

### Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture, image processing, UI drawing |
| `mediapipe` | Hand landmark detection (21 points per hand) |
| `pyautogui` | OS-level mouse and keyboard control |
| `numpy` | Coordinate mapping and array operations |

---

## How It Works

### 1. Hand Detection

MediaPipe's Hand Landmarker model detects **21 landmarks** on your hand in real-time:

```
        THUMB_TIP (4)
       /
      3
     /
    2           INDEX_TIP (8)      MIDDLE_TIP (12)    RING_TIP (16)    PINKY_TIP (20)
   /           /                  /                   /                /
  1           7                  11                  15               19
  |          /                  /                   /                /
  |         6                  10                  14               18
  |        /                  /                   /                /
  +------ 5 ------- 9 ------ 13 ------ 17 ------+
  |                                              |
  0 (WRIST) -------- PALM -----------------------+
```

### 2. Gesture Recognition (Rule-Based)

Each gesture is detected by comparing landmark positions:

- **Pinch**: Euclidean distance between thumb tip (4) and index tip (8) < 40px
- **Finger extended**: Fingertip y-coordinate < PIP joint y-coordinate (tip is above joint)
- **Closed fist**: No finger is extended
- **Scroll**: Index + middle extended, others curled, track vertical delta

### 3. Coordinate Mapping

```
Camera Frame (640x480)              Screen (1920x1080)
+---------------------------+       +---------------------------+
|    +-------------------+  |       |                           |
|    |   Active Zone     |  |  -->  |     Full Screen           |
|    |   (70% of frame)  |  |       |     Mapped Area           |
|    +-------------------+  |       |                           |
+---------------------------+       +---------------------------+

Active zone (15% margin each side) maps to full screen resolution.
This means you don't need to reach the edges of the camera view.
```

### 4. Cursor Smoothing

An **Exponential Moving Average (EMA)** filter removes hand jitter:

```
smoothed_position = previous + alpha * (current - previous)

alpha = 0.3  (lower = smoother but laggier, higher = snappier but jittery)
```

### 5. Cooldown Timers

Prevent gesture spam across consecutive frames:

| Action | Cooldown |
|---|---|
| Left click | 0.4 seconds |
| Right click | 0.4 seconds |
| Key press | 0.5 seconds |
| Keyboard toggle | 0.8 seconds |
| Scroll | 0.05 seconds |

---

## Usage Tips

- **Webcam position**: Place the camera at chest/face level for best hand tracking
- **Lighting**: Ensure good, even lighting — avoid strong backlighting
- **Hand distance**: Keep your hand 30-60 cm from the camera
- **Typing**: When using the virtual keyboard, the key presses go to **whichever application has OS focus** — click your target app (Notepad, browser, etc.) before switching to keyboard mode
- **Quit**: Press `q` in the AirPointer window to exit

---

## On-Screen Display

```
+--------------------------------------------------+
| FPS: 30   Status: Keyboard ON    Mode: KEYBOARD  |  <-- Status bar
| Gesture: KB Press: A                              |
|                                                   |
|            KEYBOARD MODE ON                       |  <-- Mode indicator
|            KEY PRESSED: A                         |  <-- Flash on press
|                                                   |
|                  (o)  <-- Finger tip tracker      |
|                                                   |
| +---+---+---+---+---+---+---+---+---+---+        |
| | Q | W | E | R | T | Y | U | I | O | P |        |
| +---+---+---+---+---+---+---+---+---+---+        |
|   +---+---+---+---+---+---+---+---+---+          |
|   |[A]| S | D | F | G | H | J | K | L |          |  <-- [A] = hovered
|   +---+---+---+---+---+---+---+---+---+          |
|     +---+---+---+---+---+---+---+------+          |
|     | Z | X | C | V | B | N | M | BKSP |          |
|     +---+---+---+---+---+---+---+------+          |
|     +----------------------------------+          |
|     |             SPACE                |          |
|     +----------------------------------+          |
|                                                   |
| Pinch dist: 32px                                  |  <-- Debug info
+--------------------------------------------------+
```

---

## Tech Stack

| Technology | Version | Role |
|---|---|---|
| Python | 3.8+ | Core language |
| OpenCV | 4.8+ | Camera + UI rendering |
| MediaPipe | 0.10.14+ | Hand landmark detection (Tasks API) |
| pyautogui | 0.9.54+ | OS mouse/keyboard control |
| NumPy | 1.24+ | Math operations |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Gray padding around webcam | Fixed — DPI awareness is set automatically on Windows |
| Webcam not opening | Check camera connection, try changing `cv2.VideoCapture(0)` to `(1)` |
| Keys not typing | Make sure the target app (Notepad, etc.) has OS focus, not the AirPointer window |
| Gestures not detected | Ensure good lighting, keep hand 30-60 cm from camera |
| Model download fails | Check internet connection — the model downloads once on first run |

---

## License

This project is open source and available for educational purposes.

---

<p align="center">
  Built by <a href="https://github.com/aryanchauhan07">Aryan Chauhan</a>
</p>
