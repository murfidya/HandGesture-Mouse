# GestureMouse

Control your mouse cursor with hand gestures using your webcam — no extra hardware needed.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)

---

## Features

- 🖱️ **Move cursor** — move your wrist in front of the camera
- 👆 **Left click** — pinch Thumb + Index finger
- ✌️ **Right click** — pinch Thumb + Middle finger
- 🔄 **Scroll** — pinch Thumb + Pinky finger and move hand up/down
- 📷 **Always-on-top camera window** — native floating preview stays visible even when the browser is minimized
- 🌐 **Web dashboard** — real-time gesture indicators, FPS counter, and adjustable settings
- ⚡ **60 FPS cursor** — dedicated cursor thread runs at 60 Hz, decoupled from camera frame rate
- 🎛️ **One Euro Filter** — adaptive smoothing that reduces jitter while keeping responsiveness
- 🎯 **Sensitivity control** — adjustable cursor sensitivity via dashboard

---

## Requirements

- Python 3.9+
- Webcam
- Windows (tested on Windows 11)

> **Note:** `tkinter` is included with the standard Python installer on Windows. If it's missing, reinstall Python and make sure "tcl/tk and IDLE" is checked.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/murfidya/HandGesture-Mouse.git
cd HandGesture-Mouse

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The MediaPipe hand landmark model (`hand_landmarker.task`, ~7 MB) is **downloaded automatically** on first run.

---

## Usage

```bash
python app.py
```

- A floating camera preview window will appear (bottom-right, always-on-top)
- Your browser will open the dashboard at `http://localhost:8765`
- Move your hand in front of the camera to start controlling the mouse

To launch without auto-opening the browser:

```bash
python app.py --no-browser
```

### Stopping

Press `Escape`, close the camera window, or press `Ctrl+C` in the terminal.

---

## Gesture Reference

| Gesture | Action |
|---|---|
| Move wrist | Move cursor |
| Thumb + Index pinch | Left click (hold = drag) |
| Thumb + Middle pinch | Right click |
| Thumb + Pinky pinch + move up/down | Scroll |

---

## Settings (via dashboard)

| Setting | Description | Default |
|---|---|---|
| Pinch Threshold | How close fingers must be to trigger a pinch | 0.05 |
| Smoothing | Controls the One Euro Filter cutoff (lower = smoother) | 0.30 |
| Sensitivity | Cursor movement multiplier (higher = faster) | 1.00 |

---

## Architecture

```
Main thread      → tkinter always-on-top camera window
Thread-1         → hand_detection_thread (OpenCV + MediaPipe Tasks API)
Thread-2         → cursor_update_thread (60 Hz One Euro Filter + pyautogui)
Thread-3         → aiohttp HTTP + WebSocket server
Thread-4         → browser auto-open
```

Detection runs entirely in Python — the browser is only used as a settings dashboard and is **not required** for the mouse control to work.

---

## Dependencies

| Package | Purpose |
|---|---|
| `mediapipe` | Hand landmark detection |
| `opencv-python` | Camera capture & frame processing |
| `pyautogui` | Mouse control |
| `aiohttp` | HTTP + WebSocket server |
| `Pillow` | Frame rendering in tkinter window |
| `tkinter` | Always-on-top native camera window (stdlib) |

---

## What's New (v1.1.0)

- **One Euro Filter** — replaced velocity-dampened smoothing with adaptive jitter reduction
- **60 FPS cursor thread** — cursor updates at 60 Hz, independent of camera frame rate
- **Sensitivity slider** — adjust cursor speed from the dashboard
- **Dashboard redesign** — centered card-grid layout with glassmorphism, animations, and Inter font
- **`--no-browser` flag** — launch without auto-opening the dashboard
- **Mouse-up fix** — mouse button releases correctly when hand leaves the frame mid-click
- **Camera failure handling** — graceful exit on webcam disconnection
- **Clean exit** — press Escape or close the camera window to quit

---

## License

MIT
